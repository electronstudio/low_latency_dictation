package main

import (
	"encoding/base64"
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/alexflint/go-arg"
	"github.com/atotto/clipboard"
	"github.com/electronstudio/low_latency_dictation/audio"
	"github.com/electronstudio/low_latency_dictation/transcribe"
	"github.com/electronstudio/low_latency_dictation/vad"
	"github.com/gdamore/tcell/v2"
)

type CLI struct {
	Model      string  `arg:"-m,--model"       default:"ggml-tiny.en-q8_0.bin" help:"Model for real-time transcription"`
	FinalModel string  `arg:"-f,--final-model" default:"ggml-base.en.bin"      help:"Model for final transcription"`
	Threads    int     `arg:"-t,--threads"     default:"0"                     help:"Threads (0=auto)"`
	UseCPU     bool    `arg:"--use-cpu"        default:"true"                  help:"Disable GPU accleration"`
	CaptureID  int     `arg:"-a,--audio-device"     default:"-1"               help:"Audio device ID"`
	LengthMs   int     `arg:"-l,--length"      default:"30000"                 help:"(ADVANCED: Buffer length in ms)"`
	KeepMs     int     `arg:"-k,--keep"        default:"200"                   help:"(ADVANCED: Keep from previous chunk (ms))"`
	MaxTokens  int     `arg:"--max-tokens"     default:"32"                    help:"(ADVANCED: Max tokens per segment)"`
	AudioCtx   int     `arg:"--audio-ctx"      default:"0"                     help:"(ADVANCED: Audio context size)"`
	VadThold   float32 `arg:"--vad-thold"      default:"0.8"                   help:"(ADVANCED: VAD threshold)"`
	FreqThold  float32 `arg:"--freq-thold"     default:"100.0"                 help:"(ADVANCED: High-pass filter cutoff)"`
	Language   string  `arg:"--lang"           default:"en"                    help:"(ADVANCED: Language code)"`
	FlashAttn  bool    `arg:"--flash-attn"     default:"true"                  help:"(ADVANCED: Use flash attention)"`
}

func (CLI) Version() string {
	return "low_latency_dictation 0.1.0"
}

type whisperParams struct {
	nThreads  int
	lengthMs  int
	keepMs    int
	captureID int
	maxTokens int
	audioCtx  int
	vadThold  float32
	freqThold float32
	language  string
	model     string
}

func printToScreen(x, y int, style tcell.Style, text string) {
	for i, r := range text {
		screen.SetContent(x+i, y, r, nil, style)
	}
}

func printWrapped(x, y, maxWidth, maxLines int, style tcell.Style, text string) {
	if maxWidth <= 0 || maxLines <= 0 {
		return
	}

	var lines []string
	for len(text) > 0 {
		if len(text) <= maxWidth {
			lines = append(lines, text)
			break
		}

		cut := maxWidth
		for i := maxWidth; i > 0; i-- {
			if text[i] == ' ' {
				cut = i
				break
			}
		}
		// If there is no space, break hard.
		if cut == 0 {
			cut = maxWidth
		}
		lines = append(lines, text[:cut])
		if text[cut] == ' ' {
			text = text[cut+1:]
		} else {
			text = text[cut:]
		}
	}

	if len(lines) > maxLines {
		lines = lines[len(lines)-maxLines:]
	}
	for i, line := range lines {
		printToScreen(x, y+i, style, line)
	}
}

func printStatus(status string) {
	if screenHeight < 1 {
		return
	}
	printToScreen(0, screenHeight-1, statusStyle, "["+status+"] (press any key to quit)")
}

func die(code int, format string, args ...interface{}) {
	screen.Fini()
	fmt.Fprintf(os.Stderr, format, args...)
	os.Exit(code)
}

var (
	screen       tcell.Screen
	screenWidth  int
	screenHeight int
	statusStyle  = tcell.StyleDefault.Reverse(true)
)

func main() {
	transcribe.BackendLoadAll()

	var cli CLI
	arg.MustParse(&cli)

	params := whisperParams{
		lengthMs:  cli.LengthMs,
		keepMs:    cli.KeepMs,
		captureID: cli.CaptureID,
		maxTokens: cli.MaxTokens,
		audioCtx:  cli.AudioCtx,
		vadThold:  cli.VadThold,
		freqThold: cli.FreqThold,
		language:  cli.Language,
		model:     cli.Model,
	}

	if cli.Threads == 0 {
		if cli.UseCPU {
			params.nThreads = runtime.NumCPU()
		} else {
			params.nThreads = 1
		}
	} else {
		params.nThreads = cli.Threads
	}

	wparams := transcribe.FullParams{
		Strategy:       transcribe.SamplingGreedy,
		PrintProgress:  false,
		PrintSpecial:   false,
		PrintRealtime:  false,
		PrintTimestamp: true,
		Translate:      false,
		SingleSegment:  false,
		MaxTokens:      params.maxTokens,
		Language:       params.language,
		NThreads:       params.nThreads,
		AudioCtx:       params.audioCtx,
		TdrzEnable:     false,
		NoFallback:     false,
		SuppressNST:    true,
		BeamSize:       -1,
	}

	mic := audio.NewAudioAsync(params.lengthMs)
	if err := mic.Init(params.captureID, transcribe.WhisperSampleRate); err != nil {
		die(1, "main: audio.Init() failed: %v\n", err)
	}
	defer mic.Close()

	if err := mic.Resume(); err != nil {
		die(1, "main: audio.Resume() failed: %v\n", err)
	}

	cp := transcribe.ContextParams{UseGPU: !cli.UseCPU, FlashAttn: cli.FlashAttn}

	ctxModelPath, err := resolveModelFile(params.model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	ctx, err := transcribe.InitFromFile(ctxModelPath, cp)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	defer ctx.Free()

	ctx2ModelPath, err := resolveModelFile(cli.FinalModel)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	ctx2, err := transcribe.InitFromFile(ctx2ModelPath, cp)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	defer ctx2.Free()

	tStart := time.Now()
	tLast := tStart

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	var isRunning atomic.Bool
	isRunning.Store(true)

	// Poll SDL events in a goroutine so the main loop can respond to both
	// SDL quit events and OS signals.
	go func() {
		for isRunning.Load() {
			if !audio.PollEvents() {
				isRunning.Store(false)
				return
			}
			time.Sleep(10 * time.Millisecond)
		}
	}()

	tcell.SetEncodingFallback(tcell.EncodingFallbackASCII)

	screen, err = tcell.NewScreen()
	if err != nil {
		fmt.Fprintf(os.Stderr, "main: failed to create tcell screen: %v\n", err)
		os.Exit(1)
	}
	if err := screen.Init(); err != nil {
		fmt.Fprintf(os.Stderr, "main: failed to init tcell screen: %v\n", err)
		os.Exit(1)
	}
	defer screen.Fini()
	screen.SetStyle(tcell.StyleDefault)
	screen.Clear()
	screenWidth, screenHeight = screen.Size()

	// Event polling goroutine for tcell (quit / resize)
	go func() {
		for isRunning.Load() {
			ev := screen.PollEvent()
			switch ev.(type) {
			case *tcell.EventKey:
				isRunning.Store(false)
				return
			}
		}
	}()

	printStatus("LISTENING")
	screen.Show()

	var segments []segment

mainloop:
	for isRunning.Load() {
		select {
		case <-sigCh:
			break mainloop
		default:
		}

		tNow := time.Now()
		tDiff := tNow.Sub(tLast).Milliseconds()

		if tDiff < 100 {
			time.Sleep(16 * time.Millisecond)
			continue
		}
		ls := mic.Get(500)

		if !vad.SimpleVAD(ls, transcribe.WhisperSampleRate, 250, params.vadThold, params.freqThold, false) {
			printStatus("SLEEP")
			screen.Show()
			time.Sleep(16 * time.Millisecond)
			continue
		}
		printStatus("NOT SLEEP")
		screen.Show()
		pcmf32New := mic.Get(params.lengthMs)
		tLast = tNow

		if len(pcmf32New) == 0 {
			time.Sleep(16 * time.Millisecond)
			continue
		}

		if err := ctx.Full(wparams, pcmf32New); err != nil {
			die(6, "main: failed to process audio: %v\n", err)
		}

		transcriptionEnd := tLast.Sub(tStart).Milliseconds()
		transcriptionStart := max(0, int(transcriptionEnd)-len(pcmf32New)*1000/transcribe.WhisperSampleRate)

		nSegments := ctx.NSegments()
		for i := 0; i < nSegments; i++ {
			text := ctx.SegmentText(i)
			t0 := ctx.SegmentT0(i) + int64(transcriptionStart)/10
			t1 := ctx.SegmentT1(i) + int64(transcriptionStart)/10

			seg := segment{
				Text:  text,
				Start: t0,
				End:   t1,
			}

			for j, existing := range segments {
				if existing.Start >= seg.Start-100 && existing.Start <= seg.Start+100 {
					segments = segments[:j]
					break
				}
			}
			if len(segments) == 0 || segments[len(segments)-1].End <= seg.Start {
				segments = append(segments, seg)
			}
		}

		// redraw all segments on screen
		screen.Clear()
		var sb strings.Builder
		for i := 0; i < nSegments; i++ {
			sb.WriteString(ctx.SegmentText(i))
		}
		printWrapped(0, 0, screenWidth, screenHeight-1, tcell.StyleDefault, sb.String())

		printStatus("LISTENING")
		screen.Show()
	}

	printStatus("DOING FINAL TRANSCRIBE...")
	screen.Show()

	mic.Pause()
	fullAudio := mic.GetFullAudio()
	if len(fullAudio) > 0 {
		if err := ctx2.Full(wparams, fullAudio); err != nil {
			die(6, "main: failed to process audio: %v\n", err)
		}
	}

	screen.Fini()
	nSegments := ctx2.NSegments()
	var sb strings.Builder
	for i := 0; i < nSegments; i++ {
		sb.WriteString(ctx2.SegmentText(i))
	}
	fmt.Print(sb.String())

	// Method 1: atotto/clipboard
	_ = clipboard.WriteAll(sb.String())

	// Method 2: /dev/clipboard (WSL/Cygwin)
	if f, err := os.OpenFile("/dev/clipboard", os.O_WRONLY|os.O_TRUNC, 0); err == nil {
		_, _ = f.WriteString(sb.String())
		_ = f.Close()
	}

	// Method 3: OSC 52 terminal escape sequence
	data := base64.StdEncoding.EncodeToString([]byte(sb.String()))
	var seq string
	if os.Getenv("TMUX") != "" {
		seq = fmt.Sprintf("\033Ptmux;\033\033]52;c;%s\007\033\\", data)
	} else if strings.HasPrefix(os.Getenv("TERM"), "screen") {
		seq = fmt.Sprintf("\033P\033]52;c;%s\007\033\\", data)
	} else {
		seq = fmt.Sprintf("\033]52;c;%s\007", data)
	}
	fmt.Print(seq)

	os.Exit(0)
}

type segment struct {
	Text  string
	Start int64
	End   int64
}
