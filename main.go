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

	"github.com/atotto/clipboard"
	"github.com/electronstudio/low_latency_dictation/audio"
	"github.com/electronstudio/low_latency_dictation/transcribe"
	"github.com/electronstudio/low_latency_dictation/vad"
	"github.com/gdamore/tcell/v2"
)

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
	screenHeight int
	statusStyle  = tcell.StyleDefault.Reverse(true)
)

func main() {
	transcribe.BackendLoadAll()

	params := whisperParams{
		nThreads:  min(1, runtime.NumCPU()),
		lengthMs:  30000,
		keepMs:    200,
		captureID: -1,
		maxTokens: 32,
		audioCtx:  0,
		vadThold:  0.8,
		freqThold: 100.0,
		language:  "en",
		model:     "ggml-tiny.en-q8_0.bin",
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

	cp := transcribe.ContextParams{UseGPU: true, FlashAttn: true}

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

	m := "ggml-base.en.bin"
	ctx2ModelPath, err := resolveModelFile(m)
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
	_, screenHeight = screen.Size()

	cursorY := screenHeight - 1 // draw from bottom up

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

		if tDiff < 200 {
			time.Sleep(16 * time.Millisecond)
			continue
		}

		pcmf32New := mic.Get(500)

		// SimpleVAD returns true when SILENCE is detected (i.e. speech stopped).
		if !vad.SimpleVAD(pcmf32New, transcribe.WhisperSampleRate, 250, params.vadThold, params.freqThold, false) {
			pcmf32New = mic.Get(params.lengthMs)
		} else {
			time.Sleep(16 * time.Millisecond)
			continue
		}
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

		// redraw all segments on screen (from bottom up)
		screen.Clear()
		_, curH := screen.Size()
		cursorY = curH - 1
		for _, seg := range segments {
			line := seg.Text
			printToScreen(0, 0, tcell.StyleDefault, line)
			cursorY--
			if cursorY < 0 {
				break
			}
		}

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
