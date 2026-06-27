package main

import (
	"encoding/base64"
	"fmt"
	"log"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/alexflint/go-arg"
	"github.com/electronstudio/low_latency_dictation/audio"
	"github.com/electronstudio/low_latency_dictation/hotkey"
	"github.com/electronstudio/low_latency_dictation/transcribe"
	"github.com/electronstudio/low_latency_dictation/typing"
	"github.com/electronstudio/low_latency_dictation/vad"
	"github.com/gdamore/tcell/v2"
	"golang.design/x/clipboard"
	"golang.design/x/hotkey/mainthread"
)

type CLI struct {
	Model      string  `arg:"-m,--model"       default:"ggml-tiny.en-q8_0.bin" help:"Model for real-time transcription, e.g. ggml-medium-q5_0.bin "`
	FinalModel string  `arg:"-f,--final-model" default:"ggml-base.en.bin"      help:"Model for finalization, e.g ggml-large-v3-turbo-q5_0.bin, none to disable"`
	Threads    int     `arg:"-t,--threads"     default:"0"                     help:"Threads (0=auto)"`
	UseCPU     bool    `arg:"--use-cpu"        default:"false"                  help:"Disable GPU accleration"`
	CaptureID  int     `arg:"-a,--audio-device"     default:"-1"               help:"Audio device ID"`
	LengthMs   int     `arg:"-l,--length"      default:"30000"                 help:"(ADVANCED: Buffer length in ms)"`
	MaxTokens  int     `arg:"--max-tokens"     default:"32"                    help:"(ADVANCED: Max tokens per segment)"`
	AudioCtx   int     `arg:"--audio-ctx"      default:"0"                     help:"(ADVANCED: Audio context size)"`
	VadThold   float32 `arg:"--vad-thold"      default:"0.8"                   help:"(ADVANCED: VAD threshold)"`
	FreqThold  float32 `arg:"--freq-thold"     default:"100.0"                 help:"(ADVANCED: High-pass filter cutoff)"`
	Language   string  `arg:"--lang"           default:"en"                    help:"(ADVANCED: Language code)"`
	FlashAttn  bool    `arg:"--flash-attn"     default:"true"                  help:"(ADVANCED: Use flash attention)"`
	LogFile    string  `arg:"--log-file"       default:""                      help:"Path to log file for actions"`
	LogLevel   string  `arg:"--log-level"      default:"warn"                   help:"whisper.cpp console log level (debug/info/warn/error/none)"`
	HotkeyMods string  `arg:"--hotkey-mods"    default:"ctrl+shift"            help:"Modifiers for the global stop hotkey (ctrl/alt/shift/cmd|win|super, joined by +)"`
	HotkeyKey  string  `arg:"--hotkey-key"     default:"d"                     help:"Key for the global stop hotkey (e.g. d, f1, space, escape)"`
}

// TODO: remove LEngthMS?
// Add final threads and final use cpu
// display final segment even it overlapped and couldnt be added, because missing end words is annoying
// do whole buffer rather than circular buffer, or at least larger circle
// option to adjust interval
// option to disable VAD
// recommended options for high, low systems
// escape to delete, escape again to quit. C to copy.  enter to exit and copy.
// colors?
// word timings?
// better models?
// resident daemon?
// split front and backend.  network?
// suggested improvements, concurrency bug,

func (CLI) Version() string {
	return "low_latency_dictation 0.1.0"
}

type whisperParams struct {
	nThreads  int
	lengthMs  int
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
	s := "[" + status + "] (press " + hotkeyLabel + " to stop)"
	if screenWidth > len(s) {
		s += strings.Repeat(" ", screenWidth-len(s))
	}
	printToScreen(0, screenHeight-1, statusStyle, s)
}

func die(code int, format string, args ...interface{}) {
	if screen != nil {
		screen.Fini()
	}
	fmt.Fprintf(os.Stderr, format, args...)
	os.Exit(code)
}

var (
	screen       tcell.Screen
	screenWidth  int
	screenHeight int
	statusStyle  = tcell.StyleDefault.Reverse(true)
	logger       *log.Logger

	// hotkeyLabel is the human-readable stop key combo shown in the status
	// line. Defaults to "any key" (the foreground terminal key) and is
	// replaced by the registered global combo when registration succeeds.
	hotkeyLabel = "any key"
)

func logActionf(format string, args ...interface{}) {
	if logger != nil {
		logger.Printf(format, args...)
	}
}

// makeWhisperSink builds the sink handed to transcribe.SetLogSink. When a log
// file is configured, whisper.cpp lines are written through the same *log.Logger
// as the app's own actions (with a timestamp, trailing newline trimmed). With no
// log file, lines go raw to os.Stderr, preserving whisper.cpp's exact format.
func makeWhisperSink(lg *log.Logger) func(transcribe.LogLevel, string) {
	if lg != nil {
		return func(_ transcribe.LogLevel, text string) {
			lg.Print(strings.TrimRight(text, "\n"))
		}
	}
	return func(_ transcribe.LogLevel, text string) {
		os.Stderr.WriteString(text)
	}
}

// main runs the program body. On macOS, mainthread.Init hands the real main
// thread to the NSApplication event loop (required by the global-hotkey
// backend, which delivers events via a CGEventTap on the main run loop) and
// runs run() in a goroutine. On Linux/Windows the hotkey backends manage
// their own threads, so Init's main-thread dispatch loop sits idle; using the
// same entry point on every platform keeps the code paths identical.
func main() { mainthread.Init(run) }

// run is the program body. It is invoked from a goroutine by main() on every
// platform.
func run() {
	var cli CLI
	arg.MustParse(&cli)

	if cli.LogFile != "" {
		f, err := os.OpenFile(cli.LogFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: failed to open log file: %v\n", err)
			os.Exit(1)
		}
		defer f.Close()
		logger = log.New(f, "", log.LstdFlags)
	}

	// Install the whisper.cpp log callback before any whisper/ggml call so
	// the level filter applies to backend discovery and model-load output.
	transcribe.SetLogLevel(transcribe.ParseLogLevel(cli.LogLevel))
	transcribe.SetLogSink(makeWhisperSink(logger))
	transcribe.InstallLogCallback()

	// Initialize the system clipboard backend. If it is unavailable (e.g. no
	// Wayland data-control manager and no X server), keep running: the
	// transcription is still printed and offered via /dev/clipboard and OSC 52,
	// but the automatic paste into the focused app is skipped.
	clipErr := clipboard.Init()
	if clipErr != nil {
		fmt.Fprintf(os.Stderr, "warning: clipboard unavailable: %v\n", clipErr)
		logActionf("clipboard init failed: %v", clipErr)
	}

	transcribe.BackendLoadAll()

	logActionf("=== startup ===")
	logActionf("model=%s final_model=%s threads=%d use_cpu=%v audio_device=%d length_ms=%d max_tokens=%d audio_ctx=%d vad_thold=%.2f freq_thold=%.1f lang=%s flash_attn=%v",
		cli.Model, cli.FinalModel, cli.Threads, cli.UseCPU, cli.CaptureID, cli.LengthMs,
		cli.MaxTokens, cli.AudioCtx, cli.VadThold, cli.FreqThold,
		cli.Language, cli.FlashAttn)

	params := whisperParams{
		lengthMs:  cli.LengthMs,
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
		logActionf("audio init failed: %v", err)
		die(1, "main: audio.Init() failed: %v\n", err)
	}
	defer mic.Close()
	logActionf("audio init ok device=%d", params.captureID)

	if err := mic.Resume(); err != nil {
		logActionf("audio resume failed: %v", err)
		die(1, "main: audio.Resume() failed: %v\n", err)
	}
	logActionf("audio resume ok")

	cp := transcribe.ContextParams{UseGPU: !cli.UseCPU, FlashAttn: cli.FlashAttn}

	ctxModelPath, err := resolveModelFile(params.model)
	if err != nil {
		logActionf("model load failed: %v", err)
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	ctx, err := transcribe.InitFromFile(ctxModelPath, cp)
	if err != nil {
		logActionf("context init failed: %v", err)
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	defer ctx.Free()
	logActionf("context init ok model=%s", ctxModelPath)

	var ctx2 *transcribe.Context

	if cli.FinalModel != "none" {
		ctx2ModelPath, err := resolveModelFile(cli.FinalModel)
		if err != nil {
			logActionf("final model load failed: %v", err)
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		ctx2, err = transcribe.InitFromFile(ctx2ModelPath, cp)
		if err != nil {
			logActionf("final context init failed: %v", err)
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		defer ctx2.Free()
		logActionf("final context init ok model=%s", ctx2ModelPath)
	}

	tStart := time.Now()
	tLast := tStart

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	var isRunning atomic.Bool
	isRunning.Store(true)

	// Register the global stop hotkey. On failure we warn and continue: the
	// foreground terminal key (handled below) still stops the app, so this
	// never regresses existing users who lack permissions or the platform
	// support.
	if hkMods, hkKey, perr := hotkey.ParseCombo(cli.HotkeyMods, cli.HotkeyKey); perr != nil {
		fmt.Fprintf(os.Stderr, "warning: invalid --hotkey/--key (%q+%q): %v\n", cli.HotkeyMods, cli.HotkeyKey, perr)
		fmt.Fprintf(os.Stderr, "warning: global hotkey disabled; stop with any terminal key instead.\n")
		logActionf("hotkey parse failed: %v", perr)
	} else if stopHK, rerr := hotkey.Register(hkMods, hkKey); rerr != nil {
		fmt.Fprintf(os.Stderr, "warning: could not register global hotkey %s+%s: %v\n", cli.HotkeyMods, cli.HotkeyKey, rerr)
		fmt.Fprintf(os.Stderr, "warning: stop with any terminal key instead.\n")
		logActionf("hotkey register failed: %v", rerr)
	} else {
		hotkeyLabel = stopHK.String()
		logActionf("hotkey registered: %s", hotkeyLabel)
		go func() {
			<-stopHK.Keydown()
			logActionf("hotkey triggered, stopping")
			isRunning.Store(false)
		}()
	}

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
			//logActionf("status= sleep")
			time.Sleep(16 * time.Millisecond)
			continue
		}
		printStatus("NOT SLEEP")
		screen.Show()
		logActionf("vad activated")
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
			logActionf("segment t0=%d t1=%d text=%q", t0, t1, text)

			for j, existing := range segments {
				if existing.Start >= seg.Start-100 && existing.Start <= seg.Start+100 {
					logActionf("truncating segments to %d because of %q", j, seg.Text)
					segments = segments[:j]
					break
				}
			}
			if len(segments) == 0 || segments[len(segments)-1].End <= seg.Start {
				logActionf("appending segment %q", seg.Text)
				segments = append(segments, seg)
			}
		}

		// redraw all segments on screen
		screen.Clear()
		var sb strings.Builder
		for _, segment := range segments {
			sb.WriteString(segment.Text)
		}
		printWrapped(0, 0, screenWidth, screenHeight-1, tcell.StyleDefault, strings.TrimSpace(sb.String()))

		printStatus("LISTENING")
		screen.Show()
	}

	printStatus("DOING FINAL TRANSCRIBE...")
	screen.Show()

	mic.Pause()
	var finalText string
	if ctx2 != nil {
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
		finalText = strings.TrimSpace(sb.String())
		logActionf("final transcription n_segments=%d text=%q", nSegments, finalText)
	} else {
		screen.Fini()
		var sb strings.Builder
		for _, segment := range segments {
			sb.WriteString(segment.Text)
		}
		finalText = strings.TrimSpace(sb.String())
		logActionf("final text from segments text=%q", finalText)
	}
	fmt.Print(finalText)

	// Offer the transcription through every available channel. The WSL/Cygwin
	// /dev/clipboard and the OSC 52 terminal escape are environment-specific
	// fallbacks that do not depend on the clipboard package; they always run.
	if f, err := os.OpenFile("/dev/clipboard", os.O_WRONLY|os.O_TRUNC, 0); err == nil {
		_, _ = f.WriteString(finalText)
		_ = f.Close()
	}
	data := base64.StdEncoding.EncodeToString([]byte(finalText))
	if clipErr == nil {
		clipboard.Write(clipboard.FmtText, []byte(finalText))
	}

	var seq string
	if os.Getenv("TMUX") != "" {
		seq = fmt.Sprintf("\033Ptmux;\033\033]52;c;%s\007\033\\", data)
	} else if strings.HasPrefix(os.Getenv("TERM"), "screen") {
		seq = fmt.Sprintf("\033P\033]52;c;%s\007\033\\", data)
	} else {
		seq = fmt.Sprintf("\033]52;c;%s\007", data)
	}
	fmt.Print(seq)

	// simulate a paste (Ctrl+V on
	// Linux/Windows, Cmd+V on macOS) into the focused application. On Linux the
	// clipboard content is served from this process (X11 selection ownership or
	// a Wayland data-source), so we linger briefly afterwards to let the target
	// application read it; macOS and Windows copy into the OS clipboard and the
	// keystroke is already queued, so no linger is needed there.

	if err := typing.Paste(); err != nil {
		fmt.Fprintf(os.Stderr, "warning: could not auto-paste: %v\n", err)
		fmt.Fprintf(os.Stderr, "warning: text is on the clipboard; paste manually with Ctrl/Cmd+V\n")
		logActionf("paste failed: %v", err)
	} else {
		logActionf("paste ok")
		if runtime.GOOS == "linux" {
			time.Sleep(200 * time.Millisecond)
		}
	}

	os.Exit(0)
}

type segment struct {
	Text  string
	Start int64
	End   int64
}
