package main

import (
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/electronstudio/low_latency_dictation/audio"
	"github.com/electronstudio/low_latency_dictation/timestamp"
	"github.com/electronstudio/low_latency_dictation/transcribe"
	"github.com/electronstudio/low_latency_dictation/vad"
	"github.com/gdamore/tcell/v2"

	_ "github.com/gdamore/tcell/v2/encoding"
)

type whisperParams struct {
	nThreads     int
	lengthMs     int
	keepMs       int
	captureID    int
	maxTokens    int
	audioCtx     int
	beamSize     int
	vadThold     float32
	freqThold    float32
	noFallback   bool
	printSpecial bool
	noContext    bool
	noTimestamps bool
	useGPU       bool
	flashAttn    bool
	language     string
	model        string
}

// Segment represents a single transcribed segment with its text and time range.
type Segment struct {
	Text  string
	Start int64
	End   int64
}

// String returns the segment formatted as "[t0 --> t1]  text".
func (s Segment) String() string {
	return fmt.Sprintf("[%s --> %s]  %s",
		timestamp.ToTimestamp(s.Start, false),
		timestamp.ToTimestamp(s.End, false),
		s.Text)
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
	debug        = false
	screen       tcell.Screen
	screenHeight int
	statusStyle  tcell.Style = tcell.StyleDefault.Reverse(true)
)

func main() {
	transcribe.BackendLoadAll()

	params := whisperParams{
		nThreads:     min(1, runtime.NumCPU()),
		lengthMs:     30000,
		keepMs:       200,
		captureID:    -1,
		maxTokens:    32,
		audioCtx:     0,
		beamSize:     -1,
		vadThold:     0.8,
		freqThold:    100.0,
		noFallback:   false,
		printSpecial: false,
		noContext:    true,
		noTimestamps: false,
		useGPU:       true,
		flashAttn:    true,
		language:     "en",
		model:        "models/ggml-tiny.en-q8_0.bin",
	}
	strategy := transcribe.SamplingGreedy
	if params.beamSize > 1 {
		strategy = transcribe.SamplingBeamSearch
	}

	wparams := transcribe.FullParams{
		Strategy:       strategy,
		PrintProgress:  false,
		PrintSpecial:   params.printSpecial,
		PrintRealtime:  false,
		PrintTimestamp: !params.noTimestamps,
		Translate:      false,
		SingleSegment:  false,
		MaxTokens:      params.maxTokens,
		Language:       params.language,
		NThreads:       params.nThreads,
		AudioCtx:       params.audioCtx,
		TdrzEnable:     false,
		NoFallback:     params.noFallback,
		SuppressNST:    true,
		BeamSize:       params.beamSize,
	}

	// init tcell screen
	tcell.SetEncodingFallback(tcell.EncodingFallbackASCII)
	var err error
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
	//die(0, "FOO "+strconv.Itoa(screenHeight))
	printStatus("LOADING")
	screen.Show()

	// init audio
	mic := audio.NewAudioAsync(params.lengthMs)
	if err := mic.Init(params.captureID, transcribe.WhisperSampleRate); err != nil {
		die(1, "main: audio.Init() failed: %v\n", err)
	}
	defer mic.Close()

	if err := mic.Resume(); err != nil {
		die(1, "main: audio.Resume() failed: %v\n", err)
	}

	// whisper init
	cp := transcribe.ContextParams{
		UseGPU:    params.useGPU,
		FlashAttn: params.flashAttn,
	}
	ctx, err := transcribe.InitFromFile(params.model, cp)
	if err != nil {
		die(2, "error: %v\n", err)
	}
	defer ctx.Free()

	m := "models/ggml-base.en.bin"
	ctx2, err := transcribe.InitFromFile(m, cp)
	if err != nil {
		die(2, "error: %v\n", err)
	}
	defer ctx2.Free()

	nIter := 0

	tStart := time.Now()
	tLast := tStart

	// Handle Ctrl+C gracefully
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	isRunning := true

	// segments collects all transcribed segments across the session.
	var segments []Segment

	// Poll SDL events in a goroutine so the main loop can respond to both
	// SDL quit events and OS signals.
	go func() {
		for isRunning {
			if !audio.PollEvents() {
				isRunning = false
				return
			}
			time.Sleep(10 * time.Millisecond)
		}
	}()

	// Event polling goroutine for tcell (quit / resize)
	go func() {
		for isRunning {
			ev := screen.PollEvent()
			switch ev.(type) {
			case *tcell.EventKey:
				isRunning = false
				return
			}
		}
	}()

	cursorY := screenHeight - 1 // draw from bottom up

	printStatus("LISTENING")
	screen.Show()

mainloop:
	for isRunning {
		select {
		case <-sigCh:
			isRunning = false
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

		if vad.SimpleVAD(pcmf32New, transcribe.WhisperSampleRate, 250, params.vadThold, params.freqThold, false) {
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

		// run the inference
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

			seg := Segment{
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
		// if debug {
		// 	debugMsg := fmt.Sprintf("### Transcription %d START | t0 = %d ms | t1 = %d ms", nIter, transcriptionStart, transcriptionEnd)
		// 	// print debug at top if room
		// 	if cursorY > 0 {
		// 		printToScreen(0, 0, tcell.StyleDefault, debugMsg)
		// 	}
		// }

		printStatus("LISTENING")
		screen.Show()
		nIter++
	}

	mic.Pause()
	fullAudio := mic.GetFullAudio()
	if len(fullAudio) > 0 {
		if err := ctx2.Full(wparams, fullAudio); err != nil {
			die(6, "main: failed to process audio: %v\n", err)
		}
	}

	// final print
	screen.Clear()
	_, finalH := screen.Size()
	cursorY = finalH - 1
	nSegments := ctx2.NSegments()
	for i := 0; i < nSegments; i++ {
		text := ctx2.SegmentText(i)
		printToScreen(0, cursorY, tcell.StyleDefault, text)
		cursorY--
		if cursorY < 0 {
			break
		}
	}

}
