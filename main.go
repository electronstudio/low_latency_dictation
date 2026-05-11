package main

import (
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/user/dictation/audio"
	"github.com/user/dictation/timestamp"
	"github.com/user/dictation/transcribe"
	"github.com/user/dictation/vad"
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

	// init audio
	mic := audio.NewAudioAsync(params.lengthMs)
	if err := mic.Init(params.captureID, transcribe.WhisperSampleRate); err != nil {
		fmt.Fprintf(os.Stderr, "main: audio.Init() failed: %v\n", err)
		os.Exit(1)
	}
	defer mic.Close()

	if err := mic.Resume(); err != nil {
		fmt.Fprintf(os.Stderr, "main: audio.Resume() failed: %v\n", err)
		os.Exit(1)
	}

	// whisper init
	cp := transcribe.ContextParams{
		UseGPU:    params.useGPU,
		FlashAttn: params.flashAttn,
	}
	ctx, err := transcribe.InitFromFile(params.model, cp)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(2)
	}
	defer ctx.Free()

	nIter := 0

	fmt.Println("[Start speaking]")
	os.Stdout.Sync()

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

		if tDiff < 500 {
			time.Sleep(33 * time.Millisecond)
			continue
		}

		pcmf32New := mic.Get(500)

		if vad.SimpleVAD(pcmf32New, transcribe.WhisperSampleRate, 250, params.vadThold, params.freqThold, false) {
			pcmf32New = mic.Get(params.lengthMs)
		} else {
			time.Sleep(33 * time.Millisecond)
			continue
		}
		tLast = tNow

		// run the inference
		strategy := transcribe.SamplingGreedy
		if params.beamSize > 1 {
			strategy = transcribe.SamplingBeamSearch
		}

		wparams := transcribe.FullParams{
			Strategy:       strategy,
			PrintProgress:  true,
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

		if err := ctx.Full(wparams, pcmf32New); err != nil {
			fmt.Fprintf(os.Stderr, "main: failed to process audio: %v\n", err)
			os.Exit(6)
		}

		transcriptionEnd := tLast.Sub(tStart).Milliseconds()
		transcriptionStart := max(0, transcriptionEnd-int64(len(pcmf32New))*1000/int64(transcribe.WhisperSampleRate))

		fmt.Println()
		fmt.Printf("### Transcription %d START | t0 = %d ms | t1 = %d ms\n", nIter, transcriptionStart, transcriptionEnd)
		fmt.Println()

		nSegments := ctx.NSegments()
		for i := 0; i < nSegments; i++ {
			text := ctx.SegmentText(i)
			t0 := ctx.SegmentT0(i) + transcriptionStart/10
			t1 := ctx.SegmentT1(i) + transcriptionStart/10

			seg := Segment{
				Text:  text,
				Start: t0,
				End:   t1,
			}

			for j, existing := range segments {
				if existing.Start == seg.Start {
					segments = segments[:j]
					break
				}
				if seg.Start > existing.Start && seg.End < existing.End && strings.HasPrefix(existing.Text, seg.Text) {
					segments = segments[:j]
					break
				}
			}
			segments = append(segments, seg)

			// fmt.Println(seg.String())
			// os.Stdout.Sync()
		}

		fmt.Println()
		for _, seg := range segments {
			fmt.Println(seg.String())
		}
		fmt.Println()
		fmt.Printf("### Transcription %d END\n", nIter)

		nIter++
		os.Stdout.Sync()
	}

	mic.Pause()
	ctx.PrintTimings()
}
