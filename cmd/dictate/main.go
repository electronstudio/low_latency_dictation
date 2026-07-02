package main

import (
	_ "embed"
	"encoding/base64"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/alexflint/go-arg"
	"github.com/electronstudio/low_latency_dictation/assets"
	"github.com/electronstudio/low_latency_dictation/audio"
	"github.com/electronstudio/low_latency_dictation/hotkey"
	"github.com/electronstudio/low_latency_dictation/toast"
	"github.com/electronstudio/low_latency_dictation/transcribe"
	"github.com/electronstudio/low_latency_dictation/tray"
	"github.com/electronstudio/low_latency_dictation/typing"
	"github.com/electronstudio/low_latency_dictation/vad"
	"golang.design/x/clipboard"
	"golang.design/x/hotkey/mainthread"
)

//go:embed VERSION
var versionString string

type CLI struct {
	Model         string  `arg:"-m,--model" default:"ggml-tiny.en-q8_0.bin" help:"Model for real-time transcription, e.g. ggml-medium-q5_0.bin"`
	Preset        string  `arg:"-q,--quality-preset" default:"" help:"low: no GPU, medium: poor GPU, high: good GPU"`
	FinalModel    string  `arg:"-f,--final-model" default:"ggml-base.en.bin"      help:"Model for finalization, e.g ggml-large-v3-turbo-q5_0.bin, none to disable"`
	Threads       int     `arg:"-t,--threads" default:"0"                     help:"Threads (0=auto)"`
	UseCPU        bool    `arg:"--use-cpu"           default:"false"                 help:"Disable GPU accleration"`
	CaptureID     int     `arg:"-a,--audio-device"   default:"-1"                    help:"Audio device ID"`
	LogFile       string  `arg:"--log-file"          default:""                      help:"Path to log file for actions"`
	LogLevel      string  `arg:"--log-level"         default:"warn"                  help:"whisper.cpp console log level (debug/info/warn/error/none)"`
	HotkeyMods    string  `arg:"--hotkey-mods"       default:"ctrl"                  help:"Modifiers for the global hotkey (ctrl/alt/shift/cmd|win|super, joined by +)"`
	HotkeyKey     string  `arg:"--hotkey-key"        default:"space"                 help:"Key for the global hotkey (e.g. d, f1, space, escape)"`
	SkipPauseMode bool    `arg:"--skip-pause-mode"   default:"false"                 help:"Start listening immediately, and do not pause after pasting."`
	LengthMs      int     `arg:"-l,--length"         default:"30000"                 help:"(ADVANCED: Buffer length in ms)"`
	MaxTokens     int     `arg:"--max-tokens"        default:"32"                    help:"(ADVANCED: Max tokens per segment)"`
	AudioCtx      int     `arg:"--audio-ctx"         default:"0"                     help:"(ADVANCED: Audio context size)"`
	VadThold      float32 `arg:"--vad-thold"         default:"0.8"                   help:"(ADVANCED: VAD threshold)"`
	FreqThold     float32 `arg:"--freq-thold"        default:"100.0"                 help:"(ADVANCED: High-pass filter cutoff)"`
	Language      string  `arg:"--lang"              default:"en"                    help:"(ADVANCED: Language code)"`
	FlashAttn     bool    `arg:"--flash-attn"         default:"true"                  help:"(ADVANCED: Use flash attention)"`
	NoTray        bool    `arg:"--no-tray"            default:"false"                 help:"Disable the system tray icon"`
}

// TODO: remove LEngthMS?
// Add final threads and final use cpu
// display final segment even it overlapped and couldnt be added, because missing end word is annoying
// do whole buffer rather than circular buffer, or at least larger circle
// option to adjust interval
// option to disable VAD
// escape to delete, escape again to quit. C to copy.  enter to exit and copy.
// colors?
// word timings?
// better models?
// resident daemon?
// split front and backend.  network?
// suggested improvements, concurrency bug,
// toast on start even when no text dictated?

func (CLI) Version() string {
	return "low_latency_dictation " + strings.TrimSpace(versionString)
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

// hotkeyEvt carries a global-hotkey keydown/keyup from the forwarder to the
// main loop. t is stamped by the forwarder at receive time so the main loop
// can measure hold duration accurately regardless of its own polling latency.
type hotkeyEvt struct {
	down     bool
	t        time.Time
	fromTray bool
}

// holdThreshold is the minimum key-down duration for the hold-to-finalize
// gesture: holding the global hotkey longer than this while in the PAUSED
// state finalizes on release (push-to-talk); a shorter tap just unpauses and
// keeps listening.
const holdThreshold = 200 * time.Millisecond

// trayFocusDelay is the minimum time to wait after a tray-initiated finalize
// before simulating the paste keystroke. A tray double-click briefly grabs
// keyboard focus in the top bar; this delay lets focus return to the target
// window so the Ctrl+V reaches the right application. Time already spent in
// finalization is subtracted so the total wait is never more than this.
const trayFocusDelay = 200 * time.Millisecond

// App owns the lifetime of the dictation program: CLI configuration,
// audio capture, whisper contexts, UI, tray, and the state machine.
type App struct {
	cli         CLI
	params      whisperParams
	wparams     transcribe.FullParams
	mic         *audio.AudioAsync
	ctx         *transcribe.Context
	ctx2        *transcribe.Context
	ui          UI
	tr          *tray.Tray
	hotkey      *hotkey.Hotkey
	hotkeyLabel string

	running   atomic.Bool
	stop      chan struct{}
	closeOnce sync.Once
	hotkeyCh  chan hotkeyEvt
	trayCh    <-chan tray.Event

	skipPause bool
	clipErr   error
	tStart    time.Time
	tLast     time.Time

	state      State
	segments   []Segment
	holdActive bool
	holdStart  time.Time
}

// NewApp constructs an App from parsed CLI flags. It does not allocate any
// runtime resources; call Setup() before runMainLoop().
func NewApp(cli CLI) *App {
	return &App{
		cli:       cli,
		state:     initialState(cli.SkipPauseMode),
		hotkeyCh:  make(chan hotkeyEvt, 4),
		stop:      make(chan struct{}),
		skipPause: cli.SkipPauseMode,
	}
}

// Close tears down all resources owned by the App. It is safe to call
// multiple times.
func (a *App) Close() {
	a.closeOnce.Do(func() {
		a.running.Store(false)
		if a.tr != nil {
			a.tr.Quit()
		}
		if a.ui != nil {
			a.ui.Close()
		}
		if a.hotkey != nil {
			_ = a.hotkey.Unregister()
		}
		if a.mic != nil {
			a.mic.Close()
		}
		if a.ctx != nil {
			a.ctx.Free()
		}
		if a.ctx2 != nil {
			a.ctx2.Free()
		}
		close(a.stop)
	})
}

// die prints a message, tears down the App, and exits with the given code.
func (a *App) die(code int, format string, args ...interface{}) {
	a.Close()
	fmt.Fprintf(os.Stderr, format, args...)
	os.Exit(code)
}

// die is a standalone helper for fatal errors before an App is created.
func die(ui UI, code int, format string, args ...interface{}) {
	if ui != nil {
		ui.Close()
	}
	fmt.Fprintf(os.Stderr, format, args...)
	os.Exit(code)
}

// initialState returns the startup state based on --skip-pause-mode.
func initialState(skipPause bool) State {
	if skipPause {
		return StateListening
	}
	return StatePaused
}

// setTrayState updates the system tray icon to reflect the given app state,
// if a tray is running.
func (a *App) setTrayState(state string) {
	if a.tr != nil {
		a.tr.SetState(state)
	}
}

// setState changes the local state and mirrors the new state to the tray.
func (a *App) setState(s State) {
	a.state = s
	a.setTrayState(s.String())
}

// logStartup logs the effective configuration.
func (a *App) logStartup() {
	logActionf("=== startup ===")
	logActionf("model=%s final_model=%s threads=%d use_cpu=%v audio_device=%d length_ms=%d max_tokens=%d audio_ctx=%d vad_thold=%.2f freq_thold=%.1f lang=%s flash_attn=%v",
		a.cli.Model, a.cli.FinalModel, a.cli.Threads, a.cli.UseCPU, a.cli.CaptureID, a.cli.LengthMs,
		a.cli.MaxTokens, a.cli.AudioCtx, a.cli.VadThold, a.cli.FreqThold,
		a.cli.Language, a.cli.FlashAttn)
}

// Setup prepares all runtime resources. It returns the first fatal error;
// best-effort subsystems print warnings but do not fail.
func (a *App) Setup() error {
	if a.cli.LogFile != "" {
		f := openLogFile(a.cli)
		defer f.Close()
	}
	setupWhisperLogging(a.cli)

	a.clipErr = initClipboard()
	if err := toast.Init(logger); err != nil {
		fmt.Fprintf(os.Stderr, "warning: %v\n", err)
	}

	a.initTray()

	transcribe.BackendLoadAll()
	a.logStartup()

	if err := a.loadModels(); err != nil {
		return err
	}

	a.params, a.wparams = buildParams(a.cli)

	if err := a.initAudio(); err != nil {
		return err
	}
	if a.state == StatePaused {
		if err := a.mic.Pause(); err != nil {
			logActionf("audio pause at startup failed: %v", err)
		}
	}

	if err := a.registerHotkey(); err != nil {
		return err
	}
	a.startSDLPoller()

	if err := a.initUI(); err != nil {
		return err
	}
	return nil
}

// initAudio opens and resumes the microphone capture.
func (a *App) initAudio() error {
	a.mic = audio.NewAudioAsync(a.params.lengthMs)
	if err := a.mic.Init(a.params.captureID, transcribe.WhisperSampleRate); err != nil {
		logActionf("audio init failed: %v", err)
		return fmt.Errorf("main: audio.Init() failed: %w", err)
	}
	logActionf("audio init ok device=%d", a.params.captureID)

	if err := a.mic.Resume(); err != nil {
		logActionf("audio resume failed: %v", err)
		return fmt.Errorf("main: audio.Resume() failed: %w", err)
	}
	logActionf("audio resume ok")
	return nil
}

// loadModels resolves (downloading if necessary) and loads the primary
// whisper context and, unless --final-model is "none", the finalization
// context.
func (a *App) loadModels() error {
	cp := transcribe.ContextParams{UseGPU: !a.cli.UseCPU, FlashAttn: a.cli.FlashAttn}

	ctxModelPath, err := resolveModelFile(a.cli.Model)
	if err != nil {
		logActionf("model load failed: %v", err)
		return err
	}
	a.ctx, err = transcribe.InitFromFile(ctxModelPath, cp)
	if err != nil {
		logActionf("context init failed: %v", err)
		return err
	}
	logActionf("context init ok model=%s", ctxModelPath)

	if a.cli.FinalModel != "none" {
		ctx2ModelPath, err := resolveModelFile(a.cli.FinalModel)
		if err != nil {
			logActionf("final model load failed: %v", err)
			return err
		}
		a.ctx2, err = transcribe.InitFromFile(ctx2ModelPath, cp)
		if err != nil {
			logActionf("final context init failed: %v", err)
			return err
		}
		logActionf("final context init ok model=%s", ctx2ModelPath)
	}
	return nil
}

// registerHotkey parses and registers the global hotkey and starts the
// forwarder goroutine that bridges hotkey events into the main loop.
func (a *App) registerHotkey() error {
	if a.cli.HotkeyKey == "" {
		fmt.Fprintf(os.Stderr, "warning: global hotkey disabled.\n")
		return nil
	}
	hkMods, hkKey, perr := hotkey.ParseCombo(a.cli.HotkeyMods, a.cli.HotkeyKey)
	if perr != nil {
		return fmt.Errorf("invalid --hotkey-mods or --hotkey-key (%q+%q): %w", a.cli.HotkeyMods, a.cli.HotkeyKey, perr)
	}
	hk, rerr := hotkey.Register(hkMods, hkKey)
	if rerr != nil {
		return fmt.Errorf("could not register global hotkey %s+%s: %w", a.cli.HotkeyMods, a.cli.HotkeyKey, rerr)
	}
	a.hotkey = hk
	a.hotkeyLabel = hk.String()
	logActionf("hotkey registered: %s", a.hotkeyLabel)
	go a.runHotkeyForwarder()
	return nil
}

// runHotkeyForwarder forwards keydown/keyup events from the hotkey package
// into the App's hotkeyCh. It exits when the App is closed.
func (a *App) runHotkeyForwarder() {
	kd := a.hotkey.Keydown()
	ku := a.hotkey.Keyup()
	for {
		select {
		case <-a.stop:
			return
		case _, ok := <-kd:
			if !ok {
				return
			}
			logActionf("hotkey keydown")
			select {
			case a.hotkeyCh <- hotkeyEvt{down: true, t: time.Now()}:
			default:
			}
		case _, ok := <-ku:
			if !ok {
				return
			}
			logActionf("hotkey keyup")
			select {
			case a.hotkeyCh <- hotkeyEvt{down: false, t: time.Now()}:
			default:
			}
		}
	}
}

// startSDLPoller polls SDL events in a goroutine so the main loop can respond
// to SDL quit events.
func (a *App) startSDLPoller() {
	a.running.Store(true)
	go func() {
		for a.running.Load() {
			if !audio.PollEvents() {
				a.running.Store(false)
				return
			}
			time.Sleep(10 * time.Millisecond)
		}
	}()
}

// initUI opens the terminal UI and shows the initial state.
func (a *App) initUI() error {
	a.ui = NewTerminalUI(a.hotkeyLabel)
	if err := a.ui.Init(); err != nil {
		return fmt.Errorf("failed to initialize UI: %w", err)
	}
	a.ui.ShowText("", a.state)
	a.setTrayState(a.state.String())
	return nil
}

// initTray starts the system tray icon when enabled. It is best-effort: if
// the platform has no tray host the icon simply does not appear but the app
// keeps running.
func (a *App) initTray() {
	if a.cli.NoTray {
		return
	}
	if t, err := tray.Start(mainthread.Call, assets.IconForState, a.state.String()); err == nil {
		a.tr = t
		a.trayCh = t.Events
	} else {
		logActionf("tray disabled: %v", err)
	}
}

// runMainLoop is the real-time transcription state machine.
func (a *App) runMainLoop() error {
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	a.tStart = time.Now()
	a.tLast = a.tStart

	for a.running.Load() {
		// Paused: mic is stopped, session preserved. Wait for input or a
		// periodic wake so a quit (running flag flip) is still noticed.
		if a.state == StatePaused {
			select {
			case <-sigCh:
				return nil
			case evt := <-a.hotkeyCh:
				if err := a.handlePausedEvent(evt); err != nil {
					return err
				}
			case <-a.ui.PauseEvents():
				a.handlePause()
			case <-a.ui.DeleteEvents():
				a.handleDelete()
			case evt := <-a.trayCh:
				a.handleTray(evt)
			case <-a.ui.QuitRequested():
				return nil
			case <-time.After(100 * time.Millisecond):
			}
			continue
		}

		// Listening/Dictating: drain input events, otherwise transcribe.
		select {
		case <-sigCh:
			return nil
		case evt := <-a.hotkeyCh:
			if err := a.handleActiveEvent(evt); err != nil {
				return err
			}
		case <-a.ui.PauseEvents():
			a.handlePause()
		case <-a.ui.DeleteEvents():
			a.handleDelete()
		case evt := <-a.trayCh:
			a.handleTray(evt)
		case <-a.ui.QuitRequested():
			return nil
		default:
			if err := a.transcribeTick(); err != nil {
				return err
			}
		}
	}
	return nil
}

// handlePausedEvent handles a global-hotkey event while the app is paused.
func (a *App) handlePausedEvent(evt hotkeyEvt) error {
	if evt.down {
		// Unpause via hotkey: begin a fresh session (discard the preserved
		// one, since the hotkey is the begin/submit gesture) and arm the
		// hold-to-finalize gesture.
		a.mic.Clear()
		a.segments = nil
		now := time.Now()
		a.tLast = now
		a.tStart = now
		if err := a.mic.Resume(); err != nil {
			logActionf("audio resume on unpause failed: %v", err)
		}
		a.holdActive = true
		a.holdStart = evt.t
		a.setState(StateListening)
		a.redrawText("", StateListening)
	} else {
		// keyup while still Paused: just disarm; nothing to finalize.
		a.holdActive = false
	}
	return nil
}

// handleActiveEvent handles a global-hotkey event while listening/dictating.
func (a *App) handleActiveEvent(evt hotkeyEvt) error {
	if evt.down {
		// Keydown while active (not an armed hold) finalizes the current
		// session, matching the original toggle behavior.
		if !a.holdActive {
			if err := a.finalize(evt.fromTray); err != nil {
				return err
			}
		}
		// A second keydown during an armed hold is ignored; only the release
		// decides whether to submit.
	} else {
		// Release of an armed hold: finalize iff it was a long press.
		if a.holdActive {
			held := evt.t.Sub(a.holdStart)
			a.holdActive = false
			if held >= holdThreshold {
				logActionf("hold finalized after %dms", held.Milliseconds())
				if err := a.finalize(evt.fromTray); err != nil {
					return err
				}
			} else {
				logActionf("hold released after %dms (tap, keep listening)", held.Milliseconds())
			}
		}
		// A stray keyup with no armed hold is ignored.
	}
	return nil
}

// handlePause toggles the paused state without finalizing. It is shared
// between the UI pause-event channel and the tray pause menu item.
func (a *App) handlePause() {
	switch a.state {
	case StatePaused:
		// Unpause: resume the preserved session in place.
		if err := a.mic.Resume(); err != nil {
			logActionf("audio resume on unpause failed: %v", err)
		}
		a.setState(StateListening)
		a.redrawText("", StateListening)
	case StateListening, StateDictating:
		// Pause: preserve the in-flight session.
		a.holdActive = false
		if err := a.mic.Pause(); err != nil {
			logActionf("audio pause failed: %v", err)
		}
		a.setState(StatePaused)
		a.ui.ShowStatus(a.state.String())
	}
}

// handleDelete discards the current session without finalizing. It is shared
// between the UI delete-event channel and the tray delete menu item.
func (a *App) handleDelete() {
	a.mic.Clear()
	a.segments = nil
	a.holdActive = false
	now := time.Now()
	a.tLast = now
	a.tStart = now
	switch a.state {
	case StatePaused:
		a.redrawText("", StatePaused)
		logActionf("delete (paused)")
	default:
		a.setState(StateListening)
		a.redrawText("", StateListening)
		logActionf("delete (active)")
	}
}

// handleTray dispatches a tray event to the appropriate handler.
func (a *App) handleTray(evt tray.Event) {
	switch evt {
	case tray.EventToggle:
		now := time.Now()
		select {
		case a.hotkeyCh <- hotkeyEvt{down: true, t: now, fromTray: true}:
		default:
		}
		select {
		case a.hotkeyCh <- hotkeyEvt{down: false, t: now, fromTray: true}:
		default:
		}
	case tray.EventDelete:
		a.handleDelete()
	case tray.EventPause:
		a.handlePause()
	case tray.EventAbout:
		toast.Show("low_latency_dictation", strings.TrimSpace(versionString), false)
	case tray.EventQuit:
		a.running.Store(false)
	}
}

// redrawText clears the screen and repaints the given text plus the status
// line, then updates the tray and toast.
func (a *App) redrawText(text string, s State) {
	a.ui.ShowText(text, s)
	if a.tr != nil {
		a.tr.SetDictationText(text)
	}
	persist := s != StatePaused
	if strings.TrimSpace(text) != "" {
		toast.Show(s.String(), text, persist)
	}
}

// segmentsText concatenates the accumulated session segments into a single
// string for display.
func (a *App) segmentsText() string {
	var sb strings.Builder
	for _, segment := range a.segments {
		sb.WriteString(segment.Text)
	}
	return sb.String()
}

// finalize runs the final transcription, emits/pastes it, clears the session,
// and transitions to the post-finalize state.
func (a *App) finalize(fromTray bool) error {
	start := time.Now()
	a.setState(StateFinalizing)
	a.redrawText(a.segmentsText(), StateFinalizing)

	finalText, err := a.produceFinalText()
	if err != nil {
		return err
	}
	a.emitFinal(finalText, fromTray, start)
	a.drainInput()

	a.mic.Clear()
	a.segments = nil
	now := time.Now()
	a.tLast = now
	a.tStart = now

	if a.skipPause {
		if err := a.mic.Resume(); err != nil {
			logActionf("audio resume after finalize failed: %v", err)
		}
		a.setState(StateListening)
		a.redrawText("", StateListening)
	} else {
		// produceFinalText already paused the mic; stay paused. Draw the
		// finalized text in green and leave it on screen while paused; any
		// unpause (hotkey or 'p') clears it for the next session.
		a.setState(StatePaused)
		a.redrawText(finalText, StatePaused)
	}
	return nil
}

// produceFinalText pauses the microphone and produces the final transcription
// text for the just-ended session.
func (a *App) produceFinalText() (string, error) {
	_ = a.mic.Pause()
	var finalText string
	if a.ctx2 != nil {
		fullAudio := a.mic.GetFullAudio()
		if len(fullAudio) > 0 {
			if err := a.ctx2.Full(a.wparams, fullAudio); err != nil {
				return "", fmt.Errorf("main: failed to process audio: %w", err)
			}
		}

		nSegments := a.ctx2.NSegments()
		var sb strings.Builder
		for i := 0; i < nSegments; i++ {
			sb.WriteString(a.ctx2.SegmentText(i))
		}
		finalText = strings.TrimSpace(sb.String())
		logActionf("final transcription n_segments=%d text=%q", nSegments, finalText)
	} else {
		var sb strings.Builder
		for _, segment := range a.segments {
			sb.WriteString(segment.Text)
		}
		finalText = strings.TrimSpace(sb.String())
		logActionf("final text from segments text=%q", finalText)
	}
	return finalText, nil
}

// emitFinal offers the transcription through every available channel and
// simulates a paste keystroke.
func (a *App) emitFinal(finalText string, fromTray bool, start time.Time) {
	if f, err := os.OpenFile("/dev/clipboard", os.O_WRONLY|os.O_TRUNC, 0); err == nil {
		_, _ = f.WriteString(finalText)
		_ = f.Close()
	}
	data := base64.StdEncoding.EncodeToString([]byte(finalText))
	if a.clipErr == nil {
		clipboard.Write(clipboard.FmtText, []byte(finalText))
	} else if err := tryWlCopy(finalText); err != nil {
		logActionf("wl-copy fallback failed: %v", err)
	} else {
		logActionf("wl-copy ok")
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

	if fromTray {
		if remaining := trayFocusDelay - time.Since(start); remaining > 0 {
			time.Sleep(remaining)
		}
	}

	if err := typing.Paste(); err != nil {
		fmt.Fprintf(os.Stderr, "warning: could not auto-paste: %v\n", err)
		fmt.Fprintf(os.Stderr, "warning: text is on the clipboard; paste manually with Ctrl/Cmd+V\n")
		logActionf("paste failed: %v", err)
	} else {
		logActionf("paste ok")
	}
}

// drainInput discards any pending hotkey/pause/delete events that arrived
// while the loop was blocked (e.g. during a slow finalization) so the next
// session starts with a clean input state.
func (a *App) drainInput() {
	for {
		select {
		case <-a.hotkeyCh:
		case <-a.ui.PauseEvents():
		case <-a.ui.DeleteEvents():
		default:
			return
		}
	}
}

// transcribeTick samples the microphone, runs VAD, and transcribes active
// speech.
func (a *App) transcribeTick() error {
	tNow := time.Now()
	tDiff := tNow.Sub(a.tLast).Milliseconds()

	if tDiff < 100 {
		time.Sleep(16 * time.Millisecond)
		return nil
	}
	ls := a.mic.Get(500)

	if !vad.SimpleVAD(ls, transcribe.WhisperSampleRate, 250, a.params.vadThold, a.params.freqThold, false) {
		a.setState(StateListening)
		a.ui.ShowStatus(a.state.String())
		time.Sleep(16 * time.Millisecond)
		return nil
	}
	a.setState(StateDictating)
	a.ui.ShowStatus(a.state.String())
	logActionf("vad activated")
	pcmf32New := a.mic.Get(a.params.lengthMs)
	a.tLast = tNow

	if len(pcmf32New) == 0 {
		time.Sleep(16 * time.Millisecond)
		return nil
	}

	if err := a.ctx.Full(a.wparams, pcmf32New); err != nil {
		return fmt.Errorf("main: failed to process audio: %w", err)
	}

	transcriptionEnd := a.tLast.Sub(a.tStart).Milliseconds()
	transcriptionStart := max(0, int(transcriptionEnd)-len(pcmf32New)*1000/transcribe.WhisperSampleRate)

	nSegments := a.ctx.NSegments()
	for i := 0; i < nSegments; i++ {
		text := a.ctx.SegmentText(i)
		t0 := a.ctx.SegmentT0(i) + int64(transcriptionStart)/10
		t1 := a.ctx.SegmentT1(i) + int64(transcriptionStart)/10

		seg := Segment{
			Text:  text,
			Start: t0,
			End:   t1,
		}
		logActionf("segment t0=%d t1=%d text=%q", t0, t1, text)

		for j, existing := range a.segments {
			if existing.Start >= seg.Start-100 && existing.Start <= seg.Start+100 {
				logActionf("truncating segments to %d because of %q", j, seg.Text)
				a.segments = a.segments[:j]
				break
			}
		}
		if len(a.segments) == 0 || a.segments[len(a.segments)-1].End <= seg.Start {
			logActionf("appending segment %q", seg.Text)
			a.segments = append(a.segments, seg)
		}
	}

	// redraw all segments on screen
	a.redrawText(a.segmentsText(), a.state)
	return nil
}

// main runs the program body. On macOS, mainthread.Init hands the real main
// thread to the NSApplication event loop (required by the global-hotkey
// backend, which delivers events via a CGEventTap on the main run loop) and
// runs run() in a goroutine. On Linux/Windows the hotkey backends manage
// their own threads, so Init's main-thread dispatch loop sits idle; using the
// same entry point on every platform keeps the code paths identical.
func main() { mainthread.Init(run) }

// run is the program body. It is invoked from a goroutine by main() on every
// platform. It parses the CLI, builds the App, runs setup, and then enters
// the main loop. All fatal exits happen from here.
func run() {
	var cli CLI
	arg.MustParse(&cli)
	applyPreset(&cli)

	if err := typing.Init(); err != nil {
		fmt.Fprintf(os.Stderr, "warning: %v\n", err)
	}
	defer typing.Close()

	app := NewApp(cli)
	if err := app.Setup(); err != nil {
		app.Close()
		die(nil, 1, "error: %v\n", err)
		return
	}
	defer app.Close()

	if err := app.runMainLoop(); err != nil {
		app.die(1, "error: %v\n", err)
		return
	}
}

// applyPreset mutates the CLI based on the selected quality preset.
func applyPreset(cli *CLI) {
	switch cli.Preset {
	case "low":
		cli.Model = "ggml-tiny.en-q5_1.bin"
		cli.FinalModel = "ggml-small.en-q5_1.bin"
		cli.UseCPU = true
		cli.LengthMs = 30000
	case "medium":
		cli.Model = "ggml-medium-q5_0.bin"
		cli.FinalModel = "ggml-large-v3-turbo-q5_0.bin"
		cli.UseCPU = false
		cli.LengthMs = 60000
	case "high":
		cli.Model = "ggml-large-v3-turbo-q5_0.bin"
		cli.FinalModel = "ggml-large-v3-turbo-q5_0.bin"
		cli.UseCPU = false
		cli.LengthMs = 1800000
	}
}

// buildParams translates the parsed CLI into a whisperParams (used for the
// audio/VAD path) and a transcribe.FullParams (used for every ctx.Full call).
func buildParams(cli CLI) (whisperParams, transcribe.FullParams) {
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
	return params, wparams
}

// initClipboard initializes the system clipboard backend. If it is unavailable
// (e.g. no Wayland data-control manager and no X11 server), the returned error
// is non-nil and the caller keeps running: the transcription is still printed
// and offered via /dev/clipboard and OSC 52.
func initClipboard() error {
	clipErr := clipboard.Init()
	if clipErr != nil {
		fmt.Fprintf(os.Stderr, "warning: clipboard unavailable: %v\n", clipErr)
		logActionf("clipboard init failed: %v", clipErr)
	}
	return clipErr
}

// tryWlCopy copies text to the Wayland clipboard by shelling out to wl-copy
// (from wl-clipboard). It is used as a fallback when the Go clipboard package
// cannot initialise — e.g. on Wayland compositors that lack the data-control
// protocol (GNOME) and inside the Flatpak sandbox where no X11 display is
// available. wl-copy uses the standard wl_data_device protocol, which works on
// all Wayland compositors. By default it forks into the background to serve
// the clipboard content and the parent exits once the selection is set, so
// Run returns after the clipboard is ready.
func tryWlCopy(text string) error {
	cmd := exec.Command("wl-copy")
	cmd.Stdin = strings.NewReader(text)
	return cmd.Run()
}
