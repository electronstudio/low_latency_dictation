package main

// State is the high-level mode of the dictation main loop. It governs
// whether the microphone is captured and whether VAD/whisper run.
type State int

const (
	// StatePaused ignores audio: the microphone is paused and the loop waits
	// for an unpause (hotkey or 'p') or a quit. Entered at startup (unless
	// --skip-pause-mode), after finalizing (by default), and when the user
	// presses 'p' mid-session. The current session's accumulated audio and
	// segments are preserved so unpausing via 'p' can continue the same
	// dictation; unpausing via the hotkey starts a fresh session.
	StatePaused State = iota
	// StateListening is the normal idle-running mode: the mic is live and VAD
	// is below threshold.
	StateListening
	// StateDictating is the normal active mode: VAD is above threshold and
	// real-time transcription is running.
	StateDictating
	// StateFinalizing processes the final transcription of the current
	// session, emits/pastes it, then transitions to StatePaused (or
	// StateListening with --skip-pause-mode).
	StateFinalizing
)

func (s State) String() string {
	switch s {
	case StatePaused:
		return "PAUSED"
	case StateListening:
		return "LISTENING"
	case StateDictating:
		return "DICTATING"
	case StateFinalizing:
		return "FINALIZING"
	default:
		return "UNKNOWN"
	}
}

// Segment is one piece of transcribed text with timing information.
type Segment struct {
	Text  string
	Start int64
	End   int64
}
