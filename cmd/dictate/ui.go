package main

// UI is the abstract interface used by the dictation state machine to show
// status and text and to receive user input events. A terminal UI and,
// eventually, a GUI can both implement it.
type UI interface {
	// Init prepares the UI (e.g. opens the tcell screen) and returns an error
	// if it cannot be started. It must be called before any other method.
	Init() error

	// Close tears down the UI (e.g. restores the terminal). It is safe to
	// call multiple times.
	Close()

	// ShowStatus updates the status line with the given status text.
	ShowStatus(status string)

	// ShowText displays the given dictation text, styling it according to
	// the current application state. The status line is also updated to match
	// the state.
	ShowText(text string, state State)

	// PauseEvents returns a channel that receives an event when the user
	// requests pause/resume (e.g. pressing 'p').
	PauseEvents() <-chan struct{}

	// DeleteEvents returns a channel that receives an event when the user
	// requests deletion of the current session (e.g. pressing 'd').
	DeleteEvents() <-chan struct{}

	// QuitRequested returns a channel that receives an event when the user
	// requests application shutdown (e.g. pressing 'q').
	QuitRequested() <-chan struct{}
}
