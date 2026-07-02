// Package tray provides an optional system tray icon for the dictation app.
//
// It is a best-effort feature: on a session without a tray host (for example a
// headless SSH login or GNOME without the AppIndicator extension) the icon
// simply does not appear, but the application keeps running. Callers should
// treat a nil *Tray (returned when the user passes --no-tray) as "no tray";
// every method is safe to call only on a non-nil Tray.
package tray

import (
	"os"
	"strings"
	"sync"

	"fyne.io/systray"
)

// gnomeDesktop reports whether the current session is GNOME. The GNOME
// AppIndicator extension ignores the SNI ToolTip property and shows the
// menu on single left-click (requiring a double-click to Activate), so the
// toggle menu item's label gets a "(double click)" hint there.
var gnomeDesktop = detectGNOME()

func detectGNOME() bool {
	for _, part := range strings.Split(os.Getenv("XDG_CURRENT_DESKTOP"), ":") {
		if strings.EqualFold(strings.TrimSpace(part), "GNOME") {
			return true
		}
	}
	return false
}

// Event is a user action originating from the tray icon or its menu.
type Event int

const (
	// EventToggle requests the same action as the global hotkey: when paused it
	// starts a fresh dictation session; when listening/dictating it finalizes
	// and pastes the current session.
	EventToggle Event = iota
	// EventPause requests pausing without finalizing (when listening) or
	// resuming the preserved session (when paused), mirroring the 'p' key.
	EventPause
	// EventDelete discards the current session without finalizing.
	EventDelete
	// EventAbout asks the caller to show the application name and version.
	EventAbout
	// EventQuit requests application shutdown.
	EventQuit
)

const appName = "low_latency_dictation"

// Tray is a running system tray instance. Events carries user actions;
// SetState updates the icon, tooltip and menu labels to reflect the app state.
// Quit tears the tray down and must be called before the process exits.
type Tray struct {
	Events chan Event

	dispatch  func(func()) // runs a func on the platform main thread when required
	iconFor   func(state string) []byte
	endFn     func()        // shutdown func (RunWithExternalLoop's end); nil on windows
	ready     chan struct{} // closed once onReady has built the menu
	mu        sync.Mutex
	state     string
	dictation *systray.MenuItem
	dictText  string
	toggle    *systray.MenuItem
	pause     *systray.MenuItem
}

// Start creates and registers the tray icon and returns it. dispatch runs the
// platform startup on the main thread when the platform requires it (macOS);
// iconFor maps an app state name (as returned by appState.String, e.g.
// "PAUSED") to the embedded icon bytes for the current platform. initialState
// is the state shown until the first SetState call. It returns an error only if
// the tray cannot be initialized on the current platform; the caller should
// then run without a tray.
func Start(dispatch func(func()), iconFor func(string) []byte, initialState string) (*Tray, error) {
	t := &Tray{
		Events:   make(chan Event, 4),
		dispatch: dispatch,
		iconFor:  iconFor,
		ready:    make(chan struct{}),
		state:    initialState,
	}
	// Install the primary tap handler before the platform tray initializes.
	// On Linux the org.kde.StatusNotifierItem "ItemIsMenu" property is
	// computed from tappedLeft at nativeStart time (before onReady runs in its
	// own goroutine), so setting it here ensures ItemIsMenu is false and the
	// host calls Activate on left-click instead of unconditionally showing the
	// menu. On macOS/Windows this is harmless (the handler is also installed
	// before the status item / message window is created).
	systray.SetOnTapped(t.emitToggle)
	if err := t.start(); err != nil {
		return nil, err
	}
	return t, nil
}

// SetState updates the tray icon, tooltip and the toggle item's label to
// reflect the given state name. It blocks until the tray's onReady has finished
// building the menu, then drops redundant updates (a no-op when state is
// unchanged) so it is cheap to call on every status refresh.
func (t *Tray) SetState(state string) {
	<-t.ready
	t.mu.Lock()
	if t.state == state {
		t.mu.Unlock()
		return
	}
	t.state = state
	t.mu.Unlock()

	systray.SetIcon(t.iconFor(state))
	systray.SetTooltip(appName + " — " + state)
	if t.toggle != nil {
		t.toggle.SetTitle(toggleTitle(state))
	}
	if t.pause != nil {
		t.pause.SetTitle(pauseTitle(state))
	}
}

// SetDictationText updates the disabled menu item at the top of the menu that
// shows a live preview of the current dictation text. An empty text hides the
// item; non-empty text is word-wrapped at 80 characters and shown.
// Redundant updates (same text as last call) are dropped to avoid unnecessary
// DBus traffic.
func (t *Tray) SetDictationText(text string) {
	<-t.ready
	text = wrapWords(strings.TrimSpace(strings.ReplaceAll(text, "\n", " ")), 80)
	t.mu.Lock()
	if t.dictText == text {
		t.mu.Unlock()
		return
	}
	t.dictText = text
	t.mu.Unlock()

	if text == "" {
		t.dictation.Hide()
	} else {
		t.dictation.SetTitle(text)
		t.dictation.Show()
	}
}

// wrapWords word-waps text at the given line width, breaking on word
// boundaries. A single space between words is preserved within a line; a
// newline is inserted where a break occurs.
func wrapWords(text string, width int) string {
	if width <= 0 {
		return text
	}
	var sb strings.Builder
	lineLen := 0
	for _, word := range strings.Fields(text) {
		if lineLen > 0 && lineLen+1+len(word) > width {
			sb.WriteByte('\n')
			lineLen = 0
		}
		if lineLen > 0 {
			sb.WriteByte(' ')
			lineLen++
		}
		sb.WriteString(word)
		lineLen += len(word)
	}
	return sb.String()
}

// toggleTitle returns the menu label for the toggle item in the given state:
// "Start dictation" when a click would begin a session, "Submit dictation"
// when a click would finalize and paste.
func toggleTitle(state string) string {
	var label string
	switch state {
	case "PAUSED", "FINALIZING":
		label = "Start dictation"
	default: // LISTENING, DICTATING
		label = "Submit dictation"
	}
	if gnomeDesktop {
		label += " (or double click icon)"
	} else {
		label += " (or left click icon)"
	}
	return label
}

// pauseTitle returns the menu label for the pause item in the given state:
// "Pause" when listening/dictating, "Resume" when paused.
func pauseTitle(state string) string {
	switch state {
	case "PAUSED":
		return "Resume"
	default: // LISTENING, DICTATING, FINALIZING
		return "Pause"
	}
}

// onReady builds the menu and installs the click handlers. It is invoked by
// the systray library on a separate goroutine once the platform tray is ready.
func (t *Tray) onReady() {
	defer close(t.ready)

	systray.SetIcon(t.iconFor(t.state))
	systray.SetTooltip(appName + " — " + t.state)

	t.dictation = systray.AddMenuItem("", "Current dictation text")
	t.dictation.Disable()
	t.dictation.Hide()

	t.toggle = systray.AddMenuItem(toggleTitle(t.state), "Start/stop dictation (same as the global hotkey)")
	t.pause = systray.AddMenuItem(pauseTitle(t.state), "Pause/resume dictation (same as the 'p' key)")
	systray.AddSeparator()
	mDelete := systray.AddMenuItem("Delete session", "Discard the current dictation without finalizing")
	mAbout := systray.AddMenuItem("About", "Show the application name and version")
	systray.AddSeparator()
	mQuit := systray.AddMenuItem("Quit", "Quit the application")

	// The primary tap handler is installed in Start before nativeStart so that
	// ItemIsMenu is false; a right-click shows this menu on every platform
	// (the secondary handler is unset, so each falls back to show_menu).
	go t.forward(t.toggle, EventToggle)
	go t.forward(t.pause, EventPause)
	go t.forward(mDelete, EventDelete)
	go t.forward(mAbout, EventAbout)
	go t.forward(mQuit, EventQuit)
}

// forward drains a menu item's click channel and forwards each click as the
// given Event. It exits when the channel is closed (on tray teardown).
func (t *Tray) forward(m *systray.MenuItem, e Event) {
	for range m.ClickedCh {
		t.emit(e)
	}
}

func (t *Tray) emit(e Event) {
	select {
	case t.Events <- e:
	default:
	}
}

func (t *Tray) emitToggle() { t.emit(EventToggle) }

func (t *Tray) onExit() {}
