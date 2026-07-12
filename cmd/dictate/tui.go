package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/gdamore/tcell/v2"
	"github.com/mattn/go-runewidth"
)

// TerminalUI implements the UI interface using tcell.
type TerminalUI struct {
	screen          tcell.Screen
	width           int
	height          int
	statusStyle     tcell.Style
	finalizingStyle tcell.Style
	finalizedStyle  tcell.Style
	hotkeyLabel     string

	pauseCh   chan struct{}
	deleteCh  chan struct{}
	quitCh    chan struct{}
	stop      chan struct{}
	closeOnce sync.Once

	app *App
}

// NewTerminalUI creates a new terminal UI instance.
func NewTerminalUI(hotkeyLabel string, app *App) *TerminalUI {
	return &TerminalUI{
		hotkeyLabel:     hotkeyLabel,
		pauseCh:         make(chan struct{}, 1),
		deleteCh:        make(chan struct{}, 1),
		quitCh:          make(chan struct{}, 1),
		stop:            make(chan struct{}),
		statusStyle:     tcell.StyleDefault.Reverse(true).Bold(true),
		finalizingStyle: tcell.StyleDefault.Foreground(tcell.ColorRed),
		finalizedStyle:  tcell.StyleDefault.Foreground(tcell.ColorGreen),
		app:             app,
	}
}

// Init opens the tcell screen and starts the input poller.
func (t *TerminalUI) Init() error {
	tcell.SetEncodingFallback(tcell.EncodingFallbackASCII)

	var err error
	t.screen, err = tcell.NewScreen()
	if err != nil {
		os.Setenv("TERM", "xterm-256color")
		t.screen, err = tcell.NewScreen()
	}
	if err != nil {
		return fmt.Errorf("failed to create tcell screen: %w", err)
	}
	if err := t.screen.Init(); err != nil {
		return fmt.Errorf("failed to init tcell screen: %w", err)
	}
	t.screen.SetStyle(tcell.StyleDefault)
	t.screen.Clear()
	t.width, t.height = t.screen.Size()

	go t.pollEvents()
	return nil
}

// Close restores the terminal and stops the input poller.
func (t *TerminalUI) Close() {
	t.closeOnce.Do(func() {
		if t.screen != nil {
			t.screen.Fini()
			t.screen = nil
		}
		close(t.stop)
	})
}

// ShowStatus updates the bottom status line and flushes the change to the
// screen.
func (t *TerminalUI) ShowStatus(state State) {
	if t.height < 4 {
		return
	}
	blankLine := strings.Repeat(" ", t.width)

	printToScreen(t.screen, 0, 0, t.statusStyle, blankLine)
	printToScreen(t.screen, 0, t.height-1, t.statusStyle, blankLine)

	var style tcell.Style
	var s string
	switch state {
	case StateDictating:
		style = t.statusStyle.Foreground(tcell.ColorBlue)
		s = "<<< LISTENING  >>>    press [" + t.hotkeyLabel + "] to finish and paste"
	case StatePaused:
		style = t.statusStyle.Foreground(tcell.ColorRed)
		s = "<<<   PAUSED   >>>    press [" + t.hotkeyLabel + "] to start"
	case StateListening:
		style = t.statusStyle.Foreground(tcell.ColorGreen)
		s = "<<< LISTENING  >>>    press [" + t.hotkeyLabel + "] to finish and paste"
	case StateFinalizing:
		style = t.statusStyle.Foreground(tcell.ColorYellow)
		s = "<<< FINALIZING >>>    text will be copied and pasted"
	}

	printToScreen(t.screen, 0, 0, style, s)

	s2 := "dictate V" + versionString + "   |   keys: [P]ause [Q]uit [D]elete   |   latency " + strconv.FormatInt(t.app.transcriptionTime.Milliseconds(), 10) + "ms  |  buffer " + strconv.Itoa(t.app.mic.AudioPos/1024)
	printToScreen(t.screen, 0, t.height-1, t.statusStyle, s2)

	t.screen.Show()
}

// ShowText clears the screen and displays the given text, styled according to
// the current application state. The status line is also refreshed.
func (t *TerminalUI) ShowText(text string, state State) {
	style := tcell.StyleDefault
	switch state {
	case StateFinalizing:
		style = t.finalizingStyle
	case StatePaused:
		if strings.TrimSpace(text) != "" {
			style = t.finalizedStyle
		}
	}

	t.screen.Clear()
	printWrapped(t.screen, 0, 1, t.width, t.height-2, style, strings.TrimSpace(text))
	t.ShowStatus(state)
	t.screen.Show()
}

// PauseEvents returns the channel that receives pause/resume requests.
func (t *TerminalUI) PauseEvents() <-chan struct{} { return t.pauseCh }

// DeleteEvents returns the channel that receives delete-session requests.
func (t *TerminalUI) DeleteEvents() <-chan struct{} { return t.deleteCh }

// QuitRequested returns the channel that receives quit requests.
func (t *TerminalUI) QuitRequested() <-chan struct{} { return t.quitCh }

// pollEvents reads keyboard events and routes them to the appropriate
// channels. It exits when Close is called.
func (t *TerminalUI) pollEvents() {
	for {
		select {
		case <-t.stop:
			return
		default:
		}

		ev := t.screen.PollEvent()
		if ev == nil {
			return
		}

		switch ev := ev.(type) {
		case *tcell.EventKey:
			if ev.Key() == tcell.KeyCtrlC || ev.Key() == tcell.KeyEscape {
				select {
				case t.quitCh <- struct{}{}:
				default:
				}
			} else if ev.Key() == tcell.KeyRune {
				switch ev.Rune() {
				case 'q':
					select {
					case t.quitCh <- struct{}{}:
					default:
					}
				case 'p', 'P':
					select {
					case t.pauseCh <- struct{}{}:
					default:
					}
				case 'd', 'D':
					select {
					case t.deleteCh <- struct{}{}:
					default:
					}
				}
			}
		}
	}
}

func printToScreen(screen tcell.Screen, x, y int, style tcell.Style, text string) {
	col := 0
	for _, r := range text {
		screen.SetContent(x+col, y, r, nil, style)
		col += runewidth.RuneWidth(r)
	}
}

func printWrapped(screen tcell.Screen, x, y, maxWidth, maxLines int, style tcell.Style, text string) {
	if maxWidth <= 0 || maxLines <= 0 {
		return
	}

	rs := []rune(text)
	n := len(rs)
	widths := make([]int, n+1)
	for i := 0; i < n; i++ {
		widths[i+1] = widths[i] + runewidth.RuneWidth(rs[i])
	}

	var lines []string
	start := 0
	for start < n {
		if widths[n]-widths[start] <= maxWidth {
			lines = append(lines, string(rs[start:]))
			break
		}

		end := start
		for end < n && widths[end+1]-widths[start] <= maxWidth {
			end++
		}

		cut := end
		for i := end; i > start; i-- {
			if rs[i] == ' ' {
				cut = i
				break
			}
		}
		if cut == end {
			// Hard break.
			lines = append(lines, string(rs[start:end+1]))
			start = end + 1
			if start < n && rs[start] == ' ' {
				start++
			}
		} else {
			// Soft break at a space.
			lines = append(lines, string(rs[start:cut]))
			start = cut + 1
		}
	}

	if len(lines) > maxLines {
		lines = lines[len(lines)-maxLines:]
	}
	for i, line := range lines {
		printToScreen(screen, x, y+i, style, line)
	}
}
