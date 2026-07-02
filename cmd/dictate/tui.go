package main

import (
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/gdamore/tcell/v2"
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
}

// NewTerminalUI creates a new terminal UI instance.
func NewTerminalUI(hotkeyLabel string) *TerminalUI {
	return &TerminalUI{
		hotkeyLabel:     hotkeyLabel,
		pauseCh:         make(chan struct{}, 1),
		deleteCh:        make(chan struct{}, 1),
		quitCh:          make(chan struct{}, 1),
		stop:            make(chan struct{}),
		statusStyle:     tcell.StyleDefault.Reverse(true),
		finalizingStyle: tcell.StyleDefault.Foreground(tcell.ColorRed),
		finalizedStyle:  tcell.StyleDefault.Foreground(tcell.ColorGreen),
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
func (t *TerminalUI) ShowStatus(status string) {
	if t.height < 1 {
		return
	}
	s := "[" + t.hotkeyLabel + "] start/end/paste [P]ause [Q]uit [D]elete   <" + status + ">"
	if t.width > len(s) {
		s += strings.Repeat(" ", t.width-len(s))
	}
	printToScreen(t.screen, 0, t.height-1, t.statusStyle, s)
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
	printWrapped(t.screen, 0, 0, t.width, t.height-1, style, strings.TrimSpace(text))
	t.ShowStatus(state.String())
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
			if ev.Key() == tcell.KeyRune {
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
	for i, r := range text {
		screen.SetContent(x+i, y, r, nil, style)
	}
}

func printWrapped(screen tcell.Screen, x, y, maxWidth, maxLines int, style tcell.Style, text string) {
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
		printToScreen(screen, x, y+i, style, line)
	}
}
