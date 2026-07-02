//go:build windows

package tray

import "fyne.io/systray"

// start runs the tray in a dedicated goroutine. Unlike the external-loop path,
// systray.Run keeps the hidden message window and its Win32 message pump on the
// same goroutine, which is required for tray clicks to be delivered on Windows.
func (t *Tray) start() error {
	go systray.Run(t.onReady, t.onExit)
	return nil
}

// Quit requests the tray goroutine to exit.
func (t *Tray) Quit() {
	systray.Quit()
}
