//go:build !windows

package tray

import "fyne.io/systray"

// start registers the tray using RunWithExternalLoop and invokes start on the
// platform main thread via dispatch. On macOS the NSStatusItem must be created
// on the main thread, which already runs [NSApp run] via mainthread.Init; on
// Linux/BSD the DBus backend runs in goroutines and the dispatch hop is
// harmless. The returned end function is stored for Quit.
func (t *Tray) start() error {
	start, end := systray.RunWithExternalLoop(t.onReady, t.onExit)
	t.endFn = end
	t.dispatch(start)
	return nil
}

// Quit tears the tray down. On non-Windows platforms it runs the end function
// (nativeEnd + Quit) on the main thread so the NSStatusItem is removed and the
// DBus name is released cleanly.
func (t *Tray) Quit() {
	if t.endFn != nil {
		t.dispatch(t.endFn)
	}
}
