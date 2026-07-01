//go:build !windows

package main

import _ "embed"

//go:embed assets/dictate.png
var iconDictatingBytes []byte

//go:embed assets/dictate-paused.png
var iconPausedBytes []byte

//go:embed assets/dictate-listening.png
var iconListeningBytes []byte

//go:embed assets/dictate-finalizing.png
var iconFinalizingBytes []byte

// iconForState maps an app state name (as returned by appState.String) to the
// embedded PNG icon bytes for the current platform. The tray library decodes
// PNG on Linux and accepts PNG (among others) via NSImage initWithData on
// macOS.
func iconForState(state string) []byte {
	switch state {
	case "PAUSED":
		return iconPausedBytes
	case "LISTENING":
		return iconListeningBytes
	case "FINALIZING":
		return iconFinalizingBytes
	default: // DICTATING
		return iconDictatingBytes
	}
}
