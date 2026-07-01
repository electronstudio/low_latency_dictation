//go:build windows

package main

import _ "embed"

//go:embed assets/dictate.ico
var iconDictatingBytes []byte

//go:embed assets/dictate-paused.ico
var iconPausedBytes []byte

//go:embed assets/dictate-listening.ico
var iconListeningBytes []byte

//go:embed assets/dictate-finalizing.ico
var iconFinalizingBytes []byte

// iconForState maps an app state name (as returned by appState.String) to the
// embedded ICO icon bytes for Windows. The tray library writes the bytes to a
// temp file and loads them with LoadImageW(IMAGE_ICON), which expects .ico.
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
