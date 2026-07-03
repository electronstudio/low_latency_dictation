package assets

// IconForState maps an app state name (as returned by State.String) to the
// embedded icon bytes for the current platform. The tray library decodes PNG on
// Linux, accepts PNG via NSImage on macOS, and expects ICO bytes on Windows.
func IconForState(state string) []byte {
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
