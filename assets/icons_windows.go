//go:build windows

package assets

import _ "embed"

//go:embed dictate.ico
var iconDictatingBytes []byte

//go:embed dictate-paused.ico
var iconPausedBytes []byte

//go:embed dictate-listening.ico
var iconListeningBytes []byte

//go:embed dictate-finalizing.ico
var iconFinalizingBytes []byte
