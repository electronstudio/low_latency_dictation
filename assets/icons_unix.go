//go:build !windows

package assets

import _ "embed"

//go:embed dictate.png
var iconDictatingBytes []byte

//go:embed dictate-paused.png
var iconPausedBytes []byte

//go:embed dictate-listening.png
var iconListeningBytes []byte

//go:embed dictate-finalizing.png
var iconFinalizingBytes []byte
