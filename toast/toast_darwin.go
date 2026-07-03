//go:build darwin

package toast

import "log"

// Toast notifications are not yet implemented on macOS.
func initPlatform(logger *log.Logger) error          { return nil }
func show(title, message string, persist bool) error { return nil }
