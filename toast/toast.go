// Package toast displays transient "toast" style system notifications.
//
// The notification backend is platform-specific. Init must be called once
// before Show; on platforms without a backend it is a no-op that returns nil.
package toast

import (
	"log"
	"strings"
	"sync"
)

var (
	cacheMu  sync.Mutex
	lastText string
)

// Init prepares the toast subsystem. It must be called once before any Show
// call. logger, if non-nil, receives debug output (e.g. the exact command
// run and its result); a nil logger disables diagnostics. On platforms that do
// not implement toast notifications it is a no-op that returns nil; on Linux
// it verifies that notify-send is available and returns an error otherwise.
func Init(logger *log.Logger) error { return initPlatform(logger) }

// Show displays a toast notification with the given title and message,
// replacing the previous notification from this process. When persist is true
// the notification is marked critical urgency (does not auto-dismiss);
// otherwise it is normal urgency. It returns an error if the notification
// could not be displayed. On platforms without a backend it is a no-op that
// returns nil.
//
// Show suppresses the notification when the message is identical to the last
// shown message (ignoring trailing whitespace), so redraws with unchanged
// dictation text do not create flashing or stacking notifications.
func Show(title, message string, persist bool) error {
	cacheMu.Lock()
	defer cacheMu.Unlock()

	text := strings.TrimRight(message, " \t\n\r")
	if text != "" && text == lastText {
		return nil
	}

	if err := show(title, message, persist); err != nil {
		return err
	}
	lastText = text
	return nil
}
