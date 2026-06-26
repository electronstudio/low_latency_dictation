//go:build linux

package hotkey

import (
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"syscall"

	"github.com/holoplot/go-evdev"
)

// openKeyboardDevices scans /dev/input/event* for keyboard-class devices that
// the process is permitted to read. It returns the opened devices (kept open,
// ready to read), the number of keyboard devices that were present but could
// not be opened due to permission denial (for diagnostics), or an error if the
// input directory itself cannot be enumerated.
//
// A device is considered a keyboard if it supports EV_KEY events and exposes
// at least one of a small set of keys that are universal on keyboards but
// absent on mice, power buttons, and other input devices (KEY_A, KEY_SPACE,
// KEY_ENTER, KEY_BACKSPACE).
func openKeyboardDevices() (devices []*evdev.InputDevice, permDenied int, err error) {
	entries, err := os.ReadDir("/dev/input")
	if err != nil {
		return nil, 0, fmt.Errorf("hotkey: cannot enumerate /dev/input: %w", err)
	}

	// Sort for deterministic logging/behaviour across runs.
	var eventNames []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if strings.HasPrefix(e.Name(), "event") {
			eventNames = append(eventNames, e.Name())
		}
	}
	sort.Strings(eventNames)

	disc := newKeySet(evdev.KEY_A, evdev.KEY_SPACE, evdev.KEY_ENTER, evdev.KEY_BACKSPACE)

	for _, name := range eventNames {
		path := "/dev/input/" + name
		dev, oerr := evdev.OpenWithFlags(path, os.O_RDONLY)
		if oerr != nil {
			if isPermission(oerr) {
				permDenied++
			}
			continue
		}

		if !isKeyboard(dev, disc) {
			_ = dev.Close()
			continue
		}
		devices = append(devices, dev)
	}
	return devices, permDenied, nil
}

// isKeyboard reports whether dev looks like a keyboard: it must support
// EV_KEY events and expose at least one of the discriminator keys.
func isKeyboard(dev *evdev.InputDevice, disc keySet) bool {
	hasEvKey := false
	for _, t := range dev.CapableTypes() {
		if t == evdev.EV_KEY {
			hasEvKey = true
			break
		}
	}
	if !hasEvKey {
		return false
	}
	for _, c := range dev.CapableEvents(evdev.EV_KEY) {
		if disc.has(c) {
			return true
		}
	}
	return false
}

type keySet map[evdev.EvCode]struct{}

func newKeySet(codes ...evdev.EvCode) keySet {
	s := make(keySet, len(codes))
	for _, c := range codes {
		s[c] = struct{}{}
	}
	return s
}

func (s keySet) has(c evdev.EvCode) bool {
	_, ok := s[c]
	return ok
}

// isPermission reports whether err is a permission-denied error from opening
// a device node.
func isPermission(err error) bool {
	return errors.Is(err, syscall.EACCES) || errors.Is(err, syscall.EPERM)
}
