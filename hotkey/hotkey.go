// Package hotkey provides a uniform, cross-platform global hotkey API.
//
// On Linux it reads keyboard events via evdev (works under both X11 and
// Wayland, and even from a TTY) and requires the user to be a member of the
// 'input' group or otherwise have read permission on /dev/input/event*.
// On macOS and Windows it wraps golang.design/x/hotkey (CGEventTap and
// RegisterHotKey respectively).
//
// A hotkey is a combination of one or more modifiers and a single key.
package hotkey

import (
	"errors"
	"fmt"
	"strings"
)

// Errors shared across all platform backends.
var (
	errAlreadyRegistered = errors.New("hotkey: already registered")
	errNotRegistered     = errors.New("hotkey: not registered")
)

// Modifier is a bit-flag describing a hotkey modifier. The values are
// platform-agnostic; each backend translates them to the native mask.
type Modifier uint8

const (
	ModCtrl  Modifier = 1 << 0
	ModShift Modifier = 1 << 1
	ModAlt   Modifier = 1 << 2
	ModSuper Modifier = 1 << 3 // Cmd on macOS, Win on Windows, Super/Meta on Linux
)

// Key is a platform-agnostic key identifier. Each backend translates it to
// the native keycode/keysym.
type Key uint16

const (
	KeyUnknown Key = 0
	KeyA       Key = iota + 1
	KeyB
	KeyC
	KeyD
	KeyE
	KeyF
	KeyG
	KeyH
	KeyI
	KeyJ
	KeyK
	KeyL
	KeyM
	KeyN
	KeyO
	KeyP
	KeyQ
	KeyR
	KeyS
	KeyT
	KeyU
	KeyV
	KeyW
	KeyX
	KeyY
	KeyZ
	Key0
	Key1
	Key2
	Key3
	Key4
	Key5
	Key6
	Key7
	Key8
	Key9
	KeySpace
	KeyReturn
	KeyEscape
	KeyDelete
	KeyTab
	KeyF1
	KeyF2
	KeyF3
	KeyF4
	KeyF5
	KeyF6
	KeyF7
	KeyF8
	KeyF9
	KeyF10
	KeyF11
	KeyF12
	KeyLeft
	KeyRight
	KeyUp
	KeyDown
)

// Event is delivered when a registered hotkey is triggered or released.
type Event struct{}

// Hotkey is a registered system-wide global hotkey. The zero value is not
// usable; construct one with Register.
type Hotkey struct {
	// platformHotkey is set by each backend's register method. It is a
	// pointer so the platform struct (which may contain a sync.Mutex) is
	// never copied.
	*platformHotkey

	mods []Modifier
	key  Key
}

// Mods returns the modifiers of this hotkey.
func (h *Hotkey) Mods() []Modifier { return h.mods }

// Key returns the key of this hotkey.
func (h *Hotkey) Key() Key { return h.key }

// Register registers the given modifier+key combination as a system-wide
// global hotkey and returns a Hotkey that can be used to wait for events.
//
// On Linux the backend is evdev; if the process lacks permission to read any
// keyboard /dev/input/event* device (e.g. the user is not in the 'input'
// group), Register returns an error wrapping ErrPermissionDenied.
//
// On macOS, Register returns an error if the process is not trusted for
// Accessibility (Input Monitoring); grant it in System Settings → Privacy &
// Security → Accessibility.
func Register(mods []Modifier, key Key) (*Hotkey, error) {
	if key == KeyUnknown {
		return nil, fmt.Errorf("hotkey: cannot register an unknown key")
	}
	h := &Hotkey{mods: mods, key: key}
	if err := h.register(); err != nil {
		return nil, err
	}
	return h, nil
}

// Keydown returns a channel that receives an Event when the hotkey is
// pressed. The channel is buffered and never blocks the sender.
func (h *Hotkey) Keydown() <-chan Event { return h.keydown() }

// Keyup returns a channel that receives an Event when the hotkey is
// released. The channel is buffered and never blocks the sender.
func (h *Hotkey) Keyup() <-chan Event { return h.keyup() }

// Unregister releases the system resources held by the hotkey.
func (h *Hotkey) Unregister() error { return h.unregister() }

// String returns a human-readable representation, e.g. "ctrl+shift+d".
func (h *Hotkey) String() string {
	var b strings.Builder
	for i, m := range h.mods {
		if i > 0 {
			b.WriteByte('+')
		}
		b.WriteString(modName(m))
	}
	if len(h.mods) > 0 {
		b.WriteByte('+')
	}
	b.WriteString(keyName(h.key))
	return b.String()
}

func modName(m Modifier) string {
	switch m {
	case ModCtrl:
		return "ctrl"
	case ModShift:
		return "shift"
	case ModAlt:
		return "alt"
	case ModSuper:
		return "super"
	}
	return fmt.Sprintf("mod%d", m)
}

func keyName(k Key) string {
	if n, ok := keyNames[k]; ok {
		return n
	}
	return fmt.Sprintf("key%d", k)
}

var keyNames = map[Key]string{
	KeyA: "a", KeyB: "b", KeyC: "c", KeyD: "d", KeyE: "e", KeyF: "f",
	KeyG: "g", KeyH: "h", KeyI: "i", KeyJ: "j", KeyK: "k", KeyL: "l",
	KeyM: "m", KeyN: "n", KeyO: "o", KeyP: "p", KeyQ: "q", KeyR: "r",
	KeyS: "s", KeyT: "t", KeyU: "u", KeyV: "v", KeyW: "w", KeyX: "x",
	KeyY: "y", KeyZ: "z",
	Key0: "0", Key1: "1", Key2: "2", Key3: "3", Key4: "4",
	Key5: "5", Key6: "6", Key7: "7", Key8: "8", Key9: "9",
	KeySpace: "space", KeyReturn: "return", KeyEscape: "escape",
	KeyDelete: "delete", KeyTab: "tab",
	KeyF1: "f1", KeyF2: "f2", KeyF3: "f3", KeyF4: "f4", KeyF5: "f5",
	KeyF6: "f6", KeyF7: "f7", KeyF8: "f8", KeyF9: "f9", KeyF10: "f10",
	KeyF11: "f11", KeyF12: "f12",
	KeyLeft: "left", KeyRight: "right", KeyUp: "up", KeyDown: "down",
}

// ParseCombo parses a hotkey specification from two CLI-style strings: a
// modifiers string (e.g. "ctrl+shift", case-insensitive, '+' separated, may
// be empty) and a key string (e.g. "d", "f1", "space").
//
// Accepted modifier names: ctrl, control, shift, alt, option, cmd, command,
// super, meta, win, windows. "cmd"/"command"/"super"/"meta"/"win"/"windows"
// all map to ModSuper.
func ParseCombo(modsStr, keyStr string) ([]Modifier, Key, error) {
	keyStr = strings.ToLower(strings.TrimSpace(keyStr))
	if keyStr == "" {
		return nil, KeyUnknown, fmt.Errorf("hotkey: empty key")
	}
	key, ok := nameToKey[keyStr]
	if !ok {
		return nil, KeyUnknown, fmt.Errorf("hotkey: unknown key %q (try e.g. d, f1, space, escape)", keyStr)
	}

	var mods []Modifier
	if modsStr = strings.TrimSpace(modsStr); modsStr != "" {
		for _, part := range strings.Split(modsStr, "+") {
			name := strings.ToLower(strings.TrimSpace(part))
			if name == "" {
				continue
			}
			m, ok := nameToMod[name]
			if !ok {
				return nil, KeyUnknown, fmt.Errorf("hotkey: unknown modifier %q", name)
			}
			// Dedup so "ctrl+ctrl" does not double-set.
			var present bool
			for _, e := range mods {
				if e == m {
					present = true
					break
				}
			}
			if !present {
				mods = append(mods, m)
			}
		}
	}
	return mods, key, nil
}

var nameToMod = map[string]Modifier{
	"ctrl": ModCtrl, "control": ModCtrl,
	"shift": ModShift,
	"alt":   ModAlt, "option": ModAlt,
	"cmd": ModSuper, "command": ModSuper,
	"super": ModSuper, "meta": ModSuper,
	"win": ModSuper, "windows": ModSuper,
}

var nameToKey = func() map[string]Key {
	m := make(map[string]Key, len(keyNames))
	for k, n := range keyNames {
		m[n] = k
	}
	return m
}()
