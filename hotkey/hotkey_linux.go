//go:build linux

package hotkey

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/holoplot/go-evdev"
)

// ErrPermissionDenied is returned by Register on Linux when the process lacks
// read permission on the keyboard /dev/input/event* devices. The user must be
// added to the 'input' group (sudo usermod -aG input $USER) and re-login, or
// the process run with sufficient privileges.
var ErrPermissionDenied = errors.New("hotkey: permission denied reading keyboard devices")

type platformHotkey struct {
	mu         sync.Mutex
	registered bool

	keydownCh chan Event
	keyupCh   chan Event

	devices []*evdev.InputDevice
	wg      sync.WaitGroup
	stop    context.CancelFunc
}

func (p *platformHotkey) keydown() <-chan Event { return p.keydownCh }
func (p *platformHotkey) keyup() <-chan Event   { return p.keyupCh }

// register implements the Linux evdev backend.
//
// h.platformHotkey is nil on entry (Register creates a fresh *Hotkey), so we
// must not dereference promoted fields like h.mu until we have allocated and
// assigned the platformHotkey. All failure paths return before touching it.
func (h *Hotkey) register() error {
	targetCode, ok := linuxKeyMap[h.key]
	if !ok {
		return fmt.Errorf("hotkey: key %q is not supported by the evdev backend", keyName(h.key))
	}
	var requiredMask Modifier
	for _, m := range h.mods {
		requiredMask |= m
	}

	devices, permDenied, err := openKeyboardDevices()
	if err != nil {
		return err
	}
	if len(devices) == 0 {
		if permDenied > 0 {
			return fmt.Errorf(
				"%w\n\n"+
					"  Cannot read any keyboard /dev/input/event* device.\n"+
					"  Add your user to the 'input' group and then reboot:\n\n"+
					"      sudo usermod -aG input $USER\n\n"+
					"  Alternatively, disable hotkey feature:\n\n"+
					"      dictate --hotkey-key \"\"\n\n",
				ErrPermissionDenied)
		}
		return errors.New("hotkey: no keyboard input devices found")
	}

	ctx, cancel := context.WithCancel(context.Background())
	p := &platformHotkey{
		keydownCh:  make(chan Event, 1),
		keyupCh:    make(chan Event, 1),
		devices:    devices,
		stop:       cancel,
		registered: true,
	}
	h.platformHotkey = p

	for _, dev := range devices {
		p.wg.Add(1)
		go p.readLoop(ctx, dev, targetCode, requiredMask)
	}

	return nil
}

func (h *Hotkey) unregister() error {
	if h.platformHotkey == nil || !h.registered {
		return errNotRegistered
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if !h.registered {
		return errNotRegistered
	}
	h.stop()
	for _, dev := range h.devices {
		_ = dev.Close()
	}
	h.wg.Wait()
	h.registered = false
	return nil
}

// readLoop is run once per keyboard device. It tracks that device's held
// modifiers and emits the registered hotkey when its key+modifiers match.
// Modifier state is tracked per-device: a hotkey whose modifiers and key are
// pressed on different physical devices will not fire (an accepted
// limitation).
func (p *platformHotkey) readLoop(ctx context.Context, dev *evdev.InputDevice, targetCode evdev.EvCode, requiredMask Modifier) {
	defer p.wg.Done()

	// Make reads interruptible by Close() (see go-evdev NonBlock docs).
	_ = dev.NonBlock()

	var heldMods Modifier
	// down tracks whether we have reported a keydown that has not yet been
	// matched by a keyup, so we do not flood the channel on auto-repeat.
	var down bool

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}
		ev, err := dev.ReadOne()
		if err != nil {
			// Close() during a read surfaces as a read error; ctx is also
			// cancelled so just exit.
			select {
			case <-ctx.Done():
				return
			default:
			}
			continue
		}
		if ev.Type != evdev.EV_KEY {
			continue
		}

		// Modifier keys: update held set. Value: 1=down, 0=up, 2=repeat.
		if m, isMod := linuxModForCode(ev.Code); isMod {
			switch ev.Value {
			case 1:
				heldMods |= m
			case 0:
				heldMods &^= m
			}
			continue
		}

		// The target key itself.
		if ev.Code != targetCode {
			continue
		}
		switch ev.Value {
		case 1: // press
			// Exact modifier match (not subset): a hotkey registered as
			// Ctrl+Shift+D must not fire when Alt is also held, and a hotkey
			// with no modifiers must not fire when any modifier is held. This
			// matches the macOS (CGEventTap) and Windows (RegisterHotKey)
			// backends, which also require an exact match.
			if heldMods == requiredMask && !down {
				down = true
				select {
				case p.keydownCh <- Event{}:
				default:
				}
			}
		case 0: // release
			if down {
				down = false
				select {
				case p.keyupCh <- Event{}:
				default:
				}
			}
			// value 2 (repeat): ignore, we already fired on the initial press.
		}
	}
}

// linuxModForCode returns the Modifier bit (and true) for an evdev modifier
// keycode, or (0,false) if the code is not a modifier we track.
func linuxModForCode(code evdev.EvCode) (Modifier, bool) {
	switch code {
	case evdev.KEY_LEFTCTRL, evdev.KEY_RIGHTCTRL:
		return ModCtrl, true
	case evdev.KEY_LEFTSHIFT, evdev.KEY_RIGHTSHIFT:
		return ModShift, true
	case evdev.KEY_LEFTALT, evdev.KEY_RIGHTALT:
		return ModAlt, true
	case evdev.KEY_LEFTMETA, evdev.KEY_RIGHTMETA:
		return ModSuper, true
	}
	return 0, false
}

var linuxKeyMap = map[Key]evdev.EvCode{
	KeyA: evdev.KEY_A, KeyB: evdev.KEY_B, KeyC: evdev.KEY_C, KeyD: evdev.KEY_D,
	KeyE: evdev.KEY_E, KeyF: evdev.KEY_F, KeyG: evdev.KEY_G, KeyH: evdev.KEY_H,
	KeyI: evdev.KEY_I, KeyJ: evdev.KEY_J, KeyK: evdev.KEY_K, KeyL: evdev.KEY_L,
	KeyM: evdev.KEY_M, KeyN: evdev.KEY_N, KeyO: evdev.KEY_O, KeyP: evdev.KEY_P,
	KeyQ: evdev.KEY_Q, KeyR: evdev.KEY_R, KeyS: evdev.KEY_S, KeyT: evdev.KEY_T,
	KeyU: evdev.KEY_U, KeyV: evdev.KEY_V, KeyW: evdev.KEY_W, KeyX: evdev.KEY_X,
	KeyY: evdev.KEY_Y, KeyZ: evdev.KEY_Z,
	Key0: evdev.KEY_0, Key1: evdev.KEY_1, Key2: evdev.KEY_2, Key3: evdev.KEY_3,
	Key4: evdev.KEY_4, Key5: evdev.KEY_5, Key6: evdev.KEY_6, Key7: evdev.KEY_7,
	Key8: evdev.KEY_8, Key9: evdev.KEY_9,
	KeySpace:  evdev.KEY_SPACE,
	KeyReturn: evdev.KEY_ENTER,
	KeyEscape: evdev.KEY_ESC,
	KeyDelete: evdev.KEY_DELETE,
	KeyTab:    evdev.KEY_TAB,
	KeyF1:     evdev.KEY_F1,
	KeyF2:     evdev.KEY_F2,
	KeyF3:     evdev.KEY_F3,
	KeyF4:     evdev.KEY_F4,
	KeyF5:     evdev.KEY_F5,
	KeyF6:     evdev.KEY_F6,
	KeyF7:     evdev.KEY_F7,
	KeyF8:     evdev.KEY_F8,
	KeyF9:     evdev.KEY_F9,
	KeyF10:    evdev.KEY_F10,
	KeyF11:    evdev.KEY_F11,
	KeyF12:    evdev.KEY_F12,
	KeyLeft:   evdev.KEY_LEFT,
	KeyRight:  evdev.KEY_RIGHT,
	KeyUp:     evdev.KEY_UP,
	KeyDown:   evdev.KEY_DOWN,
}
