//go:build darwin || windows

package hotkey

import (
	"fmt"

	gohk "golang.design/x/hotkey"
)

// platformHotkey holds the upstream golang.design/x/hotkey handle and the
// channels we expose to callers. Used by both the macOS and Windows backends.
type platformHotkey struct {
	inner     *gohk.Hotkey
	keydownCh chan Event
	keyupCh   chan Event
	stop      chan struct{}
}

func (p *platformHotkey) keydown() <-chan Event { return p.keydownCh }
func (p *platformHotkey) keyup() <-chan Event   { return p.keyupCh }

// registerInner wraps an already-constructed upstream hotkey: it registers
// it, stores the handle, and starts a forwarder goroutine that bridges the
// upstream event channels onto our own (so the rest of the program consumes
// our uniform Event type).
func (h *Hotkey) registerInner(inner *gohk.Hotkey) error {
	if err := inner.Register(); err != nil {
		return err
	}
	h.platformHotkey = &platformHotkey{
		inner:     inner,
		keydownCh: make(chan Event, 1),
		keyupCh:   make(chan Event, 1),
		stop:      make(chan struct{}),
	}
	go h.forward()
	return nil
}

func (h *Hotkey) forward() {
	kd := h.inner.Keydown()
	ku := h.inner.Keyup()
	for {
		select {
		case <-h.stop:
			return
		case _, ok := <-kd:
			if !ok {
				return
			}
			select {
			case h.keydownCh <- Event{}:
			default:
			}
		case _, ok := <-ku:
			if !ok {
				return
			}
			select {
			case h.keyupCh <- Event{}:
			default:
			}
		}
	}
}

func (h *Hotkey) unregister() error {
	if h.platformHotkey == nil || h.inner == nil {
		return errNotRegistered
	}
	close(h.stop)
	err := h.inner.Unregister()
	h.inner = nil
	return err
}

// nativeMods translates our platform-agnostic Modifier bits into the
// upstream library's Modifier slice. The per-keycode mapping of ModSuper
// (Cmd/Win) is handled by the caller's mod map.
func nativeMods(mods []Modifier, modMap map[Modifier]gohk.Modifier) []gohk.Modifier {
	out := make([]gohk.Modifier, 0, len(mods))
	for _, m := range mods {
		if nm, ok := modMap[m]; ok {
			out = append(out, nm)
		}
	}
	return out
}

// unsupportedKeyErr returns a formatted error for a key absent on a platform.
func unsupportedKeyErr(k Key) error {
	return fmt.Errorf("hotkey: key %q is not supported by this platform", keyName(k))
}
