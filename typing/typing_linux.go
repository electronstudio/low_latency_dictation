//go:build linux

package typing

import (
	"errors"
	"fmt"
	"syscall"
	"time"

	"github.com/holoplot/go-evdev"
)

var errUinputPermission = errors.New("typing: permission denied opening /dev/uinput")

// Paste simulates a Ctrl+V keystroke by creating a transient uinput virtual
// keyboard, emitting the key sequence, and destroying the device. It works
// under both X11 and Wayland (and from a TTY) but requires write access to
// /dev/uinput (see errUinputPermission).
func Paste() error {
	dev, err := evdev.CreateDevice(
		"low_latency_dictation",
		evdev.InputID{BusType: 0x03, Vendor: 0x1, Product: 0x1, Version: 1},
		map[evdev.EvType][]evdev.EvCode{
			evdev.EV_KEY: {evdev.KEY_LEFTCTRL, evdev.KEY_V},
		},
	)
	if err != nil {
		if errors.Is(err, syscall.EACCES) {
			return fmt.Errorf(
				"%w\n\n"+
					"  Cannot open /dev/uinput.\n"+
					"  Add your user to the 'uinput' group and log out/in (or reboot):\n\n"+
					"      sudo usermod -aG uinput $USER\n\n"+
					"  This is required to synthesize keystrokes via the Linux uinput\n"+
					"  subsystem, which works under both X11 and Wayland but needs write\n"+
					"  access to /dev/uinput.",
				errUinputPermission)
		}
		return fmt.Errorf("typing: could not create virtual keyboard: %w", err)
	}
	// Destroy the device before closing its file descriptor (defers run LIFO).
	defer dev.Close()
	defer evdev.DestroyDevice(dev)

	// Give the display server time to attach to the newly created device so
	// the first events are not lost.
	time.Sleep(80 * time.Millisecond)

	// Ctrl down, V down, V up, Ctrl up. Each key event is followed by a
	// SYN_REPORT so the kernel delivers it as a complete frame.
	seq := []struct {
		code  evdev.EvCode
		value int32
	}{
		{evdev.KEY_LEFTCTRL, 1},
		{evdev.KEY_V, 1},
		{evdev.KEY_V, 0},
		{evdev.KEY_LEFTCTRL, 0},
	}
	for _, s := range seq {
		tv := syscall.NsecToTimeval(time.Now().UnixNano())
		if err := dev.WriteOne(&evdev.InputEvent{
			Time:  tv,
			Type:  evdev.EV_KEY,
			Code:  s.code,
			Value: s.value,
		}); err != nil {
			return fmt.Errorf("typing: failed to write key event: %w", err)
		}
		if err := dev.WriteOne(&evdev.InputEvent{
			Time:  tv,
			Type:  evdev.EV_SYN,
			Code:  evdev.SYN_REPORT,
			Value: 0,
		}); err != nil {
			return fmt.Errorf("typing: failed to write sync event: %w", err)
		}
		time.Sleep(5 * time.Millisecond)
	}

	// Let the events flush before the device is torn down.
	time.Sleep(40 * time.Millisecond)
	return nil
}
