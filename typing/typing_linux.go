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

// pasteDevice is the persistent virtual keyboard used for all paste
// operations. It is created once by Init and closed by Close.
var pasteDevice *evdev.InputDevice

// Init creates the persistent uinput virtual keyboard. On Linux this must be
// called once before the first Paste; on other platforms it is a no-op.
func Init() error {
	if pasteDevice != nil {
		return nil
	}

	dev, err := createPasteDevice()
	if err != nil {
		if errors.Is(err, syscall.EACCES) {
			return fmt.Errorf(
				"%w\n\n"+
					"  Cannot open /dev/uinput.\n"+
					"  Add your user to the 'uinput' group and then reboot:\n\n"+
					"      sudo usermod -aG uinput $USER\n\n"+
					"  Alternatively, disable hotkey feature and paste manually:\n\n"+
					"      dictate --hotkey-key \"\"\n\n",
				errUinputPermission)
		}
		return fmt.Errorf("typing: could not create virtual keyboard: %w", err)
	}

	pasteDevice = dev
	return nil
}

// createPasteDevice opens /dev/uinput and registers a virtual keyboard with
// the minimum keys needed for Ctrl+V. It sleeps briefly after creation so the
// display server has time to attach to the new device.
func createPasteDevice() (*evdev.InputDevice, error) {
	dev, err := evdev.CreateDevice(
		"low_latency_dictation",
		evdev.InputID{BusType: 0x03, Vendor: 0x1, Product: 0x1, Version: 1},
		map[evdev.EvType][]evdev.EvCode{
			evdev.EV_KEY: {evdev.KEY_LEFTCTRL, evdev.KEY_V},
		},
	)
	if err != nil {
		return nil, err
	}

	// Give the display server time to attach to the newly created device so
	// the first events are not lost.
	time.Sleep(80 * time.Millisecond)
	return dev, nil
}

// Close tears down the persistent virtual keyboard, if any.
func Close() {
	if pasteDevice == nil {
		return
	}
	_ = evdev.DestroyDevice(pasteDevice)
	_ = pasteDevice.Close()
	pasteDevice = nil
}

// Paste simulates a Ctrl+V keystroke using the persistent uinput virtual
// keyboard. It works under both X11 and Wayland (and from a TTY) but requires
// write access to /dev/uinput (see errUinputPermission).
func Paste() error {
	if pasteDevice == nil {
		if err := Init(); err != nil {
			return err
		}
	}

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
		if err := pasteDevice.WriteOne(&evdev.InputEvent{
			Time:  tv,
			Type:  evdev.EV_KEY,
			Code:  s.code,
			Value: s.value,
		}); err != nil {
			return fmt.Errorf("typing: failed to write key event: %w", err)
		}
		if err := pasteDevice.WriteOne(&evdev.InputEvent{
			Time:  tv,
			Type:  evdev.EV_SYN,
			Code:  evdev.SYN_REPORT,
			Value: 0,
		}); err != nil {
			return fmt.Errorf("typing: failed to write sync event: %w", err)
		}
		time.Sleep(5 * time.Millisecond)
	}

	return nil
}
