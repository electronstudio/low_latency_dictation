//go:build darwin

package typing

/*
#include <ApplicationServices/ApplicationServices.h>
#include <unistd.h>

// darwinPaste simulates a Cmd+V keystroke (paste) into the focused
// application via the CoreGraphics event tap.
static void darwinPaste(void) {
	CGEventSourceRef src = CGEventSourceCreate(kCGEventSourceStateHIDSystemState);
	if (src == NULL) {
		return;
	}

	// Press the left Command key (virtual keycode 55).
	CGEventRef cmdDown = CGEventCreateKeyboardEvent(src, 55, true);
	if (cmdDown != NULL) {
		CGEventPost(kCGHIDEventTap, cmdDown);
		CFRelease(cmdDown);
	}
	usleep(5000);

	// Press 'V' (virtual keycode 9) with the Command modifier flag set.
	CGEventRef vDown = CGEventCreateKeyboardEvent(src, 9, true);
	if (vDown != NULL) {
		CGEventSetFlags(vDown, kCGEventFlagMaskCommand);
		CGEventPost(kCGHIDEventTap, vDown);
		CFRelease(vDown);
	}
	usleep(5000);

	// Release 'V' with the Command modifier flag still set.
	CGEventRef vUp = CGEventCreateKeyboardEvent(src, 9, false);
	if (vUp != NULL) {
		CGEventSetFlags(vUp, kCGEventFlagMaskCommand);
		CGEventPost(kCGHIDEventTap, vUp);
		CFRelease(vUp);
	}
	usleep(5000);

	// Release the left Command key.
	CGEventRef cmdUp = CGEventCreateKeyboardEvent(src, 55, false);
	if (cmdUp != NULL) {
		CGEventPost(kCGHIDEventTap, cmdUp);
		CFRelease(cmdUp);
	}

	CFRelease(src);
}

// darwinTrusted reports whether the process has been granted Accessibility
// permission, which is required for CGEventPost to deliver synthesized
// keystrokes.
static int darwinTrusted(void) {
	return AXIsProcessTrusted() ? 1 : 0;
}
*/
import "C"

import "fmt"

// Init is a no-op on macOS; paste is handled via CoreGraphics on demand.
func Init() error { return nil }

// Close is a no-op on macOS.
func Close() {}

// Paste simulates a Cmd+V keystroke via CoreGraphics. It requires macOS
// Accessibility permission (System Settings, Privacy & Security, Accessibility).
func Paste() error {
	if C.darwinTrusted() == 0 {
		return fmt.Errorf("typing: macOS Accessibility permission not granted; enable this app in System Settings, Privacy & Security, Accessibility so synthesized keystrokes are delivered")
	}
	C.darwinPaste()
	return nil
}
