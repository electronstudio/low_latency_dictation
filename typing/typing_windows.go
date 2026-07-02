//go:build windows

package typing

import (
	"fmt"
	"syscall"
	"unsafe"
)

const (
	inputKeyboard  = 1
	vkControl      = 0x11
	vkV            = 0x56
	keyeventfKeyup = 0x0002
)

// tagKEYBDINPUT mirrors Win32 KEYBDINPUT. dwExtraInfo is ULONG_PTR, so the
// struct layout (and therefore tagINPUT below) differs between 32- and 64-bit
// Windows exactly as the native INPUT does.
type tagKEYBDINPUT struct {
	wVk         uint16
	wScan       uint16
	dwFlags     uint32
	time        uint32
	dwExtraInfo uintptr
}

// tagINPUT mirrors Win32 INPUT. The pad field pads the anonymous union out to
// the size of MOUSEINPUT (the largest member) so unsafe.Sizeof matches
// sizeof(INPUT) on both 32-bit (28 bytes) and 64-bit (40 bytes) Windows.
type tagINPUT struct {
	type_ uint32
	ki    tagKEYBDINPUT
	pad   [8]byte
}

var (
	modUser32     = syscall.NewLazyDLL("user32.dll")
	procSendInput = modUser32.NewProc("SendInput")
)

// Init is a no-op on Windows; paste is handled via SendInput on demand.
func Init() error { return nil }

// Close is a no-op on Windows.
func Close() {}

// Paste simulates a Ctrl+V keystroke via the SendInput API.
func Paste() error {
	inputs := []tagINPUT{
		{type_: inputKeyboard, ki: tagKEYBDINPUT{wVk: vkControl}},
		{type_: inputKeyboard, ki: tagKEYBDINPUT{wVk: vkV}},
		{type_: inputKeyboard, ki: tagKEYBDINPUT{wVk: vkV, dwFlags: keyeventfKeyup}},
		{type_: inputKeyboard, ki: tagKEYBDINPUT{wVk: vkControl, dwFlags: keyeventfKeyup}},
	}
	n, _, e := procSendInput.Call(
		uintptr(len(inputs)),
		uintptr(unsafe.Pointer(&inputs[0])),
		unsafe.Sizeof(inputs[0]),
	)
	if n == 0 {
		return fmt.Errorf("typing: SendInput failed: %v", e)
	}
	return nil
}
