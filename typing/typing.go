// Package typing simulates a paste keystroke (Ctrl+V on Linux and Windows,
// Cmd+V on macOS) into the focused application, so that text previously
// placed on the system clipboard is delivered as if the user pasted it
// manually.
//
// Linux uses the uinput subsystem (works under both X11 and Wayland, and even
// from a TTY) and requires the user to be a member of the 'uinput' group (or
// otherwise have write access to /dev/uinput). macOS uses CoreGraphics
// CGEventPost and requires Accessibility permission. Windows uses the SendInput
// API (no extra permissions beyond normal user integrity).
package typing
