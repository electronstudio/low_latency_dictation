package hotkey

import (
	"reflect"
	"testing"
)

// modMask converts a slice of Modifier bits to the combined mask used by the
// evdev backend's exact-match check.
func modMask(mods []Modifier) Modifier {
	var m Modifier
	for _, x := range mods {
		m |= x
	}
	return m
}

func TestParseCombo(t *testing.T) {
	tests := []struct {
		name     string
		mods     string
		key      string
		wantMods []Modifier
		wantKey  Key
		wantErr  bool
	}{
		{"ctrl+shift d", "ctrl+shift", "d", []Modifier{ModCtrl, ModShift}, KeyD, false},
		{"default combo", "ctrl+shift", "d", []Modifier{ModCtrl, ModShift}, KeyD, false},
		{"no modifiers", "", "f1", nil, KeyF1, false},
		{"single mod", "alt", "space", []Modifier{ModAlt}, KeySpace, false},
		{"control alias", "control", "a", []Modifier{ModCtrl}, KeyA, false},
		{"option alias for alt", "option", "b", []Modifier{ModAlt}, KeyB, false},
		{"cmd maps to super", "cmd", "c", []Modifier{ModSuper}, KeyC, false},
		{"win maps to super", "win", "v", []Modifier{ModSuper}, KeyV, false},
		{"super maps to super", "super", "x", []Modifier{ModSuper}, KeyX, false},
		{"meta maps to super", "meta", "z", []Modifier{ModSuper}, KeyZ, false},
		{"dedups same mod", "ctrl+ctrl", "e", []Modifier{ModCtrl}, KeyE, false},
		{"dedups cmd+win (both super)", "cmd+win", "f", []Modifier{ModSuper}, KeyF, false},
		{"three mods", "ctrl+shift+alt", "g", []Modifier{ModCtrl, ModShift, ModAlt}, KeyG, false},
		{"case insensitive mods", "CTRL+Shift", "h", []Modifier{ModCtrl, ModShift}, KeyH, false},
		{"case insensitive key", "ctrl+shift", "D", []Modifier{ModCtrl, ModShift}, KeyD, false},
		{"whitespace trimmed", " ctrl + shift ", " d ", []Modifier{ModCtrl, ModShift}, KeyD, false},
		{"numbers", "ctrl", "5", []Modifier{ModCtrl}, Key5, false},
		{"escape key", "ctrl", "escape", []Modifier{ModCtrl}, KeyEscape, false},
		{"arrow key", "ctrl", "left", []Modifier{ModCtrl}, KeyLeft, false},
		{"empty key", "ctrl", "", nil, KeyUnknown, true},
		{"unknown key", "ctrl", "xyz", nil, KeyUnknown, true},
		{"unknown modifier", "foo", "d", nil, KeyUnknown, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mods, key, err := ParseCombo(tt.mods, tt.key)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseCombo(%q,%q) err = %v, wantErr=%v", tt.mods, tt.key, err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			if !reflect.DeepEqual(mods, tt.wantMods) {
				t.Errorf("ParseCombo(%q,%q) mods = %v, want %v", tt.mods, tt.key, mods, tt.wantMods)
			}
			if key != tt.wantKey {
				t.Errorf("ParseCombo(%q,%q) key = %v, want %v", tt.mods, tt.key, key, tt.wantKey)
			}
		})
	}
}

func TestHotkeyString(t *testing.T) {
	tests := []struct {
		h    *Hotkey
		want string
	}{
		{&Hotkey{mods: []Modifier{ModCtrl, ModShift}, key: KeyD}, "ctrl+shift+d"},
		{&Hotkey{mods: []Modifier{ModSuper}, key: KeyF1}, "super+f1"},
		{&Hotkey{mods: nil, key: KeySpace}, "space"},
	}
	for _, tt := range tests {
		if got := tt.h.String(); got != tt.want {
			t.Errorf("String() = %q, want %q", got, tt.want)
		}
	}
}

// TestModifierMatch verifies the exact-modifier-match semantics shared by all
// backends: a hotkey fires only when the held modifiers equal the required
// set (no extra modifiers), and a no-modifier hotkey fires only when no
// modifiers are held. This guards against the earlier Linux subset-match bug.
func TestModifierMatch(t *testing.T) {
	tests := []struct {
		name     string
		required []Modifier
		held     Modifier
		want     bool
	}{
		// No modifiers required: must fire only with an empty held set.
		{"no-mod, none held", nil, 0, true},
		{"no-mod, ctrl held", nil, ModCtrl, false},
		{"no-mod, shift held", nil, ModShift, false},
		// Single modifier required: exact match only.
		{"ctrl, ctrl held", []Modifier{ModCtrl}, ModCtrl, true},
		{"ctrl, none held", []Modifier{ModCtrl}, 0, false},
		{"ctrl, ctrl+shift held", []Modifier{ModCtrl}, ModCtrl | ModShift, false},
		// Two modifiers required: exact match only.
		{"ctrl+shift, both held", []Modifier{ModCtrl, ModShift}, ModCtrl | ModShift, true},
		{"ctrl+shift, ctrl only", []Modifier{ModCtrl, ModShift}, ModCtrl, false},
		{"ctrl+shift, ctrl+alt", []Modifier{ModCtrl, ModShift}, ModCtrl | ModAlt, false},
		{"ctrl+shift, all three", []Modifier{ModCtrl, ModShift}, ModCtrl | ModShift | ModAlt, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// This mirrors readLoop's check: heldMods == requiredMask.
			requiredMask := modMask(tt.required)
			if got := tt.held == requiredMask; got != tt.want {
				t.Errorf("held=%b == required=%b => %v, want %v", tt.held, requiredMask, got, tt.want)
			}
		})
	}
}
