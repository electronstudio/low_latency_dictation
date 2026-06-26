package hotkey

import (
	"reflect"
	"testing"
)

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
