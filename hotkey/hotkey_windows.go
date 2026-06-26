//go:build windows

package hotkey

import (
	gohk "golang.design/x/hotkey"
)

// windowsModMap maps our platform-agnostic Modifier bits to the Windows
// RegisterHotKey modifier values used by golang.design/x/hotkey.
var windowsModMap = map[Modifier]gohk.Modifier{
	ModCtrl:  gohk.ModCtrl,
	ModShift: gohk.ModShift,
	ModAlt:   gohk.ModAlt,
	ModSuper: gohk.ModWin,
}

// windowsKeyMap maps our platform-agnostic Key to Windows virtual-key codes.
var windowsKeyMap = map[Key]gohk.Key{
	KeyA: gohk.KeyA, KeyB: gohk.KeyB, KeyC: gohk.KeyC, KeyD: gohk.KeyD,
	KeyE: gohk.KeyE, KeyF: gohk.KeyF, KeyG: gohk.KeyG, KeyH: gohk.KeyH,
	KeyI: gohk.KeyI, KeyJ: gohk.KeyJ, KeyK: gohk.KeyK, KeyL: gohk.KeyL,
	KeyM: gohk.KeyM, KeyN: gohk.KeyN, KeyO: gohk.KeyO, KeyP: gohk.KeyP,
	KeyQ: gohk.KeyQ, KeyR: gohk.KeyR, KeyS: gohk.KeyS, KeyT: gohk.KeyT,
	KeyU: gohk.KeyU, KeyV: gohk.KeyV, KeyW: gohk.KeyW, KeyX: gohk.KeyX,
	KeyY: gohk.KeyY, KeyZ: gohk.KeyZ,
	Key0: gohk.Key0, Key1: gohk.Key1, Key2: gohk.Key2, Key3: gohk.Key3,
	Key4: gohk.Key4, Key5: gohk.Key5, Key6: gohk.Key6, Key7: gohk.Key7,
	Key8: gohk.Key8, Key9: gohk.Key9,
	KeySpace:  gohk.KeySpace,
	KeyReturn: gohk.KeyReturn,
	KeyEscape: gohk.KeyEscape,
	KeyDelete: gohk.KeyDelete,
	KeyTab:    gohk.KeyTab,
	KeyF1:     gohk.KeyF1,
	KeyF2:     gohk.KeyF2,
	KeyF3:     gohk.KeyF3,
	KeyF4:     gohk.KeyF4,
	KeyF5:     gohk.KeyF5,
	KeyF6:     gohk.KeyF6,
	KeyF7:     gohk.KeyF7,
	KeyF8:     gohk.KeyF8,
	KeyF9:     gohk.KeyF9,
	KeyF10:    gohk.KeyF10,
	KeyF11:    gohk.KeyF11,
	KeyF12:    gohk.KeyF12,
	KeyLeft:   gohk.KeyLeft,
	KeyRight:  gohk.KeyRight,
	KeyUp:     gohk.KeyUp,
	KeyDown:   gohk.KeyDown,
}

func (h *Hotkey) register() error {
	key, ok := windowsKeyMap[h.key]
	if !ok {
		return unsupportedKeyErr(h.key)
	}
	mods := nativeMods(h.mods, windowsModMap)
	return h.registerInner(gohk.New(mods, key))
}
