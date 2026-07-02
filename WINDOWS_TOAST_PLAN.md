# Plan: Windows Toast Notifications for `low_latency_dictation`

> **Status:** Not yet implemented. This file is a complete, self-contained brief
> for a future agent to implement Windows toast notifications in this repo.
> Read this whole file before starting; the "Why" sections matter as much as the "What".

---

## 0. TL;DR for the implementer

Add a real Windows backend to the existing `toast/` package, keeping the public
API (`toast.Init`, `toast.Show`) unchanged. The backend speaks the Windows
Runtime (WinRT) toast COM API **directly in pure Go** via
`github.com/go-ole/go-ole` (no CGO, no PowerShell spawn, no external binary).

Key behavioral requirements (decided with the maintainer — do not change without asking):

1. **Keep firing a toast on every screen redraw** (do NOT change `main.go`).
2. **Single updating toast**: hide the previous toast before showing the next,
   so Action Center does not fill up with stale toasts. Retain the previous
   `ToastNotification` COM handle between calls.
3. **Silent audio** (`<audio silent="true"/>`).
4. **AUMID via registry only**: write
   `HKCU\SOFTWARE\Classes\AppUserModelId\<AppID>\DisplayName`. Do NOT add Start
   Menu shortcut / installer changes. Icon may be generic — accepted.

`persist` maps to `duration` attribute: `true` → `"long"`, `false` → `"short"`
(mirrors Linux's `--expire-time` 1000000ms vs 5000ms). If the maintainer later
asks for closer parity to Linux `--urgency=critical`, use
`scenario="reminder"` instead — but default to the `duration` mapping.

No changes to `toast.go`, `toast_linux.go`, `toast_darwin.go`, `main.go`, or any
Makefile. New code lives in two `//go:build windows` files.

---

## 1. Background: how it works on Linux today

The `toast/` package is a tiny cross-platform abstraction:

- `toast/toast.go` — public API, no build tag, compiled everywhere:
  ```go
  func Init(logger *log.Logger) error { return initPlatform(logger) }
  func Show(title, message string, persist bool) error { return show(title, message, persist) }
  ```
- `toast/toast_linux.go` (`//go:build linux`) — the real backend. Shells out to
  the external `notify-send` binary via `os/exec`. **No CGO, no Go deps.**
  - `initPlatform`: `exec.LookPath("notify-send")`; error if missing.
  - `show`: builds `notify-send` args, runs it, parses the notification id from
    stdout (`--print-id`), stores it in `lastID`, feeds it back as
    `--replace-id` next time so successive toasts **replace** the previous one.
  - `persist` → `--urgency=critical` + `--expire-time=1000000`; else `normal` + `5000`.
  - `--transient` (don't keep in history), `--app-name=uk.co.electronstudio.dictate.notify`.
- `toast/toast_windows.go` (`//go:build windows`) — currently a 9-line no-op stub:
  ```go
  func initPlatform(logger *log.Logger) error          { return nil }
  func show(title, message string, persist bool) error { return nil }
  ```
- `toast/toast_darwin.go` (`//go:build darwin`) — same no-op stub.

**Wiring in `main.go`** (do not change):
- `main.go:20` imports `"github.com/electronstudio/low_latency_dictation/toast"`.
- `main.go:281` calls `toast.Init(logger)` at startup; failure is a non-fatal
  warning printed to stderr:
  ```go
  if err := toast.Init(logger); err != nil {
      fmt.Fprintf(os.Stderr, "warning: %v\n", err)
  }
  ```
- `main.go:666` calls `toast.Show(stateStr, text, persist)` inside `redrawText`,
  which runs on **every screen redraw** (every new whisper segment). `persist`
  is `true` except when `state == StatePaused` (then `false`).
- `stateStr` (the toast **title**) is one of `PAUSED` / `LISTENING` /
  `DICTATING` / `FINALIZING` (see `appState.String()` at `main.go:109`).
  The **body** is the transcribed text.

`go.mod` currently has **no** notification/COM dependency. Note the module
path is `github.com/electronstudio/low_latency_dictation` (NOT
`github.com/user/dictation` as AGENTS.md says — AGENTS.md is stale).

---

## 2. Why this approach (and the rejected alternatives)

Windows toasts require an **AppUserModelID (AUMID)** identifying the app, and
the toast content is described by an **XML payload** shown via the WinRT
`Windows.UI.Notifications` COM API. Options considered:

| Option | Mechanism | Verdict |
|---|---|---|
| `github.com/go-toast/toast` | Spawns PowerShell per toast | **Rejected**: ~300ms/toast startup; no replacement (stacks); bad for per-redraw firing. |
| `github.com/gen2brain/beeep` | Native WinRT on Win10/11, PS fallback | **Rejected**: no toast-replacement (stacks); adds godbus on Linux unless `nodbus` tag. |
| Shell out to PowerShell (mirror notify-send) | `os/exec` → inline WinRT script | **Rejected**: PowerShell startup latency on every redraw; fragile. |
| `git.sr.ht/~jackmordaunt/go-toast/v2` (Jack Mordaunt) | Pure-Go WinRT COM via go-ole | **Close, but rejected as a dependency**: its public `Push()` releases all COM handles immediately (`defer toast.Release()`), so **Hide-previous is unreachable** through its API. Its COM impl lives under `internal/` (Go import rule forbids reaching across module roots). |
| CGO WinToast bindings | C++ lib | **Rejected**: complicates the Windows CGO build. |
| **Self-implement with go-ole** | WinRT COM in `toast_windows.go` | **CHOSEN**. |

**Chosen: self-implement a minimal WinRT toast client with
`github.com/go-ole/go-ole`.** Rationale:

- **CGO-free**: go-ole uses `syscall`, not cgo. Keeps the `toast` package
  CGO-free like the Linux backend.
- **Cross-platform-safe**: the new code is entirely behind
  `//go:build windows`, so it is **not compiled into the Linux/macOS binaries**.
  go-ole will appear in `go.mod` but won't touch the Linux build.
- **Supports Hide-previous**: we hold the `ToastNotification` handle across
  calls and call `notifier.Hide(prev)` before the next `Show`.
- **Drops complexity we don't need**: Jack Mordaunt's go-toast also implements
  `IClassFactory` + `INotificationActivationCallback` (via
  `syscall.NewCallback` + memory pinning) — but that machinery exists only to
  support **action-button callbacks**, which this app does not use. We only
  need: `RoInitialize` → `ToastNotificationManager.GetDefault` →
  `CreateToastNotifierWithId` → `XmlDocument.LoadXml` →
  `CreateToastNotification` → `Show`/`Hide`, plus a registry `SetAppData`.
  That's ~200–250 lines.

### Attribution & license

The WinRT vtable struct/GUID definitions below are adapted from
`git.sr.ht/~jackmordaunt/go-toast/v2` (MIT / Unlicense) which itself generates
them from Windows SDK metadata. Put an attribution comment at the top of the
new COM-glue file, e.g.:

```go
// WinRT COM vtable definitions adapted from git.sr.ht/~jackmordaunt/go-toast/v2
// (MIT / Unlicense). The originals are auto-generated from Windows SDK metadata.
```

`github.com/go-ole/go-ole` is BSD-3-Clause.

---

## 3. Files to create / change

### 3.1 `go.mod` / `go.sum` — add go-ole

Add `github.com/go-ole/go-ole v1.3.0` (latest as of writing). Run:
```
go get github.com/go-ole/go-ole@v1.3.0
```
`golang.org/x/sys` is already present (v0.33.0, indirect) — we use its
`windows/registry` subpackage. After `go get`, also run `go mod tidy` so go-ole
lands in the right `require` block and `go.sum` is complete.

> The implementing agent should confirm go-ole is only compiled on Windows.
> Sanity check: `grep -rl "go-ole" $(go env GOPATH)/...` is unnecessary — just
> ensure all imports of `go-ole` are inside `//go:build windows` files in THIS
> repo. go-ole's own internal `//go:build windows` guards mean even if it were
> imported on Linux it would not pull Windows syscalls; but keep our import
> Windows-only to avoid adding dead code to the Linux binary.

### 3.2 `toast/toast_windows.go` — replace the no-op stub (public surface)

Replace the current 9-line stub with the Windows backend's public surface and
state. Package-level vars (Windows-only, so zero-cost on other platforms):

```go
//go:build windows

package toast

import (
	"fmt"
	"log"
	"sync"
)

var (
	mu         sync.Mutex
	logger     *log.Logger
	notifier   *toastNotifier   // lazily created, cached for the process lifetime
	prevToast  *toastNotification // retained so we can Hide it next call; nil when none
	initialized bool
)
```

#### `initPlatform(lg *log.Logger) error`

1. Store `logger = lg`.
2. Call `registerAppData(appID)` (see 3.3). AppID constant:
   `const appID = "uk.co.electronstudio.dictate"` (matches the
   `--app-name` used on Linux, without the `.notify` suffix which was a
   notify-send artifact).
3. Do **not** call `ole.RoInitialize` here (do it lazily on first `show`, or
   inside the COM-glue `newNotifier`). Reason: init failures should be
   non-fatal and `main.go` already tolerates an error. Return any error
   wrapped with `"toast: "`.
4. `logf("toast: init ok")` on success (mirror Linux's logging style).

The registry write is the only thing that can really fail at init time
(permission/registry issues). If `registerAppData` fails you may still let
`show` try (toasts can sometimes show with a generic AUMID); but returning the
error here matches Linux's behavior of surfacing init problems.

#### `show(title, message string, persist bool) error`

```go
func show(title, message string, persist bool) error {
	mu.Lock()
	defer mu.Unlock()
	if err := ensureInitialized(); err != nil {   // RoInitialize + create+cache notifier
		return err
	}
	// Hide & release the previous toast so we keep a single updating toast.
	if prevToast != nil {
		_ = notifier.Hide(prevToast)   // best-effort; ignore error
		prevToast.Release()
		prevToast = nil
	}
	duration := "short"
	if persist {
		duration = "long"
	}
	xml := buildToastXML(title, message, duration)   // silent audio, see §4
	t, err := createAndShow(notifier, xml)
	if err != nil {
		return fmt.Errorf("toast: %w", err)
	}
	prevToast = t   // retain — do NOT Release here; freed on next call
	logf("toast: shown (duration=%s)", duration)
	return nil
}
```

Also include the `logf` helper (copy from `toast_linux.go`):
```go
func logf(format string, args ...interface{}) {
	if logger != nil {
		logger.Printf(format, args...)
	}
}
```

`ensureInitialized` does (idempotent, under `mu`):
- if `initialized` return nil
- `ole.RoInitialize(1)` (ignore `S_FALSE`/`RPC_E_CHANGED_MODE`-style errors;
  go-ole returns these as non-nil — treat `ole.S_FALSE == 1` as success)
- create `notifier` via `newNotifier(appID)` and cache it
- set `initialized = true`

> **Why hide-and-reshow instead of WinRT `Tag`/`Group`:** the WinRT "replace"
> mechanism is `ToastNotification.Tag`/`Group` + re-`Show` with the same tag.
> Setting `Tag` requires the `put_Tag` accessor on `iToastNotification`,
> which means adding a `Tag` `HString` setter slot to the vtable and a
> `syscall.SyscallN` call. Hide-then-Show is simpler and achieves the visible
> goal (one toast on screen, no Action Center pile-up). If the maintainer
> later wants to also collapse Action Center *history* entries, add
> `Tag`/`Group` (see §7 "Future").

### 3.3 `toast/wintoast_windows.go` — new file (COM glue, `//go:build windows`)

This file holds: vtable struct/GUID definitions (adapted from go-toast, see
attribution above), thin Go wrappers, the notifier factory, the XML builder,
and the registry AUMID writer. Keep it in package `toast` (one file, no
sub-package) — simpler than go-toast's tree, which split things across
`internal/winrt/...` for code-gen reasons we don't share.

#### Imports

```go
import (
	"fmt"
	"syscall"
	"unsafe"

	"github.com/go-ole/go-ole"
	"golang.org/x/sys/windows/registry"
)
```

#### HString lifecycle (IMPORTANT — go-toast leaks this)

go-ole exposes `ole.NewHString(s) (HString, error)` and
`ole.DeleteHString(h HString) error`. **Always `defer DeleteHString`** for
every `NewHString` you create (go-toast's generated code forgets to, which is a
leak; do better). Pattern:
```go
hApp, err := ole.NewHString(appID)
if err != nil { return err }
defer ole.DeleteHString(hApp)
```

#### Vtable definitions (verbatim from go-toast, reformatted)

These are the COM interface GUIDs and vtable layouts. They are stable Windows
Runtime contracts; copy them faithfully.

**1) `ToastNotificationManager` statics** — entry point:
```go
const guidToastNotificationManagerStatics5 = "{d6f5f569-d40d-407c-8989-88cab42cfd14}"

type iToastNotificationManagerStatics5 struct { ole.IInspectable }
type iToastNotificationManagerStatics5Vtbl struct {
	ole.IInspectableVtbl
	GetDefault uintptr
}
func (v *iToastNotificationManagerStatics5) VTable() *iToastNotificationManagerStatics5Vtbl {
	return (*iToastNotificationManagerStatics5Vtbl)(unsafe.Pointer(v.RawVTable))
}
// GetDefault returns the ToastNotificationManagerForUser.
func getDefaultManager() (*toastNotificationManagerForUser, error) {
	inspectable, err := ole.RoGetActivationFactory(
		"Windows.UI.Notifications.ToastNotificationManager",
		ole.NewGUID(guidToastNotificationManagerStatics5))
	if err != nil {
		return nil, err
	}
	defer inspectable.Release() // factory is a static; release our ref
	v := (*iToastNotificationManagerStatics5)(unsafe.Pointer(inspectable))
	var out *toastNotificationManagerForUser
	hr, _, _ := syscall.SyscallN(v.VTable().GetDefault, 0, uintptr(unsafe.Pointer(&out)))
	if hr != 0 {
		return nil, ole.NewError(hr)
	}
	return out, nil
}
```

**2) `ToastNotificationManagerForUser`** — has `CreateToastNotifierWithId`:
```go
const guidToastNotificationManagerForUser = "{79ab57f6-43fe-487b-8a7f-99567200ae94}"

type toastNotificationManagerForUser struct { ole.IUnknown }
type iToastNotificationManagerForUser struct { ole.IInspectable }
type iToastNotificationManagerForUserVtbl struct {
	ole.IInspectableVtbl
	CreateToastNotifier       uintptr
	CreateToastNotifierWithId uintptr
	GetHistory                uintptr
	GetUser                   uintptr
}
func (v *iToastNotificationManagerForUser) VTable() *iToastNotificationManagerForUserVtbl {
	return (*iToastNotificationManagerForUserVtbl)(unsafe.Pointer(v.RawVTable))
}
func (m *toastNotificationManagerForUser) createToastNotifierWithId(appID string) (*toastNotifier, error) {
	itf := m.MustQueryInterface(ole.NewGUID(guidToastNotificationManagerForUser))
	defer itf.Release()
	v := (*iToastNotificationManagerForUser)(unsafe.Pointer(itf))
	hApp, err := ole.NewHString(appID)
	if err != nil { return nil, err }
	defer ole.DeleteHString(hApp)
	var out *toastNotifier
	hr, _, _ := syscall.SyscallN(
		v.VTable().CreateToastNotifierWithId,
		uintptr(unsafe.Pointer(v)),
		uintptr(hApp),
		uintptr(unsafe.Pointer(&out)),
	)
	if hr != 0 { return nil, ole.NewError(hr) }
	return out, nil
}
```

**3) `ToastNotifier`** — `Show` + `Hide`:
```go
const guidToastNotifier = "{75927b93-03f3-41ec-91d3-6e5bac1b38e7}"

type toastNotifier struct { ole.IUnknown }
type iToastNotifier struct { ole.IInspectable }
type iToastNotifierVtbl struct {
	ole.IInspectableVtbl
	Show                           uintptr
	Hide                           uintptr
	GetSetting                     uintptr
	AddToSchedule                  uintptr
	RemoveFromSchedule             uintptr
	GetScheduledToastNotifications uintptr
}
func (v *iToastNotifier) VTable() *iToastNotifierVtbl {
	return (*iToastNotifierVtbl)(unsafe.Pointer(v.RawVTable))
}
func (n *toastNotifier) Show(t *toastNotification) error {
	itf := n.MustQueryInterface(ole.NewGUID(guidToastNotifier))
	defer itf.Release()
	v := (*iToastNotifier)(unsafe.Pointer(itf))
	hr, _, _ := syscall.SyscallN(v.VTable().Show,
		uintptr(unsafe.Pointer(v)), uintptr(unsafe.Pointer(t)))
	if hr != 0 { return ole.NewError(hr) }
	return nil
}
func (n *toastNotifier) Hide(t *toastNotification) error {
	itf := n.MustQueryInterface(ole.NewGUID(guidToastNotifier))
	defer itf.Release()
	v := (*iToastNotifier)(unsafe.Pointer(itf))
	hr, _, _ := syscall.SyscallN(v.VTable().Hide,
		uintptr(unsafe.Pointer(v)), uintptr(unsafe.Pointer(t)))
	if hr != 0 { return ole.NewError(hr) }
	return nil
}
func (n *toastNotifier) Release() { n.IUnknown.Release() }
```

**4) `ToastNotification`** + factory:
```go
const guidToastNotification       = "{997e2675-059e-4e60-8b06-1760917c8b80}"
const guidToastNotificationFactory = "{04124b20-82c6-4229-b109-fd9ed4662b53}"

type toastNotification struct { ole.IUnknown }
func (t *toastNotification) Release() { t.IUnknown.Release() }

type iToastNotification struct { ole.IInspectable }
type iToastNotificationVtbl struct {
	ole.IInspectableVtbl
	GetContent        uintptr
	SetExpirationTime uintptr
	GetExpirationTime uintptr
	AddDismissed      uintptr
	RemoveDismissed   uintptr
	AddActivated      uintptr
	RemoveActivated   uintptr
	AddFailed         uintptr
	RemoveFailed      uintptr
}

type iToastNotificationFactory struct { ole.IInspectable }
type iToastNotificationFactoryVtbl struct {
	ole.IInspectableVtbl
	CreateToastNotification uintptr
}
func (v *iToastNotificationFactory) VTable() *iToastNotificationFactoryVtbl {
	return (*iToastNotificationFactoryVtbl)(unsafe.Pointer(v.RawVTable))
}

func createToastNotification(doc *xmlDocument) (*toastNotification, error) {
	inspectable, err := ole.RoGetActivationFactory(
		"Windows.UI.Notifications.ToastNotification",
		ole.NewGUID(guidToastNotificationFactory))
	if err != nil { return nil, err }
	defer inspectable.Release()
	v := (*iToastNotificationFactory)(unsafe.Pointer(inspectable))
	var out *toastNotification
	hr, _, _ := syscall.SyscallN(
		v.VTable().CreateToastNotification,
		0,                              // static method, no `this`
		uintptr(unsafe.Pointer(doc)),
		uintptr(unsafe.Pointer(&out)),
	)
	if hr != 0 { return nil, ole.NewError(hr) }
	return out, nil
}
```

> Note: `iToastNotification` inherits `IInspectable` which has 3 methods
> (`GetIids`, `GetRuntimeClassName`, `GetTrustLevel`) before the derived
> methods. go-ole's `ole.IInspectableVtbl` already includes those 3 plus
> `IUnknown`'s 3 (`QueryInterface`, `AddRef`, `Release`). The slots above
> (`GetContent`, `SetExpirationTime`, …) are appended in declaration order,
> which matches the WinRT vtable. **Do not reorder them.**

**5) `XmlDocument`** — load the XML payload:
```go
const guidXmlDocumentIO = "{6cd0e74e-ee65-4489-9ebf-ca43e87ba637}"

type xmlDocument struct { ole.IUnknown }
type iXmlDocumentIO struct { ole.IInspectable }
type iXmlDocumentIOVtbl struct {
	ole.IInspectableVtbl
	LoadXml             uintptr
	LoadXmlWithSettings uintptr
	SaveToFileAsync     uintptr
}
func (v *iXmlDocumentIO) VTable() *iXmlDocumentIOVtbl {
	return (*iXmlDocumentIOVtbl)(unsafe.Pointer(v.RawVTable))
}

func newXmlDocument() (*xmlDocument, error) {
	inspectable, err := ole.RoActivateInstance("Windows.Data.Xml.Dom.XmlDocument")
	if err != nil { return nil, err }
	return (*xmlDocument)(unsafe.Pointer(inspectable)), nil
}
func (d *xmlDocument) LoadXml(xml string) error {
	itf := d.MustQueryInterface(ole.NewGUID(guidXmlDocumentIO))
	defer itf.Release()
	v := (*iXmlDocumentIO)(unsafe.Pointer(itf))
	hXml, err := ole.NewHString(xml)
	if err != nil { return err }
	defer ole.DeleteHString(hXml)
	hr, _, _ := syscall.SyscallN(v.VTable().LoadXml,
		uintptr(unsafe.Pointer(v)), uintptr(hXml))
	if hr != 0 { return ole.NewError(hr) }
	return nil
}
func (d *xmlDocument) Release() { d.IUnknown.Release() }
```

#### Orchestrator helpers

```go
// newNotifier creates and returns the ToastNotifier for the given AUMID.
func newNotifier(appID string) (*toastNotifier, error) {
	mgr, err := getDefaultManager()
	if err != nil { return nil, fmt.Errorf("getDefaultManager: %w", err) }
	defer mgr.Release()
	return mgr.createToastNotifierWithId(appID)
}

// createAndShow builds the doc, creates the toast, shows it, and returns the
// retained toast handle (caller must Release it later).
func createAndShow(n *toastNotifier, xml string) (*toastNotification, error) {
	doc, err := newXmlDocument()
	if err != nil { return nil, fmt.Errorf("newXmlDocument: %w", err) }
	defer doc.Release()
	if err := doc.LoadXml(xml); err != nil {
		return nil, fmt.Errorf("LoadXml: %w", err)
	}
	t, err := createToastNotification(doc)
	if err != nil { return nil, fmt.Errorf("createToastNotification: %w", err) }
	if err := n.Show(t); err != nil {
		t.Release()
		return nil, fmt.Errorf("Show: %w", err)
	}
	return t, nil
}
```

#### `buildToastXML(title, body, duration string) string`

See §4 for the template and escaping rules. Use `html.EscapeString` on title
and body, then `fmt.Sprintf`. Keep `duration` as a parameter so `show` can pass
`"short"`/`"long"`.

#### `registerAppData(appID string) error`

Writes the AUMID to the registry so Action Center shows the app name:
```go
func registerAppData(appID string) error {
	key := `SOFTWARE\Classes\AppUserModelId\` + appID
	k, _, err := registry.CreateKey(registry.CURRENT_USER, key, registry.SET_VALUE|registry.CREATE_SUB_KEY)
	if err != nil { return fmt.Errorf("open registry key: %w", err) }
	defer k.Close()
	if err := k.SetStringValue("DisplayName", appID); err != nil {
		return fmt.Errorf("set DisplayName: %w", err)
	}
	// Optional: make Action Center show the embedded exe icon. Path to the
	// running .exe; Windows extracts the icon resource. Commented out by
	// default — enable if the maintainer wants the icon and accepts the
	// dependency on the exe path being stable.
	// exe, _ := os.Executable()
	// if exe != "" { _ = k.SetStringValue("IconUri", exe) }
	return nil
}
```

> **Do NOT** set `CustomActivator` (a CLSID under
> `SOFTWARE\Classes\CLSID\{...}\LocalServer32`). That entry is for the
> `INotificationActivationCallback` COM server (action-button callbacks),
> which we are not implementing. Setting a bogus CLSID can cause Windows to
> try to launch a non-existent COM server when a toast is activated, producing
> errors. Display-only toasts need only `DisplayName` (and optionally
> `IconUri`).

---

## 4. The XML payload (silent toast)

Use the generic toast template (Win10/11 supports `ToastGeneric`; the legacy
`ToastText01` etc. templates also work but `ToastGeneric` is recommended and
allows the silent-audio element):

```xml
<toast duration="{DURATION}">
  <visual>
    <binding template="ToastGeneric">
      <text>{TITLE}</text>
      <text>{BODY}</text>
    </binding>
  </visual>
  <audio silent="true"/>
</toast>
```

Rules:
- `duration` ∈ `{"short", "long"}`. `short` ≈ 7s default; `long` ≈ 25s.
- **Always escape** title and body with `html.EscapeString` (or hand-rolled
  XML-escape of `& < >`). The transcribed text can contain `<`, `&`, etc.
- If `body == ""`, **omit** the second `<text>` element (matches Linux's
  "omit message arg if empty"). An empty `<text></text>` is fine too but
  omitting is cleaner.
- `<audio silent="true"/>` makes it not chime (decided). If the maintainer
  later wants sound, set `<audio src="ms-winsoundevent:Notification.Default"
  loop="false"/>` or use go-toast's `Default` constant.
- Do **not** add `<actions>`, `<inputs>`, or `launch=`/`activationType=`
  attributes — we have no callback handler.

A minimal Go implementation:
```go
func buildToastXML(title, body, duration string) string {
	t := html.EscapeString(title)
	if body == "" {
		return fmt.Sprintf(`<toast duration="%s"><visual><binding template="ToastGeneric"><text>%s</text></binding></visual><audio silent="true"/></toast>`, duration, t)
	}
	b := html.EscapeString(body)
	return fmt.Sprintf(`<toast duration="%s"><visual><binding template="ToastGeneric"><text>%s</text><text>%s</text></binding></visual><audio silent="true"/></toast>`, duration, t, b)
}
```
(Imports: `fmt`, `html`. `html.EscapeString` escapes `&`, `<`, `>`, `'`, `"`
only — sufficient and standard.)

---

## 5. Concurrency & lifecycle notes

- `show` holds `mu` for its whole body (matches `toast_linux.go`). This
  serializes COM calls, which is fine — toasts fire at most a few times/sec.
- `prevToast` is only touched under `mu`. Safe.
- The `notifier` is created once and cached for the process lifetime. Do not
  Release it until shutdown (and there's no shutdown hook — fine, the OS
  reclaims on exit). Re-creating a notifier per call is wasteful and can
  briefly fail on WinRT init races.
- **HString leaks**: every `ole.NewHString` must be paired with
  `ole.DeleteHString`. The orchestrator helpers above do this via `defer`.
- **Go GC vs COM**: go-ole's `IUnknown.Release` decrements the COM refcount.
  The Go pointer (`*toastNotification`) is just a view; the underlying COM
  allocation lives until its refcount hits 0. By retaining `prevToast` we keep
  the COM object alive across calls (correct — we need it for `Hide`). Call
  `Release` after `Hide` so it's freed before we create the next one.

---

## 6. Build-safety verification (no Windows machine needed)

The implementing agent should confirm the cross-compile is sound and Linux is
unaffected. Run from the repo root:

```bash
# 1. go.mod/go.sum tidy and consistent
go mod tidy
git diff go.mod go.sum   # expect: + github.com/go-ole/go-ole, maybe x/sys bump

# 2. Typecheck the Windows build WITHOUT building (no CGO/MinGW needed)
GOOS=windows GOARCH=amd64 go vet ./toast/
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build ./toast/

# 3. Linux build still works and toast package unchanged behavior
go build ./...
GOOS=linux go vet ./toast/

# 4. Confirm go-ole is not pulled into the Linux binary
go list -deps ./toast/ 2>/dev/null | grep go-ole || echo "go-ole NOT in Linux deps (correct)"
```

Expected: step 4 prints the "NOT in Linux deps" message because all our
`go-ole` imports live under `//go:build windows`.

If `go build ./...` on Linux complains about go-ole, it means a go-ole import
leaked into a non-Windows-tagged file — fix the build tags, don't add
`//go:build windows` to `toast.go` (that would break the public API on other
platforms).

### Full Windows build (slow — do NOT run automatically)
Per AGENTS.md, full `make` (which triggers `Makefile.windows` → CMake/whisper
build) is slow. The agent should NOT run `make` themselves; that's for the
maintainer on a Windows/MSYS2 box. The lightweight checks above are enough to
validate the Go code.

---

## 7. Future enhancements (out of scope here; documented for the maintainer)

- **WinRT `Tag`/`Group` for true replace-in-Action-Center-history.** Add
  `put_Tag`/`put_Group` slots to `iToastNotificationVtbl` (they sit between
  `GetContent` and `SetExpirationTime` in the real vtable — verify slot order
  against the Windows SDK `ToastNotification` interface). Set a fixed tag
  (e.g. `"live"`) and a fixed group (e.g. `"dictate"`) on every toast; Windows
  will supersede the prior toast with the same Tag/Group. Then Hide-previous
  becomes unnecessary.
- **Start Menu `.lnk` with AUMID + icon** for proper Action Center branding.
  Extend `Makefile.windows` `install`/`uninstall` (currently no-ops at
  `Makefile.windows:73`) to create/remove a `.lnk` in
  `$STARTMENU/Programs/dictate.lnk` whose `AppUserModelID` property is set to
  the AUMID. This is what makes the `IconUri`/icon actually render.
- **IconUri → running exe**: uncomment the `IconUri` block in
  `registerAppData` so Action Center pulls the icon embedded via
  `assets/dictate.rc` (the `.syso` built by `Makefile.windows`).
- **Callback support** (action buttons) would require porting go-toast's
  `impl.go` (IClassFactory + INotificationActivationCallback +
  `syscall.NewCallback` + `runtime.Pinner`). Not needed for this app.

---

## 8. Open questions to confirm with the maintainer before/while implementing

1. **`persist` mapping**: plan uses `duration` `long`/`short`. Confirm or
   switch to `scenario="reminder"` (closer to Linux `--urgency=critical`,
   more intrusive — stays until dismissed). Default if no answer: `duration`.
2. **IconUri**: leave off (generic icon in Action Center) or set to the
   running exe path? Default if no answer: leave off (matches "registry AUMID
   only" decision).
3. **AppID string**: `uk.co.electronstudio.dictate` (no `.notify`). Confirm.
   Linux used `uk.co.electronstudio.dictate.notify` but the `.notify` suffix
   was a notify-send app-name convention, not a real AUMID constraint.

---

## 9. Quick reference: where things live after implementation

```
toast/
├── toast.go              (UNCHANGED — public API)
├── toast_linux.go        (UNCHANGED — notify-send backend)
├── toast_darwin.go       (UNCHANGED — no-op stub)
├── toast_windows.go      (REPLACED — public surface + state + show/init)
└── wintoast_windows.go   (NEW — WinRT COM glue: vtables, wrappers, XML, registry)
go.mod                    (MODIFIED — + github.com/go-ole/go-ole)
go.sum                    (MODIFIED — go-ole + transitive hashes)
```

No other files change. `main.go` keeps calling `toast.Init` and `toast.Show`
exactly as it does today.
