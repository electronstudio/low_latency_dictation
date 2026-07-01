# Code Improvement Plan for low_latency_dictation

This file captures the improvement, refactoring, and simplification ideas
identified during a full read-through of the non-vendor source tree.

## 1. `main.go`: reduce globals and split the state machine

- Introduce an `App` struct that owns CLI params, audio, whisper contexts,
  tcell screen, tray, logger, and state-machine fields.
- Convert free functions (`initAudio`, `loadContexts`, `registerGlobalHotkey`,
  `initScreen`, `startScreenPoller`, `runMainLoop`, `produceFinalText`,
  `emitFinal`, `startSDLPoller`) into `App` methods or small collaborators.
- Return errors from helpers and let `run()` handle `die()` centrally instead
  of calling `os.Exit(1)` from deep in the call stack.
- Split `runMainLoop` into smaller methods: `handlePausedEvent`,
  `handleActiveEvent`, `transcribeTick`, `finalize`, `redrawText`.
- Move the state-machine closures (`segmentsText`, `redrawText`, `finalize`,
  `handleTray`) out of `runMainLoop`.
- Tie the global-hotkey forwarder goroutine to the `App` lifecycle so it is
  stopped on shutdown instead of leaking.
- Remove the large TODO comment block at the top of the file; migrate
  actionable items to issues or a dedicated TODO file.

## 2. Fix TUI rendering for non-ASCII text

- `printToScreen` uses byte index `i` as column offset; use
  `runewidth.RuneWidth(r)` to advance a column counter.
- `printWrapped` uses byte lengths as column widths; switch to
  `runewidth.StringWidth` or iterate runes.
- `printStatus` uses `len(s)` (bytes); use `utf8.RuneCountInString` or
  `runewidth.StringWidth`.

## 3. Harden `model.go` downloads

- Simplify the manual read/write loop using `io.Copy`/`io.TeeReader` with a
  small `progressWriter`.

## 4. Reduce audio callback allocations

- Avoid allocating a new `[]float32` on every SDL callback; use a `sync.Pool`
  or write directly into a ring buffer.
- Replace the `(*[1 << 30]float32)(...)` idiom with `unsafe.Slice`.
- Consider separating the circular buffer from the full-session buffer into
  two small types (`RingBuffer`, `SessionBuffer`).
- Document that `GetFullAudio` copies the entire session audio.

## 5. Clarify VAD semantics

- Rename `SimpleVAD` to `IsSilent` or `DetectSilence` because it returns `true`
  for silence.
- Document that `HighPassFilter` mutates the input slice in place.
- Add a small `absF32` helper instead of `if v < 0 { v = -v }`.

## 6. Modernize `transcribe` package

- Remove redundant legacy `// +build` lines where `//go:build` exists.
- Replace unsafe pointer conversions with `unsafe.Slice` where possible.
- Add a `Close()` alias for `Free()`.
- Consider accepting `context.Context` in `Context.Full` for cancellation.

## 7. Clean up `hotkey` package

- In `ParseCombo`, deduplicate modifiers by building a bit mask first, then
  converting to a slice.
- Consider replacing the empty `Event` struct with `chan struct{}` or include
  timestamp/state.
- Optionally merge `hotkey_darwin.go` and `hotkey_windows.go` into the common
  gohk backend file.

## 8. Harden `tray` package

- Avoid blocking forever on `<-t.ready` in `SetState`/`SetDictationText`; add
  a timeout or select.
- Use a single fan-out goroutine for menu clicks instead of one per item.
- Use `runewidth.StringWidth` for `wrapWords` if CJK support matters.


## 10. Harden `typing` package

- Replace magic sleep values in `typing_linux.go` with named constants.
- Document hardcoded macOS keycodes 55 and 9.
- Add a comment explaining the `[8]byte pad` in the Windows `tagINPUT` struct.

## 11. Makefiles / build

- Extract shared `GGML_VULKAN`/`GO_TAGS` logic into a common include file.
- Replace duplicated copy commands in `whisper_libs` with a `foreach` loop.
- Fix `Makefile.windows` duplicate `$(BINARY)` in the `clean` target.
- Add a `check` target that runs `go vet ./...`, `gofmt -l`, and tests.

## 12. Testing and CI

- Add `go test ./...` and `go vet ./...` steps to CI.
- Add tests for `model.go` with `httptest.Server`.
- Add tests for the audio ring buffer independent of SDL.
- Add tests for segment-merging logic once extracted from `runMainLoop`.
- Extend `hotkey_test.go` to cover whitespace and alias edge cases.

## 13. Documentation

- Rewrite `AGENTS.md` to match the actual module name and package layout.
- Add audio-device usage example to the README.

## 14. Small code-quality items

- `openLogFile` should return a logger instead of mutating a package-level
  variable.
- Make `Threads == 0` default behavior explicit in the CLI struct/docs.
- Use a named `logLevelSilent` constant instead of the magic `100` in
  `transcribe.go`.
- Move the `idRe` regex compilation out of package-level init if possible, or
  document why it is safe.
