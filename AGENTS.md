# Agent Notes

## Project
Single Go executable: real-time microphone speech-to-text using whisper.cpp (CGO) + SDL2 (CGO).

## Build
Compile with `make`, which detects the host platform and delegates to one of the platform-specific Makefiles:

- `Makefile.linux`
- `Makefile.macos`
- `Makefile.windows`

### Platform-specific notes

**Linux:** OpenMP is enabled via `-fopenmp`. Vulkan support is controlled by `GGML_VULKAN` (default `ON`).

**macOS:** Metal and Accelerate frameworks are enabled via `-DGGML_METAL=ON -DGGML_BLAS=OFF`.

**Windows (MSYS2/MinGW):** `Makefile.windows` handles:
- CMake library naming differences (`whisper.lib` vs `libwhisper.a`)
- SDL2 include/library paths via `cygpath -m` (Go toolchain invokes native `gcc.exe`, which needs Windows-style paths)
- Vulkan library name (`vulkan-1` instead of `vulkan`)

## Dependencies
- **Go 1.24+**
- **System-wide C headers:** `whisper.h`, `ggml*.h`, SDL2 headers
- **System libs:** `libSDL2`, `libvulkan`
- **Linux:** OpenMP (`-fopenmp`)
- **macOS:** Accelerate / Metal frameworks (via `libggml-metal.a`)
- **Vendored static archives:** `libs/libwhisper.a`, `libs/libggml*.a`

## Architecture
Module: `github.com/electronstudio/low_latency_dictation`

Top-level package `main`:
- `main.go` — TUI, event loop, signal/hotkey handling, transcription output
- `model.go` — model download/caching with SHA-1 verification
- `main_icons_unix.go` / `main_icons_windows.go` — embedded tray icons

Packages under `main`:
- `audio/` — SDL2-based async audio capture (circular buffer, mutex-protected)
- `vad/` — simple energy-based voice-activity detector + high-pass filter
- `transcribe/` — thin Go wrappers around the whisper.cpp C API
- `hotkey/` — cross-platform global hotkey (evdev on Linux, golang.design/x/hotkey on macOS/Windows)
- `toast/` — transient system notifications (Linux notify-send; stubs on macOS/Windows)
- `typing/` — simulated paste keystroke (uinput on Linux, CoreGraphics on macOS, SendInput on Windows)
- `tray/` — optional system tray icon via fyne.io/systray

## Runtime
Requires a live microphone/audio input device.

## Models
Model binaries are cached in `~/.cache/low_latency_dictation/` and are not in version control.
Defaults: `--model ggml-tiny.en-q8_0.bin --final-model ggml-base.en.bin`.
If a model is missing it is automatically downloaded from `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/`.
Known models are verified against a built-in SHA-1 digest list; unknown models download without verification.
See `--help` for all CLI options.

## Testing

Pure-Go packages (`hotkey`, `vad`, `model`, `tray`, `toast`, `typing`) can be unit-tested without building the vendored whisper.cpp libraries. CGO packages (`audio`, `transcribe`) require `make whisper_libs` first.

Run all tests with:

    go test ./...

## Runtime permissions

- **Linux hotkey:** the user must be in the `input` group, or the process needs read access to `/dev/input/event*`.
- **Linux paste:** the user must be in the `uinput` group, or the process needs write access to `/dev/uinput`.
- **macOS:** Accessibility permission is required for global hotkeys and synthesized paste keystrokes.
- **Windows:** no special permissions are needed beyond normal user integrity.

## Cross-platform main thread

`main.go` uses `golang.design/x/mainthread` because macOS global hotkeys require the `CGEventTap` callback to run on the main thread. `main()` calls `mainthread.Init(run)`; the tray uses `mainthread.Call` on macOS/Linux for startup and teardown.

## Assets

- Linux and macOS tray icons are embedded PNGs from `assets/dictate*.png`.
- Windows uses `.ico` files. The multi-resolution `assets/dictate.ico` can be regenerated from `assets/dictate.png` with ImageMagick when available (`Makefile.windows` includes the rule).

## Packaging

- **Linux:** `make install` installs the binary, desktop file (`dictate.desktop.in`), and icons.
- **Flatpak:** see `flatpak/uk.co.electronstudio.Dictate.yaml`.
- **macOS / Windows:** no automated installer; the CI workflow produces platform binaries as artifacts.

## AI usage

This project follows the guidelines in `AI-DECLARATION.md`.
