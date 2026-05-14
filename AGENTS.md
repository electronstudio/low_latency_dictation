# Agent Notes

## Project
Single Go executable: real-time microphone speech-to-text using whisper.cpp (CGO) + SDL2 (CGO).

## Build
Compile with `make`. Output binary: `dictate`.

```bash
make
make clean
```

**Windows (MSYS2/MinGW):** The Makefile handles:
- CMake library naming differences (`whisper.lib` vs `libwhisper.a`)
- SDL2 include/library paths via `cygpath -m` (go toolchain invokes native `gcc.exe` which needs Windows-style paths)
- Vulkan library name (`vulkan-1` instead of `vulkan`)

The `pkg-config` directive is only used on Linux/macOS; on Windows paths and flags are computed in the Makefile.

## Dependencies
- **Go 1.24+**
- **System-wide C headers:** `whisper.h`, `ggml*.h`, SDL2 headers
- **System libs:** `libSDL2`, `libvulkan`
- **Linux:** OpenMP (`-fopenmp`)
- **macOS:** Accelerate / Metal frameworks (via `libggml-blas.a`, `libggml-metal.a`)
- **Vendored static archives:** `libs/libwhisper.a`, `libs/libggml*.a`

## Architecture
Packages under the module `github.com/user/dictation`:
- `audio/` — SDL2-based async audio capture (circular buffer, mutex-protected)
- `vad/` — simple energy-based voice-activity detector + high-pass filter
- `transcribe/` — thin Go wrappers around the whisper.cpp C API
- `timestamp/` — formatting helpers for SRT/VTT-style timestamps
- `main.go` — event loop, SDL & signal polling, transcription output

## Runtime
Requires a live microphone/audio input device.

## Models
Model binaries are cached in `~/.cache/low_latency_dictation/` and are not in version control.
Default: `ggml-tiny.en-q8_0.bin` (hardcoded in `main.go`).
If a model is missing it is automatically downloaded from `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/`.
