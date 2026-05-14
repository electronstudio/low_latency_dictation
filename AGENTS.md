# Agent Notes

## Project
Single Go executable: real-time microphone speech-to-text using whisper.cpp (CGO) + SDL2 (CGO).

## Build
Compile with `make` using the platform-specific Makefile:

```bash
make -f Makefile.linux     # Linux
make -f Makefile.macos     # macOS
make -f Makefile.windows   # Windows (MSYS2/MinGW)
```

Output binary: `dictate`.

```bash
make -f Makefile.linux
```

### Platform-specific notes

**Linux:** OpenMP is enabled via `-fopenmp`. No `pkg-config` directives required.

**macOS:** Metal and Accelerate frameworks are enabled via `-DGGML_METAL=ON -DGGML_BLAS=ON`.

**Windows (MSYS2/MinGW):** `Makefile.windows` handles:
- CMake library naming differences (`whisper.lib` vs `libwhisper.a`)
- SDL2 include/library paths via `cygpath -m` (Go toolchain invokes native `gcc.exe`, which needs Windows-style paths)
- Vulkan library name (`vulkan-1` instead of `vulkan`)

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
Defaults: `--model ggml-tiny.en-q8_0.bin --final-model ggml-base.en.bin`.
If a model is missing it is automatically downloaded from `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/`.
See `--help` for all CLI options.
