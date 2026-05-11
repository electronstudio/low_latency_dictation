# Agent Notes

## Project
Single C++ executable: real-time microphone speech-to-text using whisper.cpp + SDL2 + Vulkan.

## Build
Compile with `make`. Output binary: `dictation`.

```bash
make
make clean
```

Note: The link order matters — `-lwhisper` must come **before** the `-lggml*` libs. The Makefile preserves this order.

## Dependencies
- **System-wide headers:** `whisper.h` (not vendored)
- **System libs:** `libSDL2`, `libvulkan`, OpenMP (`-fopenmp`)
- **Vendored static archives:** `libs/libwhisper.a`, `libs/libggml*.a`

## Runtime
Requires a live microphone/audio input device.

## Models
Model binaries live in `models/` and are not in version control.
Default: `models/ggml-tiny.en-q8_0.bin` (hardcoded in `main.cpp`).
