# dictate

A small Go program that does real-time speech-to-text from your microphone. It uses [whisper.cpp](https://github.com/ggerganov/whisper.cpp) under the hood for the actual inference, so all transcription happens locally.

## building

You need Go, a C compiler, SDL2, and the whisper.cpp headers. Then just run:

```bash
make
```

This will build the vendored `whisper.cpp` libraries and produce a `dictate` binary.

## usage

Make sure your microphone is active, then run:

```bash
./dictate
```

It'll start listening immediately. Press any key to stop; it will print the final transcription and copy it to your clipboard.

The first time you run it, the required Whisper model will be downloaded automatically.

## license

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE).

whisper.cpp is MIT licensed and copyright (c) Georgi Gerganov.
