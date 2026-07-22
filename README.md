# Putin (the dictator)

A small TUI and systray Go program that does real-time speech-to-text from your microphone. It uses [whisper.cpp](https://github.com/ggerganov/whisper.cpp) under the hood for the actual inference, so all transcription happens locally.

In terminal:
![terminal demo video](assets/demo1.gif)

In systray:
![systray demo video](assets/demo2.gif)

## FAQ

### Why make this when so many similar programs, e.g. Handy, already exist?

At the time I began development, in 2025, there were not so many options.  Handy existed but hotkeys didn't work and it was not very low latency.

### But now, why not use one of those alternatives?

Two reasons:
* I like mine better.
* Fuck off.

### Isn't this just AI slop?

It's mostly written by an AI following directions from and having its code reviewed by a senior human engineer.
If that's "AI slop" then the majority of software ever written is "junior human engineer slop" that was
developed in the same way only with junior engineers instead of AI.

[![AI-DECLARATION: pair](https://img.shields.io/badge/䷼%20AI--DECLARATION-pair-ffedd5?labelColor=ffedd5)](https://ai-declaration.md)

## Usage

Download from the [releases](releases).

Make sure your microphone is connected, then run:

```bash
./dictate
```

The first time you run it, the required Whisper model will be downloaded automatically.  Available models are these:
https://huggingface.co/ggerganov/whisper.cpp/tree/main

A numbered list of audio devices will be printed on startup.  (Exit the program if
it went too by fast to see.)  If the wrong device
is used you can change it with `--audio-device` option.

To begin dictating, tap the _global hotkey_, by default **ctrl+space**.  (If you prefer to begin immediately, use `--skip-pause-mode`)

A lower quality model will be used to display your dictation in real time.  When you have finished, tap the hotkey again to finalize.
Your dictation will then be processed by a higher quality model, copied to the clipboard, and pasted into the current window.

## Quality

If you do not have a GPU, use `--quality-preset low` option and it will use a small, fast model.
If you have any GPU, even an
old one, you should be able to use ``--quality-preset medium`` and still get reasonable performance. (Latency may be a bit worse.)
Newer GPUs should be able to use `--quality-preset high` and still have reasonable latency.

**See ``--help`` for complete control over model selection and other options.**  Options can also be set
in file `~/.config/low_latency_dictation/config.yaml`.


## Global Hotkey

The hotkey is configurable with `--hotkey-mods` and `--hotkey-key`, e.g.:

```bash
./dictate --hotkey-mods alt --hotkey-key f1
```

Accepted modifiers (joined by `+`): `ctrl`, `shift`, `alt`, and `cmd` (macOS) / `win` / `super` (the Windows/Command/Super key). Keys: `a`–`z`, `0`–`9`, `f1`–`f12`, `space`, `return`, `escape`, `delete`, `tab`, and the arrow keys.

`--hotkey-key ""` disables the hotkey.

## OS specific notes

### Linux

If you install (e.g. the flatpak or Arch package) it will install a desktop shortcut you can launch from your start menu.

Needs uinput group permissions.  Enter this command then reboot:

    sudo usermod -aG uinput $USER

On Gnome, we recommend you install a [system tray](https://extensions.gnome.org/extension/615/appindicator-support/).
You can double-click the tray icon to start or to paste the transcription into the active window.
Right click for menu.


### Windows

You can drag the systray icon out of the systray menu to pin it to the taskbar.  Single click the tray icon to start dictation or to copy the transcription to the clipboard. (Paste is not automatic.) Right click for menu.

### macOS (experimental and untested)

On macOS the hotkey requires the app to be trusted for **Accessibility** (Input Monitoring). Grant it the first time in *System Settings → Privacy & Security → Accessibility*. 


## Building

### Arch (CachyOS, Artix, etc) Linux

```bash
git clone https://github.com/electronstudio/low_latency_dictation.git
cd low_latency_dictation
makepkg -si
```

### Others

You need Go, a C compiler, SDL2, and the Vulkan headers.

This will build the vendored `whisper.cpp` libraries and produce a `dictate` binary.

```bash
sudo apt install -y build-essential cmake golang-go git libsdl2-dev libshaderc-dev glslc libvulkan-dev 
git clone https://github.com/electronstudio/low_latency_dictation.git
cd low_latency_dictation
git submodule update --init --recursive
make
sudo make install 
```


## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE).

whisper.cpp is MIT licensed and copyright (c) Georgi Gerganov.
