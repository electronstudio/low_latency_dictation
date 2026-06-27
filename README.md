# dictate

A small Go program that does real-time speech-to-text from your microphone. It uses [whisper.cpp](https://github.com/ggerganov/whisper.cpp) under the hood for the actual inference, so all transcription happens locally.

## building

```bash
sudo apt install -y libsdl2-dev libshaderc-dev glslc libvulkan-dev 
git submodule update --init --recursive
make -f Makefile.linux
```

You need Go, a C compiler, SDL2, and the Vulkan headers. 


This will build the vendored `whisper.cpp` libraries and produce a `dictate` binary.

## requirements

On Linux, needs uinput group permissions:

    sudo usermod -aG uinput $USER.

## usage

Make sure your microphone is active, then run:

```bash
./dictate
```

It'll start listening immediately. Stop it with the global hotkey (by default **Ctrl+Shift+D**, works while another app has focus) or by pressing any key in its terminal; it will print the final transcription and **paste it into whatever app has focus** (via the system clipboard + a simulated Ctrl/Cmd+V). The text is also copied to the clipboard, offered to WSL/Cygwin via `/dev/clipboard`, and sent as an OSC 52 terminal escape.

The first time you run it, the required Whisper model will be downloaded automatically.

## global hotkey

The stop hotkey is configurable with `--hotkey-mods` and `--hotkey-key`, e.g.:

```bash
./dictate --hotkey-mods alt --hotkey-key f1
```

Accepted modifiers (joined by `+`): `ctrl`, `shift`, `alt`, and `cmd` (macOS) / `win` / `super` (the Windows/Command/Super key). Keys: `a`–`z`, `0`–`9`, `f1`–`f12`, `space`, `return`, `escape`, `delete`, `tab`, and the arrow keys.

### Linux

On Linux the hotkey is read directly from the kernel input subsystem via **evdev**, so it works under both X11 and Wayland. Reading input devices requires your user to be in the **`input`** group. If it is not, the app prints a warning and continues; you can still stop it with any key in its terminal. To enable the global hotkey, run once and log out/in (or reboot):

```bash
sudo usermod -aG input $USER
```

### macOS

On macOS the hotkey is delivered through a CGEventTap, which requires the app to be trusted for **Accessibility** (Input Monitoring). Grant it the first time in *System Settings → Privacy & Security → Accessibility*. If permission is missing, the app prints a warning and continues; stop it with any key in its terminal instead.

### Windows

On Windows the hotkey uses `RegisterHotKey` and needs no special permission.

## license

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE).

whisper.cpp is MIT licensed and copyright (c) Georgi Gerganov.
