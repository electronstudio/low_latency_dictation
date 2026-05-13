// Package audio provides SDL2-based asynchronous audio capture.
package audio

/*
#cgo linux pkg-config: sdl2
#cgo linux LDFLAGS: -lSDL2
#cgo darwin pkg-config: sdl2
#cgo darwin LDFLAGS: -lSDL2
#cgo windows LDFLAGS: -lmingw32 -mwindows -lSDL2main -lSDL2

#include <SDL.h>
#include <SDL_audio.h>
#include <stdlib.h>
#include <stdint.h>

extern void goAudioCallback(uintptr_t handle, Uint8* stream, int len);
extern void sdlCallback(void* userdata, Uint8* stream, int len);

static inline void setResampleHint(void) {
    SDL_SetHintWithPriority("SDL_AUDIO_RESAMPLING_MODE", "medium", SDL_HINT_OVERRIDE);
}

static inline void zeroAudioSpec(SDL_AudioSpec* spec) {
    memset(spec, 0, sizeof(SDL_AudioSpec));
}

static inline SDL_bool eventIsQuit(SDL_Event* ev) {
    return ev->type == SDL_QUIT;
}

static inline void* uintptrToVoid(uintptr_t v) {
    return (void*)v;
}
*/
import "C"
import (
	"fmt"
	"os"
	"runtime/cgo"
	"sync"
	"unsafe"
)

//export goAudioCallback
func goAudioCallback(handle C.uintptr_t, stream *C.Uint8, len C.int) {
	h := *(*cgo.Handle)(unsafe.Pointer(&handle))
	audioptr := h.Value().(*AudioAsync)

	nSamples := int(len) / int(unsafe.Sizeof(float32(0)))
	if nSamples <= 0 {
		return
	}

	samples := make([]float32, nSamples)
	src := (*[1 << 30]float32)(unsafe.Pointer(stream))[:nSamples:nSamples]
	copy(samples, src)
	audioptr.callback(samples)
}

// AudioAsync captures audio asynchronously via SDL into a circular buffer.
type AudioAsync struct {
	lenMs      int
	sampleRate int
	devID      C.SDL_AudioDeviceID
	handle     cgo.Handle

	running bool
	mu      sync.Mutex

	audio     []float32
	audioPos  int
	audioLen  int
	fullAudio []float32
}

// NewAudioAsync creates a new async audio capture instance that keeps lenMs
// milliseconds of audio in its circular buffer.
func NewAudioAsync(lenMs int) *AudioAsync {
	return &AudioAsync{lenMs: lenMs}
}

// Init initialises SDL audio capture with the requested sample rate.
// captureID < 0 means default device.
func (a *AudioAsync) Init(captureID int, sampleRate int) error {
	C.SDL_LogSetPriority(C.SDL_LOG_CATEGORY_APPLICATION, C.SDL_LOG_PRIORITY_INFO)

	if C.SDL_Init(C.SDL_INIT_AUDIO) < 0 {
		return fmt.Errorf("couldn't initialize SDL: %s", C.GoString(C.SDL_GetError()))
	}

	C.setResampleHint()

	nDevices := int(C.SDL_GetNumAudioDevices(C.SDL_TRUE))
	fmt.Fprintf(os.Stderr, "audio.Init: found %d capture devices:\n", nDevices)
	for i := 0; i < nDevices; i++ {
		name := C.GoString(C.SDL_GetAudioDeviceName(C.int(i), C.SDL_TRUE))
		fmt.Fprintf(os.Stderr, "audio.Init:    - Capture device #%d: '%s'\n", i, name)
	}

	var req, obt C.SDL_AudioSpec
	C.zeroAudioSpec(&req)
	C.zeroAudioSpec(&obt)

	req.freq = C.int(sampleRate)
	req.format = C.AUDIO_F32
	req.channels = 1
	req.samples = 1024
	req.callback = (C.SDL_AudioCallback)(C.sdlCallback)

	a.handle = cgo.NewHandle(a)
	req.userdata = C.uintptrToVoid(C.uintptr_t(a.handle))

	if captureID >= 0 {
		name := C.SDL_GetAudioDeviceName(C.int(captureID), C.SDL_TRUE)
		fmt.Fprintf(os.Stderr, "audio.Init: attempt to open capture device %d : '%s' ...\n", captureID, C.GoString(name))
		a.devID = C.SDL_OpenAudioDevice(name, C.SDL_TRUE, &req, &obt, 0)
	} else {
		fmt.Fprintf(os.Stderr, "audio.Init: attempt to open default capture device ...\n")
		a.devID = C.SDL_OpenAudioDevice(nil, C.SDL_TRUE, &req, &obt, 0)
	}

	if a.devID == 0 {
		return fmt.Errorf("couldn't open an audio device for capture: %s", C.GoString(C.SDL_GetError()))
	}

	fmt.Fprintf(os.Stderr, "audio.Init: obtained spec for input device (SDL Id = %d):\n", a.devID)
	fmt.Fprintf(os.Stderr, "audio.Init:     - sample rate:       %d\n", obt.freq)
	fmt.Fprintf(os.Stderr, "audio.Init:     - format:            %d (required: %d)\n", obt.format, req.format)
	fmt.Fprintf(os.Stderr, "audio.Init:     - channels:          %d (required: %d)\n", obt.channels, req.channels)
	fmt.Fprintf(os.Stderr, "audio.Init:     - samples per frame: %d\n", obt.samples)

	a.sampleRate = int(obt.freq)
	bufSize := (a.sampleRate * a.lenMs) / 1000
	a.audio = make([]float32, bufSize)
	a.audioPos = 0
	a.audioLen = 0
	a.fullAudio = make([]float32, 0, bufSize)
	return nil
}

// Resume starts capturing audio.
func (a *AudioAsync) Resume() error {
	if a.devID == 0 {
		return fmt.Errorf("no audio device to resume")
	}
	if a.running {
		return fmt.Errorf("already running")
	}
	C.SDL_PauseAudioDevice(a.devID, 0)
	a.running = true
	return nil
}

// Pause stops capturing audio.
func (a *AudioAsync) Pause() error {
	if a.devID == 0 {
		return fmt.Errorf("no audio device to pause")
	}
	if !a.running {
		return fmt.Errorf("already paused")
	}
	C.SDL_PauseAudioDevice(a.devID, 1)
	a.running = false
	return nil
}

// Close releases the SDL audio device and cleans up the cgo.Handle.
func (a *AudioAsync) Close() {
	if a.devID != 0 {
		C.SDL_CloseAudioDevice(a.devID)
		a.devID = 0
	}
	if a.handle != 0 {
		a.handle.Delete()
		a.handle = 0
	}
}

// callback is called by the SDL audio thread. It copies samples into the
// circular buffer.
func (a *AudioAsync) callback(samples []float32) {
	if !a.running {
		return
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	a.fullAudio = append(a.fullAudio, samples...)

	nSamples := len(samples)
	if nSamples > len(a.audio) {
		nSamples = len(a.audio)
		samples = samples[len(samples)-nSamples:]
	}

	bufSize := len(a.audio)
	for i := 0; i < nSamples; i++ {
		a.audio[a.audioPos] = samples[i]
		a.audioPos = (a.audioPos + 1) % bufSize
	}
	if a.audioLen < bufSize {
		a.audioLen += nSamples
		if a.audioLen > bufSize {
			a.audioLen = bufSize
		}
	}
}

// Get copies the last ms milliseconds of captured audio into result.
// If ms <= 0 it returns up to lenMs milliseconds.
func (a *AudioAsync) Get(ms int) []float32 {
	if a.devID == 0 {
		fmt.Fprintf(os.Stderr, "audio.Get: no audio device to get audio from!\n")
		return nil
	}
	if !a.running {
		fmt.Fprintf(os.Stderr, "audio.Get: not running!\n")
		return nil
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	if ms <= 0 {
		ms = a.lenMs
	}
	nSamples := (a.sampleRate * ms) / 1000
	if nSamples > a.audioLen {
		nSamples = a.audioLen
	}
	if nSamples == 0 {
		return nil
	}

	out := make([]float32, nSamples)
	bufSize := len(a.audio)
	s0 := a.audioPos - nSamples
	if s0 < 0 {
		s0 += bufSize
	}

	for i := 0; i < nSamples; i++ {
		out[i] = a.audio[(s0+i)%bufSize]
	}
	return out
}

// SampleRate returns the actual sample rate obtained from SDL.
func (a *AudioAsync) SampleRate() int { return a.sampleRate }

// GetFullAudio returns the complete, un-truncated audio captured so far.
func (a *AudioAsync) GetFullAudio() []float32 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.fullAudio
}

// PollEvents returns false if the user requested quit (window close, etc.).
func PollEvents() bool {
	var event C.SDL_Event
	for C.SDL_PollEvent(&event) != 0 {
		if C.eventIsQuit(&event) == C.SDL_TRUE {
			return false
		}
	}
	return true
}
