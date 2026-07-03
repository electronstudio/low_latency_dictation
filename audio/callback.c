#include <SDL.h>
#include <SDL_audio.h>

extern void goAudioCallback(uintptr_t handle, Uint8* stream, int len);

void sdlCallback(void* userdata, Uint8* stream, int len) {
    goAudioCallback((uintptr_t)userdata, stream, len);
}
