#ifndef AUDIO_CALLBACK_H
#define AUDIO_CALLBACK_H

#include <SDL.h>
#include <SDL_audio.h>
#include <stdlib.h>

// C -> Go bridge.  Defined in a Go file with //export.
extern void goAudioCallback(void* userdata, Uint8* stream, int len);

// Forwarder called by SDL that relays to the Go callback.
void sdlCallback(void* userdata, Uint8* stream, int len);

static inline void setResampleHint(void) {
    SDL_SetHintWithPriority("SDL_AUDIO_RESAMPLING_MODE", "medium", SDL_HINT_OVERRIDE);
}

static inline void zeroAudioSpec(SDL_AudioSpec* spec) {
    memset(spec, 0, sizeof(SDL_AudioSpec));
}

static inline SDL_bool eventIsQuit(SDL_Event* ev) {
    return ev->type == SDL_QUIT;
}

#endif
