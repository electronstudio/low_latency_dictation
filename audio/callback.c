#include "callback.h"

void sdlCallback(void* userdata, Uint8* stream, int len) {
    goAudioCallback(userdata, stream, len);
}
