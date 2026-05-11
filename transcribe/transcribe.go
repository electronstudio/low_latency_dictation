// Package transcribe provides Go wrappers around the whisper.cpp C API.
package transcribe

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: ${SRCDIR}/../libs/libwhisper.a ${SRCDIR}/../libs/libggml.a ${SRCDIR}/../libs/libggml-base.a ${SRCDIR}/../libs/libggml-cpu.a ${SRCDIR}/../libs/libggml-vulkan.a -lvulkan -fopenmp -lstdc++ -lm

#include <whisper.h>
#include <ggml-backend.h>
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// WhisperSampleRate is the expected PCM sample rate in Hz.
const WhisperSampleRate = 16000

type SamplingStrategy int

const (
	SamplingGreedy     SamplingStrategy = C.WHISPER_SAMPLING_GREEDY
	SamplingBeamSearch SamplingStrategy = C.WHISPER_SAMPLING_BEAM_SEARCH
)

// Context wraps a whisper_context*.
type Context struct {
	ctx *C.struct_whisper_context
}

// ContextParams holds parameters for initialising a whisper context.
type ContextParams struct {
	UseGPU    bool
	FlashAttn bool
}

// DefaultContextParams returns the default context parameters.
func DefaultContextParams() ContextParams {
	cp := C.whisper_context_default_params()
	return ContextParams{
		UseGPU:    bool(cp.use_gpu),
		FlashAttn: bool(cp.flash_attn),
	}
}

// InitFromFile loads a model and returns a Context.
func InitFromFile(modelPath string, params ContextParams) (*Context, error) {
	cparams := C.whisper_context_default_params()
	cparams.use_gpu = C.bool(params.UseGPU)
	cparams.flash_attn = C.bool(params.FlashAttn)

	cpath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cpath))

	ctx := C.whisper_init_from_file_with_params(cpath, cparams)
	if ctx == nil {
		return nil, fmt.Errorf("failed to initialize whisper context")
	}
	return &Context{ctx: ctx}, nil
}

// BackendLoadAll pre-loads all registered GGML backends.
func BackendLoadAll() {
	C.ggml_backend_load_all()
}

// Free releases the whisper context.
func (c *Context) Free() {
	if c.ctx != nil {
		C.whisper_free(c.ctx)
		c.ctx = nil
	}
}

// PrintTimings prints performance timing information to stderr.
func (c *Context) PrintTimings() {
	C.whisper_print_timings(c.ctx)
}

// FullParams holds parameters for a single transcription run.
type FullParams struct {
	Strategy       SamplingStrategy
	PrintProgress  bool
	PrintSpecial   bool
	PrintRealtime  bool
	PrintTimestamp bool
	Translate      bool
	SingleSegment  bool
	MaxTokens      int
	Language       string
	NThreads       int
	AudioCtx       int
	TdrzEnable     bool
	NoFallback     bool
	SuppressNST    bool
	BeamSize       int
	PromptTokens   []int32
}

// Full runs the full transcription pipeline on the provided samples.
func (c *Context) Full(params FullParams, samples []float32) error {
	var cpp C.struct_whisper_full_params
	if params.Strategy == SamplingBeamSearch {
		cpp = C.whisper_full_default_params(C.WHISPER_SAMPLING_BEAM_SEARCH)
	} else {
		cpp = C.whisper_full_default_params(C.WHISPER_SAMPLING_GREEDY)
	}

	cpp.print_progress = C.bool(params.PrintProgress)
	cpp.print_special = C.bool(params.PrintSpecial)
	cpp.print_realtime = C.bool(params.PrintRealtime)
	cpp.print_timestamps = C.bool(params.PrintTimestamp)
	cpp.translate = C.bool(params.Translate)
	cpp.single_segment = C.bool(params.SingleSegment)
	cpp.max_tokens = C.int(params.MaxTokens)
	cpp.n_threads = C.int(params.NThreads)
	cpp.audio_ctx = C.int(params.AudioCtx)
	cpp.tdrz_enable = C.bool(params.TdrzEnable)
	cpp.suppress_nst = C.bool(params.SuppressNST)

	if params.BeamSize > 1 {
		cpp.beam_search.beam_size = C.int(params.BeamSize)
	}

	if params.NoFallback {
		// whisper_full_default_params sets temperature_inc to some value;
		// setting it to 0 disables temperature fallback.
		cpp.temperature_inc = 0.0
	}

	var langPtr *C.char
	if params.Language != "" {
		langPtr = C.CString(params.Language)
		defer C.free(unsafe.Pointer(langPtr))
	}
	cpp.language = langPtr

	var promptPtr *C.whisper_token
	if len(params.PromptTokens) > 0 {
		promptPtr = (*C.whisper_token)(unsafe.Pointer(&params.PromptTokens[0]))
	}
	cpp.prompt_tokens = promptPtr
	cpp.prompt_n_tokens = C.int(len(params.PromptTokens))

	ret := C.whisper_full(c.ctx, cpp, (*C.float)(unsafe.Pointer(&samples[0])), C.int(len(samples)))
	if ret != 0 {
		return fmt.Errorf("whisper_full failed with code %d", ret)
	}
	return nil
}

// NSegments returns the number of text segments produced by the last Full call.
func (c *Context) NSegments() int {
	return int(C.whisper_full_n_segments(c.ctx))
}

// SegmentText returns the text for segment i.
func (c *Context) SegmentText(i int) string {
	return C.GoString(C.whisper_full_get_segment_text(c.ctx, C.int(i)))
}

// SegmentT0 returns the start timestamp of segment i (in 10ms units, roughly).
func (c *Context) SegmentT0(i int) int64 {
	return int64(C.whisper_full_get_segment_t0(c.ctx, C.int(i)))
}

// SegmentT1 returns the end timestamp of segment i.
func (c *Context) SegmentT1(i int) int64 {
	return int64(C.whisper_full_get_segment_t1(c.ctx, C.int(i)))
}

// SegmentSpeakerTurnNext returns true if the next segment is a speaker turn.
func (c *Context) SegmentSpeakerTurnNext(i int) bool {
	return bool(C.whisper_full_get_segment_speaker_turn_next(c.ctx, C.int(i)))
}
