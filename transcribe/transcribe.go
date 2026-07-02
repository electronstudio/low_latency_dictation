// Package transcribe provides Go wrappers around the whisper.cpp C API.
package transcribe

/*
#cgo CFLAGS: -I${SRCDIR}/../whisper.cpp/include -I${SRCDIR}/../whisper.cpp/ggml/include -I/usr/local/include
#cgo LDFLAGS: ${SRCDIR}/../libs/libwhisper.a ${SRCDIR}/../libs/libggml.a ${SRCDIR}/../libs/libggml-base.a ${SRCDIR}/../libs/libggml-cpu.a
#cgo linux LDFLAGS: -fopenmp -static-libgcc -Wl,-Bstatic -lstdc++ -Wl,-Bdynamic -lm
#cgo windows LDFLAGS: -static-libgcc -Wl,-Bstatic -Wl,--start-group -lgomp -lstdc++ -lwinpthread -Wl,--end-group -Wl,-Bdynamic -lm
#cgo darwin LDFLAGS: ${SRCDIR}/../libs/libggml-metal.a -framework Accelerate -framework Metal -framework Foundation -framework CoreGraphics -lstdc++ -lm

#include <whisper.h>
#include <ggml.h>
#include <ggml-backend.h>
#include <stdlib.h>

// GoDictationLogCallback is implemented in Go (see //export below) and used by
// the C trampoline to forward whisper.cpp log lines back into Go, where they
// are filtered by level and routed to the configured sink.
extern void GoDictationLogCallback(enum ggml_log_level level, char * text, void * user_data);

// dictationLogTrampoline adapts whisper.cpp's ggml_log_callback signature to
// the Go callback. It exists so we never hand a Go function pointer directly
// to C; whisper_log_set receives a plain C function pointer instead.
static void dictationLogTrampoline(enum ggml_log_level level, const char * text, void * user_data) {
	GoDictationLogCallback(level, (char *)text, user_data);
}

// dictationInstallLogCallback registers the trampoline as both the whisper and
// ggml log callback (whisper_log_set forwards to ggml_log_set). Must be called
// before any whisper/ggml call that emits log lines.
static void dictationInstallLogCallback(void) {
	whisper_log_set(dictationLogTrampoline, NULL);
}
*/
import "C"
import (
	"fmt"
	"os"
	"strings"
	"sync"
	"unsafe"
)

// WhisperSampleRate is the expected PCM sample rate in Hz.
const WhisperSampleRate = 16000

// LogLevel is the minimum ggml_log_level that whisper.cpp is allowed to emit.
// It mirrors the ggml_log_level enum in ggml.h (NONE=0, DEBUG=1, INFO=2,
// WARN=3, ERROR=4). LogLevelNone is intentionally above ERROR so it silences
// everything, including errors.
type LogLevel int

const (
	LogLevelNone  LogLevel = 100 // silence all whisper.cpp output
	LogLevelDebug LogLevel = 1   // GGML_LOG_LEVEL_DEBUG
	LogLevelInfo  LogLevel = 2   // GGML_LOG_LEVEL_INFO
	LogLevelWarn  LogLevel = 3   // GGML_LOG_LEVEL_WARN (default)
	LogLevelError LogLevel = 4   // GGML_LOG_LEVEL_ERROR
)

// ParseLogLevel maps a --log-level string to a LogLevel. Empty or unrecognised
// values fall back to LogLevelWarn, preserving the historical verbosity.
func ParseLogLevel(s string) LogLevel {
	switch strings.ToLower(s) {
	case "debug":
		return LogLevelDebug
	case "info":
		return LogLevelInfo
	case "warn", "warning":
		return LogLevelWarn
	case "error":
		return LogLevelError
	case "none", "silent", "quiet":
		return LogLevelNone
	default:
		return LogLevelWarn
	}
}

var (
	logMu        sync.Mutex
	logThreshold = int(LogLevelWarn)
	logSink      func(LogLevel, string)
)

// SetLogLevel sets the minimum ggml_log_level emitted by whisper.cpp. Lines
// below this level are dropped by the installed callback.
func SetLogLevel(level LogLevel) {
	logMu.Lock()
	logThreshold = int(level)
	logMu.Unlock()
}

// SetLogSink sets the destination for surviving whisper.cpp log lines. Pass
// nil to write lines to os.Stderr (whisper.cpp's default behaviour).
func SetLogSink(sink func(LogLevel, string)) {
	logMu.Lock()
	logSink = sink
	logMu.Unlock()
}

// InstallLogCallback registers the whisper.cpp/ggml log callback. It must be
// called before any whisper or ggml call that emits log lines, i.e. before
// BackendLoadAll and InitFromFile.
func InstallLogCallback() {
	C.dictationInstallLogCallback()
}

// GoDictationLogCallback is the Go side of the whisper.cpp log callback. It is
// exported (//export) so the C trampoline can invoke it. It filters lines by
// the configured level and forwards survivors to the configured sink.
//
//export GoDictationLogCallback
func GoDictationLogCallback(level C.enum_ggml_log_level, text *C.char, _ unsafe.Pointer) {
	lvl := LogLevel(level)
	logMu.Lock()
	threshold := logThreshold
	sink := logSink
	logMu.Unlock()
	if int(lvl) < threshold {
		return
	}
	s := C.GoString(text)
	if sink != nil {
		sink(lvl, s)
	} else {
		os.Stderr.WriteString(s)
	}
}

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
	if len(samples) == 0 {
		return nil
	}

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
