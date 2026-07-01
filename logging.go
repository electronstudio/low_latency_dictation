package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/electronstudio/low_latency_dictation/transcribe"
)

var logger *log.Logger

func logActionf(format string, args ...interface{}) {
	if logger != nil {
		logger.Printf(format, args...)
	}
}

// makeWhisperSink builds the sink handed to transcribe.SetLogSink. When a log
// file is configured, whisper.cpp lines are written through the same *log.Logger
// as the app's own actions (with a timestamp, trailing newline trimmed). With no
// log file, lines go raw to os.Stderr, preserving whisper.cpp's exact format.
func makeWhisperSink(lg *log.Logger) func(transcribe.LogLevel, string) {
	if lg != nil {
		return func(_ transcribe.LogLevel, text string) {
			lg.Print(strings.TrimRight(text, "\n"))
		}
	}
	return func(_ transcribe.LogLevel, text string) {
		os.Stderr.WriteString(text)
	}
}

// openLogFile creates (or appends to) the action log file and wires it to the
// package-level logger. The caller owns the returned file handle.
func openLogFile(cli CLI) *os.File {
	f, err := os.OpenFile(cli.LogFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: failed to open log file: %v\n", err)
		os.Exit(1)
	}
	logger = log.New(f, "", log.LstdFlags)
	return f
}

// setupWhisperLogging configures the whisper.cpp log level and sink before any
// whisper/ggml call so the level filter applies to backend discovery and
// model-load output.
func setupWhisperLogging(cli CLI) {
	transcribe.SetLogLevel(transcribe.ParseLogLevel(cli.LogLevel))
	transcribe.SetLogSink(makeWhisperSink(logger))
	transcribe.InstallLogCallback()
}
