//go:build windows

package toast

import (
	"fmt"
	"log"
	"runtime"
	"sync"

	"github.com/go-ole/go-ole"
)

const appID = "uk.co.electronstudio.dictate"

var (
	mu          sync.Mutex
	logger      *log.Logger
	toastCh     chan toastRequest
	initialized bool
)

type toastRequest struct {
	title   string
	message string
	persist bool
}

// initPlatform stores the logger and registers the AUMID in the registry so
// Action Center shows an app name instead of a generic entry.
func initPlatform(lg *log.Logger) error {
	logger = lg
	if err := registerAppData(appID); err != nil {
		return fmt.Errorf("toast: %w", err)
	}
	logf("toast: init ok")
	return nil
}

// show sends the toast request to a dedicated worker goroutine. The worker runs
// COM on a single locked OS thread so Go's goroutine scheduler cannot migrate
// COM calls across Windows apartments, which otherwise interferes with the
// systray's message loop.
func show(title, message string, persist bool) error {
	mu.Lock()
	if !initialized {
		if err := startWorker(); err != nil {
			mu.Unlock()
			return err
		}
		initialized = true
	}
	mu.Unlock()

	req := toastRequest{title: title, message: message, persist: persist}
	select {
	case toastCh <- req:
		logf("toast: queued (persist=%v)", persist)
		return nil
	default:
		// Worker is busy; replace the stale buffered request with the latest one
		// so redrawText never blocks and only the most recent toast matters.
		select {
		case <-toastCh:
		default:
		}
		select {
		case toastCh <- req:
			logf("toast: queued (replaced stale, persist=%v)", persist)
			return nil
		default:
			return fmt.Errorf("toast: worker overloaded")
		}
	}
}

// startWorker launches the COM worker goroutine and waits for it to finish
// one-time initialization.
func startWorker() error {
	toastCh = make(chan toastRequest, 1)
	ready := make(chan error, 1)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()

		if err := ole.RoInitialize(1); err != nil {
			// S_FALSE means already initialized in this thread; RPC_E_CHANGED_MODE
			// means a different threading model is already active. Both are benign.
			type hrError interface{ HResult() uintptr }
			if he, ok := err.(hrError); ok {
				hr := he.HResult()
				if hr != 1 && hr != 0x80010106 { // S_FALSE, RPC_E_CHANGED_MODE
					ready <- fmt.Errorf("RoInitialize: %w", err)
					return
				}
			} else {
				ready <- fmt.Errorf("RoInitialize: %w", err)
				return
			}
		}
		notifier, err := newNotifier(appID)
		if err != nil {
			ready <- fmt.Errorf("create notifier: %w", err)
			return
		}
		defer notifier.Release()

		ready <- nil

		var prevToast *toastNotification
		for req := range toastCh {
			if prevToast != nil {
				_ = notifier.Hide(prevToast) // best-effort; ignore error
				prevToast.Release()
				prevToast = nil
			}
			duration := "short"
			if req.persist {
				duration = "long"
			}
			xml := buildToastXML(req.title, req.message, duration)
			t, err := createAndShow(notifier, xml)
			if err != nil {
				logf("toast: %v", err)
				continue
			}
			prevToast = t // retain — do NOT Release here; freed on next request
			logf("toast: shown (duration=%s)", duration)
		}
		if prevToast != nil {
			prevToast.Release()
		}
	}()
	if err := <-ready; err != nil {
		return fmt.Errorf("toast: %w", err)
	}
	return nil
}

// logf writes a diagnostic line to the configured logger, if any.
func logf(format string, args ...interface{}) {
	if logger != nil {
		logger.Printf(format, args...)
	}
}
