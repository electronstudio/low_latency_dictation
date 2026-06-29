//go:build linux

package toast

import (
	"bytes"
	"fmt"
	"log"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
)

var (
	mu         sync.Mutex
	lastID     int    // 0 = create new; notify-send treats --replace-id=0 as new
	notifyPath string // "" until Init succeeds
	logger     *log.Logger
)

// initPlatform verifies that notify-send is installed and on PATH.
func initPlatform(lg *log.Logger) error {
	p, err := exec.LookPath("notify-send")
	if err != nil {
		return fmt.Errorf("toast: notify-send not found: %w", err)
	}
	mu.Lock()
	notifyPath = p
	logger = lg
	mu.Unlock()
	logf("toast: init ok, notify-send at %s", p)
	return nil
}

// show runs notify-send, replacing the previous notification from this
// process. It stores the new notification id for use as --replace-id next time.
func show(title, message string, persist bool) error {
	mu.Lock()
	defer mu.Unlock()
	if notifyPath == "" {
		return fmt.Errorf("toast: not initialized")
	}
	urgency := "normal"
	timeoutMs := 5000
	if persist {
		urgency = "critical"
		timeoutMs = 1000000
	}
	args := []string{
		fmt.Sprintf("--urgency=%s", urgency),
		"--print-id",
		fmt.Sprintf("--replace-id=%d", lastID),
		"--transient",
		"--app-name=uk.co.electronstudio.dictate.notify",
		fmt.Sprintf("--expire-time=%d", timeoutMs),
		title,
	}
	if message != "" {
		args = append(args, message)
	}
	logf("toast: running: %s %s", notifyPath, strings.Join(args, " "))

	var stdout, stderr bytes.Buffer
	cmd := exec.Command(notifyPath, args...)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()

	if cmd.ProcessState != nil {
		logf("toast: exit code: %d", cmd.ProcessState.ExitCode())
	}
	logf("toast: stdout: %q", stdout.String())
	logf("toast: stderr: %q", stderr.String())

	if err != nil {
		return fmt.Errorf("toast: notify-send failed: %w", err)
	}
	if m := idRe.Find(stdout.Bytes()); m != nil {
		if n, perr := strconv.Atoi(string(m)); perr == nil {
			lastID = n
			logf("toast: parsed notification id: %d", n)
		} else {
			logf("toast: failed to parse id %q: %v", m, perr)
		}
	} else {
		logf("toast: no id found in stdout")
	}
	if !persist {
		lastID = 0
	}
	return nil
}

// logf writes a diagnostic line to the configured logger, if any.
func logf(format string, args ...interface{}) {
	if logger != nil {
		logger.Printf(format, args...)
	}
}

// idRe matches the first integer in notify-send's stdout (the notification
// id), robust to the exact line format.
var idRe = regexp.MustCompile(`\d+`)
