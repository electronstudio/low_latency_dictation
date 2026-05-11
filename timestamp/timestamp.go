// Package timestamp provides utilities for formatting timestamps.
package timestamp

import "fmt"

// ToTimestamp converts a time in 10ms units to a formatted string.
// If comma is true, uses a comma as the millisecond separator (SRT style),
// otherwise uses a period (VTT style).
func ToTimestamp(t int64, comma bool) string {
	msec := t * 10
	hr := msec / (1000 * 60 * 60)
	msec -= hr * (1000 * 60 * 60)
	min := msec / (1000 * 60)
	msec -= min * (1000 * 60)
	sec := msec / 1000
	msec -= sec * 1000

	sep := "."
	if comma {
		sep = ","
	}

	if hr > 0 {
		return fmt.Sprintf("%02d:%02d:%02d%s%03d", hr, min, sec, sep, msec)
	}
	return fmt.Sprintf("%02d:%02d%s%03d", min, sec, sep, msec)
}
