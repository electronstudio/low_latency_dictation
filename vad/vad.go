// Package vad provides a simple voice-activity detector and a high-pass filter.
package vad

import "math"

// HighPassFilter applies a first-order high-pass filter to the provided audio data
// in-place.  cutoff is in Hz, sampleRate in samples/second.
func HighPassFilter(data []float32, cutoff, sampleRate float32) {
	if len(data) == 0 {
		return
	}
	rc := 1.0 / (2.0 * math.Pi * float64(cutoff))
	dt := 1.0 / float64(sampleRate)
	alpha := dt / (rc + dt)

	y := float64(data[0])
	for i := 1; i < len(data); i++ {
		y = alpha * (y + float64(data[i]) - float64(data[i-1]))
		data[i] = float32(y)
	}
}

// SimpleVAD returns true when the *last* lastMs milliseconds of audio contain
// relatively low energy — i.e. no speech is detected.
//
// It is the inverse of the typical VAD: returning true means "no speech".
func SimpleVAD(pcmf32 []float32, sampleRate int, lastMs int, vadThold, freqThold float32, verbose bool) bool {
	nSamples := len(pcmf32)
	nSamplesLast := (sampleRate * lastMs) / 1000

	if nSamplesLast >= nSamples {
		// Not enough samples – assume no speech
		return true
	}

	if freqThold > 0.0 {
		HighPassFilter(pcmf32, freqThold, float32(sampleRate))
	}

	var energyAll, energyLast float32
	for i := 0; i < nSamples; i++ {
		v := pcmf32[i]
		if v < 0 {
			v = -v
		}
		energyAll += v
		if i >= nSamples-nSamplesLast {
			energyLast += v
		}
	}

	energyAll /= float32(nSamples)
	energyLast /= float32(nSamplesLast)

	if verbose {
		// Uses fmt.Fprintf to stderr via a log would be better, but keeping parity with C++
		// version which printed to stderr.
		println("vad_simple: energy_all:", energyAll, "energy_last:", energyLast,
			"vad_thold:", vadThold, "freq_thold:", freqThold)
	}

	// If the recent energy is *higher* than the threshold relative to the average,
	// we consider that speech — so return false (not quiet).
	if energyLast > vadThold*energyAll {
		return false
	}
	return true
}
