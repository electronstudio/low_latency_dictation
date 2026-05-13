package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

const baseURL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"

func resolveModelFile(modelName string) (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get home directory: %w", err)
	}

	cacheDir := filepath.Join(home, ".cache", "low_latency_dictation")
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create cache directory %s: %w", cacheDir, err)
	}

	modelPath := filepath.Join(cacheDir, modelName)
	if _, err := os.Stat(modelPath); err == nil {
		return modelPath, nil
	}

	url := baseURL + modelName
	fmt.Printf("Downloading %s...\n", modelName)

	resp, err := http.Get(url)
	if err != nil {
		return "", fmt.Errorf("failed to download %s: %w", modelName, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("failed to download %s: HTTP %s", modelName, resp.Status)
	}

	total := resp.ContentLength

	tmp, err := os.CreateTemp(cacheDir, modelName+".tmp-*")
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}
	tmpPath := tmp.Name()

	defer func() {
		tmp.Close()
		if err != nil {
			os.Remove(tmpPath)
		}
	}()

	var downloaded int64
	buf := make([]byte, 32*1024)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := tmp.Write(buf[:n]); werr != nil {
				err = fmt.Errorf("failed to write temp file: %w", werr)
				return "", err
			}
			downloaded += int64(n)
			if total > 0 {
				pct := float64(downloaded) * 100.0 / float64(total)
				fmt.Printf("\rDownloading %s: %.1f%%", modelName, pct)
			}
		}
		if readErr != nil {
			if readErr == io.EOF {
				break
			}
			err = fmt.Errorf("failed to download %s: %w", modelName, readErr)
			return "", err
		}
	}

	if err := tmp.Close(); err != nil {
		return "", fmt.Errorf("failed to close temp file: %w", err)
	}

	if err := os.Rename(tmpPath, modelPath); err != nil {
		return "", fmt.Errorf("failed to move downloaded file to cache: %w", err)
	}

	fmt.Printf("\rDownloading %s: done\n", modelName)
	return modelPath, nil
}
