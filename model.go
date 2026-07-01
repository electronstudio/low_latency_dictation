package main

import (
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const baseURL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"

// httpClient is used for model downloads. The timeout covers the entire
// request, including reading the response body, so large models do not hang
// indefinitely on a stalled connection.
var httpClient = &http.Client{Timeout: 60 * time.Minute}

// modelSHASums maps the whisper.cpp model key (without the "ggml-" prefix
// and ".bin" suffix) to the expected SHA-1 digest of the downloaded file.
// Models not in this list are downloaded without integrity verification.
var modelSHASums = map[string]string{
	"tiny":                "bd577a113a864445d4c299885e0cb97d4ba92b5f",
	"tiny-q5_1":           "2827a03e495b1ed3048ef28a6a4620537db4ee51",
	"tiny-q8_0":           "19e8118f6652a650569f5a949d962154e01571d9",
	"tiny.en":             "c78c86eb1a8faa21b369bcd33207cc90d64ae9df",
	"tiny.en-q5_1":        "3fb92ec865cbbc769f08137f22470d6b66e071b6",
	"tiny.en-q8_0":        "802d6668e7d411123e672abe4cb6c18f12306abb",
	"base":                "465707469ff3a37a2b9b8d8f89f2f99de7299dac",
	"base-q5_1":           "a3733eda680ef76256db5fc5dd9de8629e62c5e7",
	"base-q8_0":           "7bb89bb49ed6955013b166f1b6a6c04584a20fbe",
	"base.en":             "137c40403d78fd54d454da0f9bd998f78703390c",
	"base.en-q5_1":        "d26d7ce5a1b6e57bea5d0431b9c20ae49423c94a",
	"base.en-q8_0":        "bb1574182e9b924452bf0cd1510ac034d323e948",
	"small":               "55356645c2b361a969dfd0ef2c5a50d530afd8d5",
	"small-q5_1":          "6fe57ddcfdd1c6b07cdcc73aaf620810ce5fc771",
	"small-q8_0":          "bcad8a2083f4e53d648d586b7dbc0cd673d8afad",
	"small.en":            "db8a495a91d927739e50b3fc1cc4c6b8f6c2d022",
	"small.en-q5_1":       "20f54878d608f94e4a8ee3ae56016571d47cba34",
	"small.en-q8_0":       "9d75ff4ccfa0a8217870d7405cf8cef0a5579852",
	"small.en-tdrz":       "b6c6e7e89af1a35c08e6de56b66ca6a02a2fdfa1",
	"medium":              "fd9727b6e1217c2f614f9b698455c4ffd82463b4",
	"medium-q5_0":         "7718d4c1ec62ca96998f058114db98236937490e",
	"medium-q8_0":         "e66645948aff4bebbec71b3485c576f3d63af5d6",
	"medium.en":           "8c30f0e44ce9560643ebd10bbe50cd20eafd3723",
	"medium.en-q5_0":      "bb3b5281bddd61605d6fc76bc5b92d8f20284c3b",
	"medium.en-q8_0":      "b1cf48c12c807e14881f634fb7b6c6ca867f6b38",
	"large-v1":            "b1caaf735c4cc1429223d5a74f0f4d0b9b59a299",
	"large-v2":            "0f4c8e34f21cf1a914c59d8b3ce882345ad349d6",
	"large-v2-q5_0":       "00e39f2196344e901b3a2bd5814807a769bd1630",
	"large-v2-q8_0":       "da97d6ca8f8ffbeeb5fd147f79010eeea194ba38",
	"large-v3":            "ad82bf6a9043ceed055076d0fd39f5f186ff8062",
	"large-v3-q5_0":       "e6e2ed78495d403bef4b7cff42ef4aaadcfea8de",
	"large-v3-turbo":      "4af2b29d7ec73d781377bfd1758ca957a807e941",
	"large-v3-turbo-q5_0": "e050f7970618a659205450ad97eb95a18d69c9ee",
	"large-v3-turbo-q8_0": "01bf15bedffe9f39d65c1b6ff9b687ea91f59e0e",
}

// expectedSHAForModel extracts the model key from a whisper.cpp filename
// (e.g. "ggml-tiny.en-q8_0.bin" -> "tiny.en-q8_0") and looks up its
// expected SHA-1 digest. The second return value is false if the model is
// not in the known list.
func expectedSHAForModel(modelName string) (string, bool) {
	key := strings.TrimPrefix(modelName, "ggml-")
	key = strings.TrimSuffix(key, ".bin")
	sha, ok := modelSHASums[key]
	return sha, ok
}

// countingWriter wraps a writer and prints a progress percentage.
type countingWriter struct {
	w       io.Writer
	current int64
	total   int64
	model   string
}

func (cw *countingWriter) Write(p []byte) (int, error) {
	n, err := cw.w.Write(p)
	cw.current += int64(n)
	if cw.total > 0 {
		pct := float64(cw.current) * 100.0 / float64(cw.total)
		fmt.Printf("\rDownloading %s: %.1f%%", cw.model, pct)
	}
	return n, err
}

func resolveModelFile(modelName string) (string, error) {
	cacheBase, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("failed to get cache directory: %w", err)
	}

	cacheDir := filepath.Join(cacheBase, "low_latency_dictation")
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create cache directory %s: %w", cacheDir, err)
	}

	modelPath := filepath.Join(cacheDir, modelName)
	if _, err := os.Stat(modelPath); err == nil {
		return modelPath, nil
	}

	expectedSHA, verify := expectedSHAForModel(modelName)

	url := baseURL + modelName
	fmt.Printf("Downloading %s...\n", modelName)

	resp, err := httpClient.Get(url)
	if err != nil {
		return "", fmt.Errorf("failed to download %s: %w", modelName, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("failed to download %s: HTTP %s", modelName, resp.Status)
	}

	tmp, err := os.CreateTemp(cacheDir, modelName+".tmp-*")
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}
	tmpPath := tmp.Name()
	defer tmp.Close()

	hasher := sha1.New()
	mw := io.MultiWriter(tmp, hasher)
	cw := &countingWriter{w: mw, total: resp.ContentLength, model: modelName}

	buf := make([]byte, 32*1024)
	if _, err = io.CopyBuffer(cw, resp.Body, buf); err != nil {
		os.Remove(tmpPath)
		return "", fmt.Errorf("failed to download %s: %w", modelName, err)
	}

	fmt.Println()

	if err = tmp.Close(); err != nil {
		os.Remove(tmpPath)
		return "", fmt.Errorf("failed to close temp file: %w", err)
	}

	if verify {
		gotSHA := hex.EncodeToString(hasher.Sum(nil))
		if gotSHA != expectedSHA {
			os.Remove(tmpPath)
			return "", fmt.Errorf("SHA-1 mismatch for %s: got %s, want %s", modelName, gotSHA, expectedSHA)
		}
		fmt.Printf("Verified %s SHA-1: %s\n", modelName, gotSHA)
	}

	if err = os.Rename(tmpPath, modelPath); err != nil {
		os.Remove(tmpPath)
		return "", fmt.Errorf("failed to move downloaded file to cache: %w", err)
	}

	fmt.Printf("Downloading %s: done\n", modelName)
	return modelPath, nil
}
