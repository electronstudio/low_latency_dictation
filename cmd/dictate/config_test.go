package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWriteDefaultConfigContents(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")

	if err := writeDefaultConfig(path); err != nil {
		t.Fatalf("writeDefaultConfig: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read config: %v", err)
	}
	s := string(data)

	if !strings.Contains(s, "# model: ggml-tiny.en.bin") {
		t.Errorf("expected commented model default, got:\n%s", s)
	}
	if !strings.Contains(s, "# quality-preset:") {
		t.Errorf("expected commented quality-preset, got:\n%s", s)
	}
	if !strings.Contains(s, "# flash-attn: true") {
		t.Errorf("expected commented flash-attn default, got:\n%s", s)
	}
	if strings.Contains(s, "quality-preset: low") {
		t.Errorf("preset default should be commented out, got:\n%s", s)
	}
}

func TestLoadConfigFromPathDefaults(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")

	var cli CLI
	if err := loadConfigFromPath(path, &cli, true); err != nil {
		t.Fatalf("loadConfigFromPath: %v", err)
	}

	if cli.Model != "ggml-tiny.en.bin" {
		t.Errorf("Model = %q, want %q", cli.Model, "ggml-tiny.en.bin")
	}
	if cli.LengthMs != 30000 {
		t.Errorf("LengthMs = %d, want 30000", cli.LengthMs)
	}
	if cli.Threads != 0 {
		t.Errorf("Threads = %d, want 0", cli.Threads)
	}
	if cli.Language != "en" {
		t.Errorf("Language = %q, want %q", cli.Language, "en")
	}
	if !cli.FlashAttn {
		t.Errorf("FlashAttn = false, want true")
	}
}

func TestLoadConfigFromPathOverrides(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte(`model: ggml-base.en.bin
length: 60000
use-cpu: true
`), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	var cli CLI
	if err := loadConfigFromPath(path, &cli, false); err != nil {
		t.Fatalf("loadConfigFromPath: %v", err)
	}

	if cli.Model != "ggml-base.en.bin" {
		t.Errorf("Model = %q, want %q", cli.Model, "ggml-base.en.bin")
	}
	if cli.LengthMs != 60000 {
		t.Errorf("LengthMs = %d, want 60000", cli.LengthMs)
	}
	if !cli.UseCPU {
		t.Errorf("UseCPU = false, want true")
	}
	if cli.FinalModel != "ggml-small.en.bin" {
		t.Errorf("FinalModel = %q, want %q", cli.FinalModel, "ggml-small.en.bin")
	}
}

func TestLoadConfigFromPathUnknownKey(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte("unknown-key: value\n"), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	var cli CLI
	if err := loadConfigFromPath(path, &cli, false); err == nil {
		t.Errorf("expected error for unknown config key")
	}
}
