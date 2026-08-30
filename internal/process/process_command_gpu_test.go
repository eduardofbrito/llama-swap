package process

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mostlygeek/llama-swap/internal/config"
)

// startWithOptsAndWaitReady starts the process via RunWithOptions and blocks
// until it is ready, mirroring runAsync but with Options applied.
func startWithOptsAndWaitReady(t *testing.T, p *ProcessCommand, opts Options) <-chan error {
	t.Helper()
	ch := make(chan error, 1)
	go func() { ch <- p.RunWithOptions(testStartTimeout, opts) }()
	ctx, cancel := context.WithTimeout(context.Background(), testStartTimeout)
	defer cancel()
	if err := p.WaitReady(ctx); err != nil {
		t.Fatalf("WaitReady: %v", err)
	}
	return ch
}

// newProxyFront wraps a ready process in an httptest server so requests can
// be proxied to the child, as the real router does.
func newProxyFront(t *testing.T, p *ProcessCommand) *httptest.Server {
	t.Helper()
	front := httptest.NewServer(p)
	t.Cleanup(front.Close)
	return front
}

// childEnvBody returns the child process's own environment as reported by
// simple-responder's /env endpoint: KEY=VALUE entries concatenated with no
// delimiter, so callers must match whole "KEY=value" substrings.
func childEnvBody(t *testing.T, proxyURL string) string {
	t.Helper()
	resp, err := http.Get(proxyURL + "/env")
	if err != nil {
		t.Fatalf("GET /env: %v", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read /env: %v", err)
	}
	return string(body)
}

// TestProcessCommand_GPUOverrideSetsChildEnv proves the UI's GPU selection
// reaches the child process: with GpuOverride set, CUDA_VISIBLE_DEVICES in
// the child env is exactly the override, a configured value for the same
// variable is replaced (not duplicated), and unrelated config env survives.
func TestProcessCommand_GPUOverrideSetsChildEnv(t *testing.T) {
	skipIfNoSimpleResponder(t)

	cmd, port := simpleResponderCmd(t, "-silent")
	p := newProcessCommand(t, config.ModelConfig{
		Cmd:                cmd,
		Proxy:              fmt.Sprintf("http://127.0.0.1:%d", port),
		CheckEndpoint:      "/health",
		HealthCheckTimeout: 10,
		Env:                []string{gpuEnvVar + "=7", "OTHER_VAR=keep"},
	})
	t.Cleanup(func() { p.Stop(testStopTimeout) }) //nolint: errcheck

	_ = startWithOptsAndWaitReady(t, p, Options{GpuOverride: "2"})
	front := newProxyFront(t, p)

	body := childEnvBody(t, front.URL)
	// The test values contain no '=' and no overlap with other env var
	// names, so whole "KEY=value" substring matching is unambiguous.
	if n := strings.Count(body, gpuEnvVar+"="); n != 1 {
		t.Errorf("child env has %d %s entries, want exactly 1", n, gpuEnvVar)
	}
	if !strings.Contains(body, gpuEnvVar+"=2") {
		t.Errorf("child env missing %s=2 (override must apply)", gpuEnvVar)
	}
	if strings.Contains(body, gpuEnvVar+"=7") {
		t.Errorf("child env still has configured %s=7 (override must replace it)", gpuEnvVar)
	}
	if !strings.Contains(body, "OTHER_VAR=keep") {
		t.Errorf("child env missing OTHER_VAR=keep (unrelated config env must survive)")
	}
}

// TestProcessCommand_NoGPUOverrideKeepsConfigEnv proves the default path:
// without an override the configured CUDA_VISIBLE_DEVICES passes through
// untouched, so "no selection in the UI" means "whatever the config says".
func TestProcessCommand_NoGPUOverrideKeepsConfigEnv(t *testing.T) {
	skipIfNoSimpleResponder(t)

	cmd, port := simpleResponderCmd(t, "-silent")
	p := newProcessCommand(t, config.ModelConfig{
		Cmd:                cmd,
		Proxy:              fmt.Sprintf("http://127.0.0.1:%d", port),
		CheckEndpoint:      "/health",
		HealthCheckTimeout: 10,
		Env:                []string{gpuEnvVar + "=5"},
	})
	t.Cleanup(func() { p.Stop(testStopTimeout) }) //nolint: errcheck

	_ = startWithOptsAndWaitReady(t, p, Options{})
	front := newProxyFront(t, p)

	body := childEnvBody(t, front.URL)
	if n := strings.Count(body, gpuEnvVar+"="); n != 1 {
		t.Fatalf("child env has %d %s entries, want exactly 1", n, gpuEnvVar)
	}
	if !strings.Contains(body, gpuEnvVar+"=5") {
		t.Errorf("child env missing %s=5 (config value must pass through unchanged)", gpuEnvVar)
	}
}
