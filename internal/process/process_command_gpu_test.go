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
		Env:                []string{GPUEnvVar + "=7", "OTHER_VAR=keep"},
	})
	t.Cleanup(func() { p.Stop(testStopTimeout) }) //nolint: errcheck

	_ = startWithOptsAndWaitReady(t, p, Options{GpuOverride: "2"})
	front := newProxyFront(t, p)

	body := childEnvBody(t, front.URL)
	// The test values contain no '=' and no overlap with other env var
	// names, so whole "KEY=value" substring matching is unambiguous.
	if n := strings.Count(body, GPUEnvVar+"="); n != 1 {
		t.Errorf("child env has %d %s entries, want exactly 1", n, GPUEnvVar)
	}
	if !strings.Contains(body, GPUEnvVar+"=2") {
		t.Errorf("child env missing %s=2 (override must apply)", GPUEnvVar)
	}
	if strings.Contains(body, GPUEnvVar+"=7") {
		t.Errorf("child env still has configured %s=7 (override must replace it)", GPUEnvVar)
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
		Env:                []string{GPUEnvVar + "=5"},
	})
	t.Cleanup(func() { p.Stop(testStopTimeout) }) //nolint: errcheck

	_ = startWithOptsAndWaitReady(t, p, Options{})
	front := newProxyFront(t, p)

	body := childEnvBody(t, front.URL)
	if n := strings.Count(body, GPUEnvVar+"="); n != 1 {
		t.Fatalf("child env has %d %s entries, want exactly 1", n, GPUEnvVar)
	}
	if !strings.Contains(body, GPUEnvVar+"=5") {
		t.Errorf("child env missing %s=5 (config value must pass through unchanged)", GPUEnvVar)
	}
}

// TestDefaultGPU checks extraction of the model's configured GPU from its env list.
func TestDefaultGPU(t *testing.T) {
	cases := []struct {
		name string
		env  []string
		want string
	}{
		{"none set", []string{"OTHER=1"}, ""},
		{"empty env", nil, ""},
		{"simple index", []string{"CUDA_VISIBLE_DEVICES=0", "OTHER=1"}, "0"},
		{"not last entry", []string{"A=1", "CUDA_VISIBLE_DEVICES=2"}, "2"},
		{"list value", []string{"CUDA_VISIBLE_DEVICES=0,1,3"}, "0,1,3"},
		{"surrounding spaces", []string{"CUDA_VISIBLE_DEVICES= 4 "}, "4"},
		{"similar name ignored", []string{"MY_CUDA_VISIBLE_DEVICES=9", "CUDA_VISIBLE=1"}, ""},
		{"empty value", []string{"CUDA_VISIBLE_DEVICES="}, ""},
		{"duplicate uses first", []string{"CUDA_VISIBLE_DEVICES=1", "CUDA_VISIBLE_DEVICES=9"}, "1"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := DefaultGPU(tc.env); got != tc.want {
				t.Errorf("DefaultGPU(%v) = %q, want %q", tc.env, got, tc.want)
			}
		})
	}
}
