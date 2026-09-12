package server

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
)

func writeTestConfigFile(t *testing.T, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatalf("writing config file: %v", err)
	}
	return p
}

const apiconfigTestKey = "test-config-key"

// newConfigEditServer builds a server whose config-editing endpoints are fully
// enabled: an apiKey is configured (the endpoints refuse to run without one)
// and a -config file path is wired in. It returns the server and the path of
// the config file on disk.
func newConfigEditServer(t *testing.T, content string, reloadFn func()) (*Server, string) {
	t.Helper()
	cfg := config.Config{RequiredAPIKeys: []string{apiconfigTestKey}}
	s := newTestServerWithConfig(cfg, newStubRouter(nil, ""), newStubRouter(nil, ""))
	path := writeTestConfigFile(t, content)
	s.WithConfigEdit(path, reloadFn)
	return s, path
}

// newConfigRequest builds an authenticated request for a config endpoint.
func newConfigRequest(method, path string, body io.Reader) *http.Request {
	r := httptest.NewRequest(method, path, body)
	r.Header.Set("Authorization", "Bearer "+apiconfigTestKey)
	return r
}

const apiconfigTestConfig = `healthCheckTimeout: 30
apiKeys:
  - ` + apiconfigTestKey + `
models:
  alpha:
    proxy: http://127.0.0.1:11435
`

func TestServer_ConfigEndpoints_NoConfigFile(t *testing.T) {
	s := newTestServer(newStubRouter(nil, ""), newStubRouter(nil, ""))

	for _, tc := range []struct {
		method string
		path   string
	}{
		{http.MethodGet, "/api/config/model/alpha"},
		{http.MethodPut, "/api/config/model/alpha"},
		{http.MethodPost, "/api/config/model"},
		{http.MethodPost, "/api/config/reload"},
	} {
		w := httptest.NewRecorder()
		s.ServeHTTP(w, httptest.NewRequest(tc.method, tc.path, strings.NewReader(`{}`)))
		if w.Code != http.StatusNotImplemented {
			t.Errorf("%s %s status = %d, want 501", tc.method, tc.path, w.Code)
		}
	}
}

func TestServer_ConfigEndpoints_GetModel(t *testing.T) {
	s, _ := newConfigEditServer(t, apiconfigTestConfig, nil)

	w := httptest.NewRecorder()
	s.ServeHTTP(w, newConfigRequest(http.MethodGet, "/api/config/model/alpha", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "11435") {
		t.Errorf("body missing model proxy: %s", w.Body.String())
	}

	// Unknown model -> 404.
	w = httptest.NewRecorder()
	s.ServeHTTP(w, newConfigRequest(http.MethodGet, "/api/config/model/ghost", nil))
	if w.Code != http.StatusNotFound {
		t.Errorf("ghost status = %d, want 404", w.Code)
	}
}

func TestServer_ConfigEndpoints_PutModel(t *testing.T) {
	s, path := newConfigEditServer(t, apiconfigTestConfig, nil)

	body := `{"yaml":"proxy: http://127.0.0.1:22435\nname: Alpha2\n"}`
	w := httptest.NewRecorder()
	s.ServeHTTP(w, newConfigRequest(http.MethodPut, "/api/config/model/alpha", strings.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	data, _ := os.ReadFile(path)
	if !strings.Contains(string(data), "22435") {
		t.Errorf("config file not updated: %s", data)
	}
	if !strings.Contains(string(data), "healthCheckTimeout") {
		t.Errorf("other top-level keys lost: %s", data)
	}

	// Invalid YAML -> 422 and the file must be unchanged.
	body = `{"yaml":"not: [a list\n"}`
	w = httptest.NewRecorder()
	s.ServeHTTP(w, newConfigRequest(http.MethodPut, "/api/config/model/alpha", strings.NewReader(body)))
	if w.Code != http.StatusUnprocessableEntity {
		t.Errorf("invalid yaml status = %d, want 422", w.Code)
	}
	data, _ = os.ReadFile(path)
	if !strings.Contains(string(data), "22435") {
		t.Errorf("failed edit modified the file: %s", data)
	}
}

func TestServer_ConfigEndpoints_AddModel(t *testing.T) {
	s, path := newConfigEditServer(t, apiconfigTestConfig, nil)

	body := `{"id":"beta","yaml":"proxy: http://127.0.0.1:11436\n"}`
	w := httptest.NewRecorder()
	s.ServeHTTP(w, newConfigRequest(http.MethodPost, "/api/config/model", strings.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	data, _ := os.ReadFile(path)
	if !strings.Contains(string(data), "beta") || !strings.Contains(string(data), "11436") {
		t.Errorf("model not added to file: %s", data)
	}

	// Duplicate id -> 422, missing id -> 400.
	w = httptest.NewRecorder()
	s.ServeHTTP(w, newConfigRequest(http.MethodPost, "/api/config/model", strings.NewReader(`{"id":"alpha","yaml":"proxy: http://127.0.0.1:1\n"}`)))
	if w.Code != http.StatusUnprocessableEntity {
		t.Errorf("duplicate status = %d, want 422", w.Code)
	}
	w = httptest.NewRecorder()
	s.ServeHTTP(w, newConfigRequest(http.MethodPost, "/api/config/model", strings.NewReader(`{"id":"  ","yaml":"proxy: http://127.0.0.1:1\n"}`)))
	if w.Code != http.StatusBadRequest {
		t.Errorf("missing id status = %d, want 400", w.Code)
	}
}

func TestServer_ConfigEndpoints_Reload_Surgical(t *testing.T) {
	// Live config must mirror the file BEFORE the edit, so the only delta the
	// reload sees is alpha's proxy. That delta is model-only -> surgical path.
	live, err := config.LoadConfigFromReader(strings.NewReader(apiconfigTestConfig))
	if err != nil {
		t.Fatal(err)
	}
	local := newStubRouter([]string{"alpha"}, "ok")
	fullReloads := make(chan struct{}, 1)
	path := writeTestConfigFile(t, apiconfigTestConfig)
	s := newTestServerWithConfig(live, local, newStubRouter(nil, ""))
	s.WithConfigEdit(path, func() { fullReloads <- struct{}{} })
	s.WithConfigDir("")

	// Edit alpha's proxy in the file, exactly as the Conf tab's PUT would.
	edited := `{"yaml":"proxy: http://127.0.0.1:22435\n"}`
	w := httptest.NewRecorder()
	s.ServeHTTP(w, newConfigRequest(http.MethodPut, "/api/config/model/alpha", strings.NewReader(edited)))
	if w.Code != http.StatusOK {
		t.Fatalf("PUT status = %d, body = %s", w.Code, w.Body.String())
	}

	w = httptest.NewRecorder()
	s.ServeHTTP(w, newConfigRequest(http.MethodPost, "/api/config/reload?model=alpha", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("surgical reload status = %d, body = %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), `"scope":"model"`) {
		t.Errorf("expected scope=model, got %s", w.Body.String())
	}
	if len(local.refreshModels) != 1 || local.refreshModels[0] != "alpha" {
		t.Errorf("refreshModels = %v, want [alpha]", local.refreshModels)
	}
	if !strings.Contains(string(mustRead(t, path)), "22435") {
		t.Error("config file not updated before reload")
	}
	// The live config snapshot must carry the new value.
	if got := s.Cfg().Models["alpha"].Proxy; got != "http://127.0.0.1:22435" {
		t.Errorf("live cfg alpha.proxy = %q, want the edited value", got)
	}
	select {
	case <-fullReloads:
		t.Error("full reload callback fired for a model-only edit")
	case <-time.After(100 * time.Millisecond):
	}
}

func TestServer_ConfigEndpoints_Reload_SurgicalFallback(t *testing.T) {
	// A global key (globalConcurrencyLimit) differs between the live config and
	// the file -> the diff is not model-only -> fall back to the full reload.
	liveYAML := apiconfigTestConfig // no globalConcurrencyLimit
	fileYAML := apiconfigTestConfig + "globalConcurrencyLimit: 5\n"
	live, err := config.LoadConfigFromReader(strings.NewReader(liveYAML))
	if err != nil {
		t.Fatal(err)
	}
	local := newStubRouter([]string{"alpha"}, "ok")
	fullReloads := make(chan struct{}, 1)
	path := writeTestConfigFile(t, fileYAML)
	s := newTestServerWithConfig(live, local, newStubRouter(nil, ""))
	s.WithConfigEdit(path, func() { fullReloads <- struct{}{} })
	s.WithConfigDir("")

	w := httptest.NewRecorder()
	s.ServeHTTP(w, newConfigRequest(http.MethodPost, "/api/config/reload?model=alpha", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), `"scope":"full"`) {
		t.Errorf("expected scope=full fallback, got %s", w.Body.String())
	}
	if len(local.refreshModels) != 0 {
		t.Errorf("refreshModels = %v, want none (full reload path)", local.refreshModels)
	}
	select {
	case <-fullReloads:
	case <-time.After(2 * time.Second):
		t.Error("full reload callback was not invoked on fallback")
	}
}

func mustRead(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	return data
}

func TestServer_ConfigEndpoints_Reload(t *testing.T) {
	// Reload not wired -> 501.
	s, _ := newConfigEditServer(t, apiconfigTestConfig, nil)
	w := httptest.NewRecorder()
	s.ServeHTTP(w, newConfigRequest(http.MethodPost, "/api/config/reload", nil))
	if w.Code != http.StatusNotImplemented {
		t.Errorf("reload status = %d, want 501", w.Code)
	}

	// Reload wired -> 200 and the callback fires asynchronously.
	done := make(chan struct{})
	s, _ = newConfigEditServer(t, apiconfigTestConfig, func() {
		close(done)
	})
	w = httptest.NewRecorder()
	s.ServeHTTP(w, newConfigRequest(http.MethodPost, "/api/config/reload", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("reload status = %d, want 200", w.Code)
	}
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Error("reload callback was not invoked")
	}
}

// TestServer_ConfigEndpoints_RefusedWithoutAPIKeys pins the security gate: a
// caller that can write a model block chooses that model's `cmd` and can then
// start it with a plain GET, so the editing endpoints must refuse to operate
// while the config declares no apiKeys (the auth middleware is a deliberate
// pass-through in that state).
func TestServer_ConfigEndpoints_RefusedWithoutAPIKeys(t *testing.T) {
	// Config file path is wired (the operator passed -enable-config-api) but
	// the live config has no apiKeys.
	s := newTestServerWithConfig(config.Config{}, newStubRouter(nil, ""), newStubRouter(nil, ""))
	path := writeTestConfigFile(t, apiconfigTestConfig)
	s.WithConfigEdit(path, func() { t.Error("reload must not run without apiKeys") })

	for _, tc := range []struct {
		method string
		path   string
		body   string
	}{
		{http.MethodGet, "/api/config/model/alpha", ""},
		{http.MethodPut, "/api/config/model/alpha", `{"yaml":"proxy: http://127.0.0.1:1\n"}`},
		{http.MethodPost, "/api/config/model", `{"id":"evil","yaml":"cmd: /bin/sh -c id\nproxy: http://127.0.0.1:1\n"}`},
		{http.MethodPost, "/api/config/reload", ""},
	} {
		w := httptest.NewRecorder()
		s.ServeHTTP(w, httptest.NewRequest(tc.method, tc.path, strings.NewReader(tc.body)))
		if w.Code != http.StatusForbidden {
			t.Errorf("%s %s status = %d, want 403", tc.method, tc.path, w.Code)
		}
	}

	// Nothing may have reached the file.
	if data := mustRead(t, path); strings.Contains(string(data), "evil") {
		t.Errorf("a refused request still wrote to the config file: %s", data)
	}
}

// TestServer_ConfigEndpoints_Status reports editability to the UI so it can
// hide controls that could only fail.
func TestServer_ConfigEndpoints_Status(t *testing.T) {
	// Editing off (no -enable-config-api): editable=false with a reason.
	s := newTestServer(newStubRouter(nil, ""), newStubRouter(nil, ""))
	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/config/status", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("status endpoint = %d, want 200", w.Code)
	}
	if !strings.Contains(w.Body.String(), `"editable":false`) {
		t.Errorf("want editable=false, got %s", w.Body.String())
	}

	// Fully enabled: editable=true.
	enabled, _ := newConfigEditServer(t, apiconfigTestConfig, nil)
	w = httptest.NewRecorder()
	enabled.ServeHTTP(w, newConfigRequest(http.MethodGet, "/api/config/status", nil))
	if !strings.Contains(w.Body.String(), `"editable":true`) {
		t.Errorf("want editable=true, got %s", w.Body.String())
	}
}
