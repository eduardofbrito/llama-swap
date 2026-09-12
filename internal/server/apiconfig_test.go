package server

import (
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

const apiconfigTestConfig = `healthCheckTimeout: 30
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
	s := newTestServer(newStubRouter(nil, ""), newStubRouter(nil, ""))
	s.WithConfigEdit(writeTestConfigFile(t, apiconfigTestConfig), nil)

	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/config/model/alpha", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "11435") {
		t.Errorf("body missing model proxy: %s", w.Body.String())
	}

	// Unknown model -> 404.
	w = httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/config/model/ghost", nil))
	if w.Code != http.StatusNotFound {
		t.Errorf("ghost status = %d, want 404", w.Code)
	}
}

func TestServer_ConfigEndpoints_PutModel(t *testing.T) {
	path := writeTestConfigFile(t, apiconfigTestConfig)
	s := newTestServer(newStubRouter(nil, ""), newStubRouter(nil, ""))
	s.WithConfigEdit(path, nil)

	body := `{"yaml":"proxy: http://127.0.0.1:22435\nname: Alpha2\n"}`
	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPut, "/api/config/model/alpha", strings.NewReader(body)))
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
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPut, "/api/config/model/alpha", strings.NewReader(body)))
	if w.Code != http.StatusUnprocessableEntity {
		t.Errorf("invalid yaml status = %d, want 422", w.Code)
	}
	data, _ = os.ReadFile(path)
	if !strings.Contains(string(data), "22435") {
		t.Errorf("failed edit modified the file: %s", data)
	}
}

func TestServer_ConfigEndpoints_AddModel(t *testing.T) {
	path := writeTestConfigFile(t, apiconfigTestConfig)
	s := newTestServer(newStubRouter(nil, ""), newStubRouter(nil, ""))
	s.WithConfigEdit(path, nil)

	body := `{"id":"beta","yaml":"proxy: http://127.0.0.1:11436\n"}`
	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/config/model", strings.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	data, _ := os.ReadFile(path)
	if !strings.Contains(string(data), "beta") || !strings.Contains(string(data), "11436") {
		t.Errorf("model not added to file: %s", data)
	}

	// Duplicate id -> 422, missing id -> 400.
	w = httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/config/model", strings.NewReader(`{"id":"alpha","yaml":"proxy: http://127.0.0.1:1\n"}`)))
	if w.Code != http.StatusUnprocessableEntity {
		t.Errorf("duplicate status = %d, want 422", w.Code)
	}
	w = httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/config/model", strings.NewReader(`{"id":"  ","yaml":"proxy: http://127.0.0.1:1\n"}`)))
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
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPut, "/api/config/model/alpha", strings.NewReader(edited)))
	if w.Code != http.StatusOK {
		t.Fatalf("PUT status = %d, body = %s", w.Code, w.Body.String())
	}

	w = httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/config/reload?model=alpha", nil))
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
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/config/reload?model=alpha", nil))
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
	s := newTestServer(newStubRouter(nil, ""), newStubRouter(nil, ""))
	s.WithConfigEdit(writeTestConfigFile(t, apiconfigTestConfig), nil)
	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/config/reload", nil))
	if w.Code != http.StatusNotImplemented {
		t.Errorf("reload status = %d, want 501", w.Code)
	}

	// Reload wired -> 200 and the callback fires asynchronously.
	done := make(chan struct{})
	s = newTestServer(newStubRouter(nil, ""), newStubRouter(nil, ""))
	s.WithConfigEdit(writeTestConfigFile(t, apiconfigTestConfig), func() {
		close(done)
	})
	w = httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/config/reload", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("reload status = %d, want 200", w.Code)
	}
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Error("reload callback was not invoked")
	}
}
