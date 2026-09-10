package server

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
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
