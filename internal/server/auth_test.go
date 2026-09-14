package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mostlygeek/llama-swap/internal/config"
)

func TestServer_SanitizeAccessControlRequestHeaders(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"Content-Type, Authorization", "Content-Type, Authorization"},
		{"  X-Custom ,  Accept ", "X-Custom, Accept"},
		{"Valid, Bad Header", "Valid"},
		{"Bad@Header", ""},
		{"", ""},
	}
	for _, c := range cases {
		if got := sanitizeAccessControlRequestHeaderValues(c.in); got != c.want {
			t.Errorf("sanitize(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestServer_IsTokenChar(t *testing.T) {
	for _, r := range "abcXYZ0129!#$%&'*+-.^_`|~" {
		if !isTokenChar(r) {
			t.Errorf("isTokenChar(%q) = false, want true", r)
		}
	}
	for _, r := range " @()/\t\"" {
		if isTokenChar(r) {
			t.Errorf("isTokenChar(%q) = true, want false", r)
		}
	}
}

func TestServer_RequestContextMiddleware(t *testing.T) {
	cfg := config.Config{
		Models: map[string]config.ModelConfig{
			"llama3": {},
		},
	}

	final := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	mw := CreateRequestContextMiddleware(cfgAt(cfg))

	t.Run("known model passes through", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"llama3"}`))
		r.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		mw(final).ServeHTTP(w, r)
		if w.Code != http.StatusOK {
			t.Errorf("status = %d, want 200", w.Code)
		}
	})

	t.Run("missing model returns 404", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`))
		r.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		mw(final).ServeHTTP(w, r)
		if w.Code != http.StatusNotFound {
			t.Errorf("status = %d, want 404", w.Code)
		}
	})
}

func TestServer_AuthMiddleware(t *testing.T) {
	final := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	t.Run("no keys configured passes through", func(t *testing.T) {
		mw := CreateAuthMiddleware(cfgAt(config.Config{}), config.Config.InferenceAPIKeys)
		w := httptest.NewRecorder()
		mw(final).ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/", nil))
		if w.Code != http.StatusOK {
			t.Errorf("status = %d, want 200", w.Code)
		}
	})

	cfg := config.Config{RequiredAPIKeys: []string{"secret"}}

	t.Run("valid key", func(t *testing.T) {
		mw := CreateAuthMiddleware(cfgAt(cfg), config.Config.InferenceAPIKeys)
		r := httptest.NewRequest(http.MethodGet, "/", nil)
		r.Header.Set("Authorization", "Bearer secret")
		w := httptest.NewRecorder()
		mw(final).ServeHTTP(w, r)
		if w.Code != http.StatusOK {
			t.Errorf("status = %d, want 200", w.Code)
		}
	})

	t.Run("invalid key", func(t *testing.T) {
		mw := CreateAuthMiddleware(cfgAt(cfg), config.Config.InferenceAPIKeys)
		r := httptest.NewRequest(http.MethodGet, "/", nil)
		r.Header.Set("Authorization", "Bearer wrong")
		w := httptest.NewRecorder()
		mw(final).ServeHTTP(w, r)
		if w.Code != http.StatusUnauthorized {
			t.Errorf("status = %d, want 401", w.Code)
		}
		if w.Header().Get("WWW-Authenticate") == "" {
			t.Error("missing WWW-Authenticate header")
		}
	})
}

// TestServer_AuthMiddleware_KeySeparation pins the split between the
// inference key set and the UI key set: an inference-only apiKeys entry must
// not open the dashboard/control-plane surface, while a uiApiKeys entry must
// work on BOTH — the whole point of a UI login also driving the Playground's
// real inference calls.
func TestServer_AuthMiddleware_KeySeparation(t *testing.T) {
	final := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	req := func(key string) *http.Request {
		r := httptest.NewRequest(http.MethodGet, "/", nil)
		if key != "" {
			r.Header.Set("Authorization", "Bearer "+key)
		}
		return r
	}
	status := func(mw func(http.Handler) http.Handler, key string) int {
		w := httptest.NewRecorder()
		mw(final).ServeHTTP(w, req(key))
		return w.Code
	}

	cfg := config.Config{
		RequiredAPIKeys:   []string{"inference-key"},
		UIRequiredAPIKeys: []string{"ui-key"},
	}
	inferenceMW := CreateAuthMiddleware(cfgAt(cfg), config.Config.InferenceAPIKeys)
	uiMW := CreateAuthMiddleware(cfgAt(cfg), config.Config.UIAPIKeys)

	t.Run("inference key works on the inference surface", func(t *testing.T) {
		if got := status(inferenceMW, "inference-key"); got != http.StatusOK {
			t.Errorf("status = %d, want 200", got)
		}
	})

	t.Run("inference key does NOT work on the UI surface", func(t *testing.T) {
		if got := status(uiMW, "inference-key"); got != http.StatusUnauthorized {
			t.Errorf("status = %d, want 401; an inference-only key must not open the dashboard", got)
		}
	})

	t.Run("UI key works on the UI surface", func(t *testing.T) {
		if got := status(uiMW, "ui-key"); got != http.StatusOK {
			t.Errorf("status = %d, want 200", got)
		}
	})

	t.Run("UI key ALSO works on the inference surface", func(t *testing.T) {
		if got := status(inferenceMW, "ui-key"); got != http.StatusOK {
			t.Errorf("status = %d, want 200; a dashboard session must be able to drive the Playground", got)
		}
	})

	t.Run("wrong key rejected on both surfaces", func(t *testing.T) {
		if got := status(inferenceMW, "wrong"); got != http.StatusUnauthorized {
			t.Errorf("inference status = %d, want 401", got)
		}
		if got := status(uiMW, "wrong"); got != http.StatusUnauthorized {
			t.Errorf("UI status = %d, want 401", got)
		}
	})
}

// TestServer_AuthMiddleware_UIKeyFallback pins backward compatibility: a
// config that only ever set apiKeys (never learned about uiApiKeys) must keep
// gating the UI/control-plane surface exactly as before this feature existed
// — apiKeys alone still protects everything.
func TestServer_AuthMiddleware_UIKeyFallback(t *testing.T) {
	final := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	cfg := config.Config{RequiredAPIKeys: []string{"shared-key"}} // no uiApiKeys set
	uiMW := CreateAuthMiddleware(cfgAt(cfg), config.Config.UIAPIKeys)

	r := httptest.NewRequest(http.MethodGet, "/ui/", nil)
	r.Header.Set("Authorization", "Bearer shared-key")
	w := httptest.NewRecorder()
	uiMW(final).ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200; apiKeys must still gate the UI when uiApiKeys is unset", w.Code)
	}

	w = httptest.NewRecorder()
	uiMW(final).ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/ui/", nil))
	if w.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401 without a key", w.Code)
	}
}

// TestServer_RouteAuthSeparation drives the real mux (routes() wires it, not
// a middleware built in isolation) to pin exactly which key set each family
// of endpoints accepts, once both apiKeys and uiApiKeys are configured with
// different values.
func TestServer_RouteAuthSeparation(t *testing.T) {
	cfg := config.Config{
		RequiredAPIKeys:   []string{"inference-key"},
		UIRequiredAPIKeys: []string{"ui-key"},
		Models:            map[string]config.ModelConfig{config.ComfyUIModelID: {}},
	}
	local := newStubRouter([]string{config.ComfyUIModelID}, "ok")
	s := newTestServerWithConfig(cfg, local, newStubRouter(nil, ""))

	get := func(path, key string) int {
		r := httptest.NewRequest(http.MethodGet, path, nil)
		if key != "" {
			r.Header.Set("Authorization", "Bearer "+key)
		}
		w := httptest.NewRecorder()
		s.ServeHTTP(w, r)
		return w.Code
	}

	// "/upstream" (no trailing slash, no model) is a bare, deliberately
	// unauthenticated redirect to the docs page — /upstream/{model}/ is the
	// real, keyed passthrough.
	inferenceRoutes := []string{"/v1/models", "/comfyui/", "/upstream/foo/"}
	uiRoutes := []string{"/ui/", "/api/version", "/api/gpus"}

	for _, path := range inferenceRoutes {
		t.Run("inference route "+path+" accepts the inference key", func(t *testing.T) {
			if got := get(path, "inference-key"); got == http.StatusUnauthorized {
				t.Errorf("status = %d, want anything but 401", got)
			}
		})
		t.Run("inference route "+path+" accepts the UI key too", func(t *testing.T) {
			if got := get(path, "ui-key"); got == http.StatusUnauthorized {
				t.Errorf("status = %d, want anything but 401; a UI key must drive inference too", got)
			}
		})
		t.Run("inference route "+path+" rejects no key", func(t *testing.T) {
			if got := get(path, ""); got != http.StatusUnauthorized {
				t.Errorf("status = %d, want 401", got)
			}
		})
	}

	for _, path := range uiRoutes {
		t.Run("UI route "+path+" accepts the UI key", func(t *testing.T) {
			if got := get(path, "ui-key"); got == http.StatusUnauthorized {
				t.Errorf("status = %d, want anything but 401", got)
			}
		})
		t.Run("UI route "+path+" rejects the inference-only key", func(t *testing.T) {
			if got := get(path, "inference-key"); got != http.StatusUnauthorized {
				t.Errorf("status = %d, want 401; an inference key must not reach the dashboard", got)
			}
		})
		t.Run("UI route "+path+" rejects no key", func(t *testing.T) {
			if got := get(path, ""); got != http.StatusUnauthorized {
				t.Errorf("status = %d, want 401", got)
			}
		})
	}
}
