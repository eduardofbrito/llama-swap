package server

import (
	"net/http"
	"testing"
)

// TestServer_LoadProbeSucceededAcceptsUpstreamsWithoutRoot pins that a model
// whose server has nothing at / is not reported as a failed load. vLLM is the
// case that prompted this: its preload was logged as "status 404" immediately
// after its own health check passed.
func TestServer_LoadProbeSucceededAcceptsUpstreamsWithoutRoot(t *testing.T) {
	for _, tc := range []struct {
		status int
		want   bool
		why    string
	}{
		{http.StatusOK, true, "llama-server serves a UI at /"},
		{http.StatusNotFound, true, "vLLM has no route at /"},
		{http.StatusMethodNotAllowed, true, "upstream answered, GET not allowed at /"},
		{http.StatusServiceUnavailable, false, "the router refused or the load failed"},
		{http.StatusInternalServerError, false, "the upstream broke"},
		{http.StatusBadGateway, false, "nothing answered"},
	} {
		if got := loadProbeSucceeded(tc.status); got != tc.want {
			t.Errorf("loadProbeSucceeded(%d) = %v, want %v (%s)", tc.status, got, tc.want, tc.why)
		}
	}
}
