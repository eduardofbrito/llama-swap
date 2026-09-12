package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeTestConfig(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatalf("writing config: %v", err)
	}
	return p
}

func TestConfigFileEdit_AddModel(t *testing.T) {
	p := writeTestConfig(t, `models:
  alpha:
    proxy: http://127.0.0.1:11435
`)
	err := AddModelYAML(p, "beta", "proxy: http://127.0.0.1:11436\nname: Beta\n")
	if err != nil {
		t.Fatalf("AddModelYAML: %v", err)
	}
	c, err := LoadConfig(p)
	if err != nil {
		t.Fatalf("LoadConfig after add: %v", err)
	}
	if _, ok := c.Models["beta"]; !ok {
		t.Errorf("beta not present in loaded config: %+v", c.Models)
	}
	if c.Models["beta"].Name != "Beta" {
		t.Errorf("beta name = %q, want Beta", c.Models["beta"].Name)
	}
	// alpha must survive with its value intact.
	if c.Models["alpha"].Proxy != "http://127.0.0.1:11435" {
		t.Errorf("alpha proxy = %q, want unchanged", c.Models["alpha"].Proxy)
	}
}

func TestConfigFileEdit_AddModelRejectsDuplicate(t *testing.T) {
	p := writeTestConfig(t, "models:\n  alpha:\n    proxy: http://127.0.0.1:11435\n")
	if err := AddModelYAML(p, "alpha", "proxy: http://127.0.0.1:9999\n"); err == nil {
		t.Fatal("AddModelYAML for an existing id should fail")
	}
	// The file must be untouched after a failed add.
	data, _ := os.ReadFile(p)
	if strings.Contains(string(data), "9999") {
		t.Error("failed add wrote to the file")
	}
}

func TestConfigFileEdit_ReplaceModel(t *testing.T) {
	p := writeTestConfig(t, "models:\n  alpha:\n    proxy: http://127.0.0.1:11435\n")
	err := ReplaceModelYAML(p, "alpha", "proxy: http://127.0.0.1:22435\nname: Alpha2\n")
	if err != nil {
		t.Fatalf("ReplaceModelYAML: %v", err)
	}
	c, err := LoadConfig(p)
	if err != nil {
		t.Fatalf("LoadConfig after replace: %v", err)
	}
	if c.Models["alpha"].Proxy != "http://127.0.0.1:22435" {
		t.Errorf("alpha proxy = %q, want 22435", c.Models["alpha"].Proxy)
	}
	if c.Models["alpha"].Name != "Alpha2" {
		t.Errorf("alpha name = %q, want Alpha2", c.Models["alpha"].Name)
	}
}

func TestConfigFileEdit_ReplaceMissingFails(t *testing.T) {
	p := writeTestConfig(t, "models:\n  alpha:\n    proxy: http://127.0.0.1:11435\n")
	if err := ReplaceModelYAML(p, "nope", "proxy: http://127.0.0.1:1\n"); err == nil {
		t.Fatal("replace of missing model should fail")
	}
}

func TestModelYAMLText_ReturnsBlock(t *testing.T) {
	p := writeTestConfig(t, "models:\n  alpha:\n    proxy: http://127.0.0.1:11435\n    name: Alpha\n")
	text, found, err := ModelYAMLText(p, "alpha")
	if err != nil {
		t.Fatalf("ModelYAMLText: %v", err)
	}
	if !found {
		t.Fatal("expected found=true for alpha")
	}
	if !strings.Contains(text, "11435") || !strings.Contains(text, "Alpha") {
		t.Errorf("block text missing content: %q", text)
	}
}

func TestModelYAMLText_NotFound(t *testing.T) {
	p := writeTestConfig(t, "models:\n  alpha:\n    proxy: http://127.0.0.1:11435\n")
	_, found, err := ModelYAMLText(p, "ghost")
	if err != nil {
		t.Fatalf("ModelYAMLText: %v", err)
	}
	if found {
		t.Error("expected found=false for ghost")
	}
}

func TestValidateModelYAML(t *testing.T) {
	if err := ValidateModelYAML("proxy: http://127.0.0.1:1\nname: ok\n"); err != nil {
		t.Errorf("valid model rejected: %v", err)
	}
	if err := ValidateModelYAML("not: [a list\n"); err == nil {
		t.Error("invalid YAML accepted")
	}
	if err := ValidateModelYAML("- just\n- a\n- list\n"); err == nil {
		t.Error("non-mapping block accepted")
	}
	if err := ValidateModelYAML(""); err == nil {
		t.Error("empty block accepted")
	}
}

func TestConfigFileEdit_PreservesOtherKeys(t *testing.T) {
	p := writeTestConfig(t, `healthCheckTimeout: 30
models:
  alpha:
    proxy: http://127.0.0.1:11435
`)
	if err := AddModelYAML(p, "beta", "proxy: http://127.0.0.1:11436\n"); err != nil {
		t.Fatalf("AddModelYAML: %v", err)
	}
	c, err := LoadConfig(p)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if c.HealthCheckTimeout != 30 {
		t.Errorf("top-level healthCheckTimeout = %d, want 30", c.HealthCheckTimeout)
	}
}

// TestConfigFileEdit_ReadDoesNotWriteFile pins the invariant that reading a
// model's block is a pure read. A rewrite here would reformat the operator's
// file, bump its mtime/size (waking the -watch-config watcher into a full
// reload that drops every running model), and fail outright on a read-only
// bind mount.
func TestConfigFileEdit_ReadDoesNotWriteFile(t *testing.T) {
	const original = `# top comment
models:
  alpha:
    # keep this comment
    proxy: http://127.0.0.1:11435
`
	p := writeTestConfig(t, original)
	before, err := os.Stat(p)
	if err != nil {
		t.Fatalf("stat before: %v", err)
	}

	if _, _, err := ModelYAMLText(p, "alpha"); err != nil {
		t.Fatalf("ModelYAMLText: %v", err)
	}
	// A model that is not in the file must not write either.
	if _, _, err := ModelYAMLText(p, "ghost"); err != nil {
		t.Fatalf("ModelYAMLText(ghost): %v", err)
	}

	data, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("reading config: %v", err)
	}
	if string(data) != original {
		t.Errorf("reading rewrote the config file:\ngot:\n%s\nwant:\n%s", data, original)
	}
	after, err := os.Stat(p)
	if err != nil {
		t.Fatalf("stat after: %v", err)
	}
	if !before.ModTime().Equal(after.ModTime()) || before.Size() != after.Size() {
		t.Errorf("reading changed file metadata (mtime %v -> %v, size %d -> %d); the config watcher would fire",
			before.ModTime(), after.ModTime(), before.Size(), after.Size())
	}
}

// TestConfigFileEdit_PreservesFileMode checks that a write keeps the config
// file's permission bits: a config restricted to 0600 (it can hold API keys)
// must not come back world-readable.
func TestConfigFileEdit_PreservesFileMode(t *testing.T) {
	p := writeTestConfig(t, "models:\n  alpha:\n    proxy: http://127.0.0.1:11435\n")
	if err := os.Chmod(p, 0o600); err != nil {
		t.Fatalf("chmod: %v", err)
	}
	if err := ReplaceModelYAML(p, "alpha", "proxy: http://127.0.0.1:22435\n"); err != nil {
		t.Fatalf("ReplaceModelYAML: %v", err)
	}
	info, err := os.Stat(p)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if got := info.Mode().Perm(); got != 0o600 {
		t.Errorf("file mode = %v, want 0600 (permissions must survive an edit)", got)
	}
}
