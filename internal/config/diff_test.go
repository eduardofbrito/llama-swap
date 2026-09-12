package config

import (
	"strings"
	"testing"
)

const diffTestBase = `
models:
  alpha:
    proxy: http://127.0.0.1:11435
  beta:
    proxy: http://127.0.0.1:11436
`

func TestDiffModels_NoChange(t *testing.T) {
	oldCfg, err := LoadConfigFromReader(strings.NewReader(diffTestBase))
	if err != nil {
		t.Fatal(err)
	}
	newCfg, err := LoadConfigFromReader(strings.NewReader(diffTestBase))
	if err != nil {
		t.Fatal(err)
	}
	changed, ok := DiffModels(oldCfg, newCfg)
	if !ok {
		t.Fatal("ok = false, want true")
	}
	if len(changed) != 0 {
		t.Errorf("changed = %v, want none", changed)
	}
}

func TestDiffModels_SingleModelChange(t *testing.T) {
	oldCfg, err := LoadConfigFromReader(strings.NewReader(diffTestBase))
	if err != nil {
		t.Fatal(err)
	}
	changedYAML := strings.Replace(diffTestBase, "11435", "22435", 1)
	newCfg, err := LoadConfigFromReader(strings.NewReader(changedYAML))
	if err != nil {
		t.Fatal(err)
	}
	changed, ok := DiffModels(oldCfg, newCfg)
	if !ok {
		t.Fatal("ok = false, want true")
	}
	if len(changed) != 1 || changed[0] != "alpha" {
		t.Fatalf("changed = %v, want [alpha]", changed)
	}
}

func TestDiffModels_TailcatRuntimeStateIgnored(t *testing.T) {
	oldCfg, err := LoadConfigFromReader(strings.NewReader(diffTestBase))
	if err != nil {
		t.Fatal(err)
	}
	oldCfg.SetTailcatEnabled(true) // runtime-only flag, never in the file
	newCfg, err := LoadConfigFromReader(strings.NewReader(diffTestBase))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := DiffModels(oldCfg, newCfg); !ok {
		t.Fatal("ok = false; tailcatEnabled must not force a full reload")
	}
}

func TestDiffModels_GlobalChangeNotOK(t *testing.T) {
	oldCfg, err := LoadConfigFromReader(strings.NewReader(diffTestBase))
	if err != nil {
		t.Fatal(err)
	}
	newCfg, err := LoadConfigFromReader(strings.NewReader(diffTestBase + "healthCheckTimeout: 99\n"))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := DiffModels(oldCfg, newCfg); ok {
		t.Fatal("ok = true, want false for a global setting change")
	}
}

func TestDiffModels_ModelAddedNotOK(t *testing.T) {
	oldCfg, err := LoadConfigFromReader(strings.NewReader(diffTestBase))
	if err != nil {
		t.Fatal(err)
	}
	newCfg, err := LoadConfigFromReader(strings.NewReader(diffTestBase + "  gamma:\n    proxy: http://127.0.0.1:1\n"))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := DiffModels(oldCfg, newCfg); ok {
		t.Fatal("ok = true, want false when the model count differs")
	}
}

func TestDiffModels_AliasDriftIgnored(t *testing.T) {
	// Adding an explicit alias is a model-block change (reported as a model
	// diff, not a structural change).
	baseA, err := LoadConfigFromReader(strings.NewReader(diffTestBase))
	if err != nil {
		t.Fatal(err)
	}
	withAliasYAML := `
models:
  alpha:
    proxy: http://127.0.0.1:11435
    aliases: [a1]
  beta:
    proxy: http://127.0.0.1:11436
`
	withAlias, err := LoadConfigFromReader(strings.NewReader(withAliasYAML))
	if err != nil {
		t.Fatal(err)
	}
	changed, ok := DiffModels(baseA, withAlias)
	if !ok {
		t.Fatal("ok = false; adding an alias is a model-block change")
	}
	if len(changed) != 1 || changed[0] != "alpha" {
		t.Fatalf("changed = %v, want [alpha]", changed)
	}
}
