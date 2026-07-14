package process

import (
	"slices"
	"testing"

	"github.com/mostlygeek/llama-swap/internal/config"
)

func TestProcessCommand_StartEnvAppliesRuntimeOverride(t *testing.T) {
	tests := []struct {
		name     string
		confEnv  []string
		override []string
		want     []string
	}{
		{
			name:    "no override keeps the config env",
			confEnv: []string{"CUDA_VISIBLE_DEVICES=0", "FOO=bar"},
			want:    []string{"CUDA_VISIBLE_DEVICES=0", "FOO=bar"},
		},
		{
			name:     "override replaces the variable the config also sets",
			confEnv:  []string{"CUDA_VISIBLE_DEVICES=0", "FOO=bar"},
			override: []string{"CUDA_VISIBLE_DEVICES=2"},
			want:     []string{"FOO=bar", "CUDA_VISIBLE_DEVICES=2"},
		},
		{
			name:     "override adds a variable the config does not set",
			confEnv:  []string{"FOO=bar"},
			override: []string{"CUDA_VISIBLE_DEVICES=1"},
			want:     []string{"FOO=bar", "CUDA_VISIBLE_DEVICES=1"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := &ProcessCommand{config: config.ModelConfig{Env: tc.confEnv}}
			if tc.override != nil {
				p.SetRuntimeEnv(tc.override)
			}
			if got := p.startEnv(); !slices.Equal(got, tc.want) {
				t.Errorf("startEnv() = %v, want %v", got, tc.want)
			}
		})
	}
}
