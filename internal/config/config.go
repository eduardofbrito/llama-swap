package config

import (
	"fmt"
	"os"
	"sort"

	"gopkg.in/yaml.v3"
)

const DEFAULT_GROUP_ID = "(default)"
const DEFAULT_UNLOAD_TIMEOUT = 10
const (
	LogToStdoutProxy    = "proxy"
	LogToStdoutUpstream = "upstream"
	LogToStdoutBoth     = "both"
	LogToStdoutNone     = "none"
)

type MacroEntry struct {
	Name  string
	Value any
}

type MacroList []MacroEntry

// UnmarshalYAML implements custom YAML unmarshaling that preserves macro definition order
func (ml *MacroList) UnmarshalYAML(value *yaml.Node) error {
	if value.Kind != yaml.MappingNode {
		return fmt.Errorf("macros must be a mapping")
	}

	// yaml.Node.Content for a mapping contains alternating key/value nodes
	entries := make([]MacroEntry, 0, len(value.Content)/2)
	for i := 0; i < len(value.Content); i += 2 {
		keyNode := value.Content[i]
		valueNode := value.Content[i+1]

		var name string
		if err := keyNode.Decode(&name); err != nil {
			return fmt.Errorf("failed to decode macro name: %w", err)
		}

		var val any
		if err := valueNode.Decode(&val); err != nil {
			return fmt.Errorf("failed to decode macro value for '%s': %w", name, err)
		}

		entries = append(entries, MacroEntry{Name: name, Value: val})
	}

	*ml = entries
	return nil
}

// MarshalYAML renders the list back as an ordered mapping, the shape it is
// written in, rather than the default slice-of-structs. Only diagnostics such
// as the config__get_config tool marshal a Config, but when they do the macro
// block should read like the source file.
func (ml MacroList) MarshalYAML() (interface{}, error) {
	node := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
	for _, entry := range ml {
		var key, value yaml.Node
		if err := key.Encode(entry.Name); err != nil {
			return nil, fmt.Errorf("encoding macro name %q: %w", entry.Name, err)
		}
		if err := value.Encode(entry.Value); err != nil {
			return nil, fmt.Errorf("encoding macro value for %q: %w", entry.Name, err)
		}
		node.Content = append(node.Content, &key, &value)
	}
	return node, nil
}

// Get retrieves a macro value by name
func (ml MacroList) Get(name string) (any, bool) {
	for _, entry := range ml {
		if entry.Name == name {
			return entry.Value, true
		}
	}
	return nil, false
}

type GroupConfig struct {
	Swap       bool     `yaml:"swap"`
	Exclusive  bool     `yaml:"exclusive"`
	Persistent bool     `yaml:"persistent"`
	Members    []string `yaml:"members"`
}

// set default values for GroupConfig
func (c *GroupConfig) UnmarshalYAML(unmarshal func(interface{}) error) error {
	type rawGroupConfig GroupConfig
	defaults := rawGroupConfig{
		Swap:       true,
		Exclusive:  true,
		Persistent: false,
		Members:    []string{},
	}

	if err := unmarshal(&defaults); err != nil {
		return err
	}

	*c = GroupConfig(defaults)
	return nil
}

type HooksConfig struct {
	OnStartup HookOnStartup `yaml:"on_startup"`
}

type HookOnStartup struct {
	Preload []string `yaml:"preload"`
	Profile string   `yaml:"profile"`
}

type Store struct {
	Path string `yaml:"path"`
}

type UIConfig struct {
	Activity UIActivityConfig `yaml:"activity" json:"activity"`
}

type UIActivityConfig struct {
	SessionID []string `yaml:"session_id" json:"session_id"`
}

// ProfileConfig describes a runtime-selectable set of model ID rewrites.
// Empty pin targets disable the corresponding model ID while the profile is
// active. YAML null values decode to the same empty string representation.
type ProfileConfig struct {
	Description string            `yaml:"description" json:"description"`
	Pins        map[string]string `yaml:"pins" json:"pins"`
}

// UnmarshalYAML rejects the removed list-shaped profile syntax with a useful
// migration error while allowing null pin values to normalize to empty strings.
func (c *ProfileConfig) UnmarshalYAML(value *yaml.Node) error {
	if value.Kind != yaml.MappingNode {
		return fmt.Errorf("profile must be a mapping with description and pins; the legacy list syntax is no longer supported")
	}
	type rawProfileConfig ProfileConfig
	var raw rawProfileConfig
	if err := value.Decode(&raw); err != nil {
		return err
	}
	*c = ProfileConfig(raw)
	return nil
}

type Config struct {
	Tailcat            *TailcatConfig    `yaml:"tailcat"`
	HealthCheckTimeout int               `yaml:"healthCheckTimeout"`
	LogRequests        bool              `yaml:"logRequests"`
	LogLevel           string            `yaml:"logLevel"`
	LogTimeFormat      string            `yaml:"logTimeFormat"`
	LogToStdout        string            `yaml:"logToStdout"`
	MetricsMaxInMemory int               `yaml:"metricsMaxInMemory"`
	CaptureBuffer      int               `yaml:"captureBuffer"`
	Store              *Store            `yaml:"store"`
	UI                 UIConfig          `yaml:"ui"`
	Performance        PerformanceConfig `yaml:"performance"`
	GlobalTTL          int               `yaml:"globalTTL"`
	UnloadTimeout      int               `yaml:"unloadTimeout"`

	Models    map[string]ModelConfig    `yaml:"models"` /* key is model ID */
	Profiles  map[string]ProfileConfig  `yaml:"profiles"`
	Selectors map[string]SelectorConfig `yaml:"selectors"`

	// GlobalConcurrencyLimit caps the number of inference requests served at
	// once across all models. 0 (default) means no limit. See issue #1086.
	GlobalConcurrencyLimit int `yaml:"globalConcurrencyLimit"`

	// routing is the canonical source for swap/scheduling configuration.
	// New code must read Routing, never the backwards-compat fields below.
	Routing RoutingConfig `yaml:"routing"`

	// Groups and Matrix are permanent backwards-compat input fields for the
	// legacy top-level `groups:`/`matrix:` keys. They are normalized into
	// Routing by LoadConfigFromReader. New code must not read them directly.
	Groups map[string]GroupConfig `yaml:"groups"` /* key is group ID */
	Matrix *MatrixConfig          `yaml:"matrix"`

	// for key/value replacements in model's cmd, cmdStop, proxy, checkEndPoint
	Macros MacroList `yaml:"macros"`

	// map aliases to actual model IDs
	aliases map[string]string

	// automatic port assignments
	StartPort int `yaml:"startPort"`

	// hooks, see: #209
	Hooks HooksConfig `yaml:"hooks"`

	// send loading state in reasoning
	SendLoadingState bool `yaml:"sendLoadingState"`

	// present aliases to /v1/models OpenAI API listing
	IncludeAliasesInList bool `yaml:"includeAliasesInList"`

	// support API keys, see issue #433, #50, #251
	RequiredAPIKeys []string `yaml:"apiKeys"`

	// support remote peers, see issue #433, #296
	Peers PeerDictionaryConfig `yaml:"peers"`

	// upstream controls behaviour of the /upstream passthrough endpoint
	Upstream UpstreamConfig `yaml:"upstream"`

	// tailcatEnabled records whether this process started a Tailcat listener.
	// It is runtime state, not user configuration, so it must never appear in
	// rendered configuration output.
	tailcatEnabled bool
}

// SetTailcatEnabled records whether this process has a Tailcat listener.
// main owns this startup-only setting from -listen-tailcat.
func (c *Config) SetTailcatEnabled(enabled bool) {
	c.tailcatEnabled = enabled
}

// TailcatEnabled reports whether this process has a Tailcat listener.
func (c Config) TailcatEnabled() bool {
	return c.tailcatEnabled
}

// RoutingConfig is the canonical, normalized routing/scheduling configuration.
type RoutingConfig struct {
	Scheduler SchedulerConfig `yaml:"scheduler"`
	Router    RouterConfig    `yaml:"router"`
}

type SchedulerConfig struct {
	Use      string            `yaml:"use"` // default "fifo"
	Settings SchedulerSettings `yaml:"settings"`
}

type SchedulerSettings struct {
	Fifo FifoConfig `yaml:"fifo"`
}

type FifoConfig struct {
	Priority map[string]int `yaml:"priority"` // model ID -> priority, default 0

	// RecentPoolSize keeps the N most recently used models loaded instead of
	// unloading them: when the router's swapper asks for a model to be
	// evicted, the scheduler holds it back while the model is among the N most
	// recent, and only lets the least recently used one go once the pool is
	// full. The pool can only hold back evictions the swapper asked for —
	// models it keeps resident anyway (a persistent group) are never affected
	// — so the hardware must have room for N models at once. 0 or 1 (the
	// default) leaves every eviction decision to the swapper.
	RecentPoolSize int `yaml:"recentPoolSize"`

	// VramCheck refuses to start a model when the GPU it would load onto does
	// not have enough free memory for it, instead of letting the load fail
	// deep inside the runtime. It exists for GPUs that llama-swap does not
	// have to itself: another process (a training job, a second llama-swap, a
	// desktop session) can hold memory that the router knows nothing about, so
	// "every model I evicted is stopped" is not the same as "the memory is
	// free". Off by default.
	//
	// A model's requirement is models.*.vramMB when declared, otherwise the
	// value measured on its last successful load. A model with neither is
	// always admitted: refusing on ignorance would be worse than today's
	// behaviour.
	VramCheck bool `yaml:"vramCheck"`

	// VramMarginPct is the headroom kept free on top of a model's requirement,
	// as a percentage of it. Runtimes allocate more than the weights (KV
	// cache growth, fragmentation, CUDA context), and a measurement taken on
	// one run is not a ceiling for the next. Defaults to
	// DefaultVramMarginPct when VramCheck is on.
	VramMarginPct int `yaml:"vramMarginPct"`

	// EvictBeyondPoolOnPressure lets a load that does not fit fall back to
	// evicting models the recent-model pool was holding back, instead of being
	// refused. It trades the pool's retention for availability: with it off
	// (the default) the pool is honoured and the request gets a 503, which
	// keeps capacity honest and predictable.
	EvictBeyondPoolOnPressure bool `yaml:"evictBeyondPoolOnPressure"`
}

// DefaultVramMarginPct is the headroom kept on top of a model's VRAM
// requirement when routing.scheduler.settings.fifo.vramMarginPct is unset.
const DefaultVramMarginPct = 5

// VramMargin returns the configured headroom percentage, or the default when
// unset. A negative value disables the margin entirely.
func (c FifoConfig) VramMargin() int {
	if c.VramMarginPct == 0 {
		return DefaultVramMarginPct
	}
	if c.VramMarginPct < 0 {
		return 0
	}
	return c.VramMarginPct
}

type RouterConfig struct {
	Use      string         `yaml:"use"` // "group" (default) | "matrix"
	Settings RouterSettings `yaml:"settings"`
}

type RouterSettings struct {
	Groups map[string]GroupConfig `yaml:"groups"`
	Matrix *MatrixConfig          `yaml:"matrix"`
}

func (c *Config) RealModelName(search string) (string, bool) {
	if _, found := c.Models[search]; found {
		return search, true
	} else if name, found := c.aliases[search]; found {
		return name, found
	} else {
		return "", false
	}
}

func (c *Config) FindConfig(modelName string) (ModelConfig, string, bool) {
	if realName, found := c.RealModelName(modelName); !found {
		return ModelConfig{}, "", false
	} else {
		return c.Models[realName], realName, true
	}
}

// ResolveBaseModel resolves a name without applying profiles. Local model IDs
// and aliases take precedence over peer model IDs, matching server dispatch.
func (c *Config) ResolveBaseModel(search string) (string, bool) {
	if realName, found := c.RealModelName(search); found {
		return realName, true
	}
	if _, _, found := c.ResolvePeerModel(search); found {
		return search, true
	}
	return "", false
}

func LoadConfig(path string) (Config, error) {
	file, err := os.Open(path)
	if err != nil {
		return Config{}, err
	}
	defer file.Close()
	return LoadConfigFromReader(file)
}

// rewrites the yaml to include a default group with any orphaned models
func AddDefaultGroupToConfig(config Config) Config {

	if config.Groups == nil {
		config.Groups = make(map[string]GroupConfig)
	}

	defaultGroup := GroupConfig{
		Swap:      true,
		Exclusive: true,
		Members:   []string{},
	}
	// if groups is empty, create a default group and put
	// all models into it
	if len(config.Groups) == 0 {
		for modelName := range config.Models {
			defaultGroup.Members = append(defaultGroup.Members, modelName)
		}
	} else {
		// iterate over existing group members and add non-grouped models into the default group
		for modelName := range config.Models {
			foundModel := false
		found:
			// search for the model in existing groups
			for _, groupConfig := range config.Groups {
				for _, member := range groupConfig.Members {
					if member == modelName {
						foundModel = true
						break found
					}
				}
			}

			if !foundModel {
				defaultGroup.Members = append(defaultGroup.Members, modelName)
			}
		}
	}

	sort.Strings(defaultGroup.Members) // make consistent ordering for testing
	config.Groups[DEFAULT_GROUP_ID] = defaultGroup

	return config
}
