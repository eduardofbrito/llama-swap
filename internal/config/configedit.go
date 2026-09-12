package config

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"syscall"

	"gopkg.in/yaml.v3"
)

// isReadOnlyErr reports whether err comes from a read-only filesystem or a
// read-only mount (the case when the config file is bind-mounted :ro).
func isReadOnlyErr(err error) bool {
	return errors.Is(err, syscall.EROFS)
}

// editMu serializes edits to the config file so concurrent UI saves cannot
// interleave read-modify-write cycles.
var editMu sync.Mutex

// configFileRead parses the config file at path and hands the document to fn
// for inspection. It never writes: a read must not reformat the operator's
// file, bump its mtime (which would wake the -watch-config watcher and force a
// full reload), or fail on a read-only bind mount. fn must treat the document
// as immutable; use ConfigFileEdit to change it.
func configFileRead(path string, fn func(doc *yaml.Node) error) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("reading config file: %w", err)
	}
	var doc yaml.Node
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return fmt.Errorf("parsing config file: %w", err)
	}
	if fn == nil {
		return nil
	}
	return fn(&doc)
}

// fileMode returns the file's current permission bits so a rewrite preserves
// them. It falls back to 0o644 when the file cannot be stat'ed.
func fileMode(path string) os.FileMode {
	info, err := os.Stat(path)
	if err != nil {
		return 0o644
	}
	return info.Mode().Perm()
}

// ConfigFileEdit applies fn to the parsed document of the config file at
// path and writes the result back atomically. fn may restructure the
// document; the modified document is validated with the regular load
// pipeline before the file is touched, so a bad edit never corrupts the
// config on disk. Comments on untouched nodes survive the round trip.
func ConfigFileEdit(path string, fn func(doc *yaml.Node) error) error {
	editMu.Lock()
	defer editMu.Unlock()

	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("reading config file: %w", err)
	}
	var doc yaml.Node
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return fmt.Errorf("parsing config file: %w", err)
	}
	if fn != nil {
		if err := fn(&doc); err != nil {
			return err
		}
	}
	out, err := yaml.Marshal(&doc)
	if err != nil {
		return fmt.Errorf("marshaling config: %w", err)
	}
	if _, err := LoadConfigFromReader(bytes.NewReader(out)); err != nil {
		return fmt.Errorf("resulting config is invalid: %w", err)
	}

	// The replacement inherits the original file's permission bits: a config
	// deliberately restricted to 0600 (it can hold API keys) must not come
	// back world-readable.
	mode := fileMode(path)
	tmp := fmt.Sprintf("%s.tmp-%d", path, os.Getpid())
	if err := os.WriteFile(tmp, out, mode); err != nil {
		if isReadOnlyErr(err) {
			return fmt.Errorf("config file is mounted read-only; remount it writable (drop :ro in the volume) to edit from the UI: %w", err)
		}
		return fmt.Errorf("writing config file: %w", err)
	}
	// WriteFile only applies mode when it creates the file; a leftover temp
	// file from an earlier crash would keep its old bits, so set them here.
	if err := os.Chmod(tmp, mode); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("setting config file permissions: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		// Renaming over a bind mount (e.g. Docker -v file:/app/config.yaml)
		// fails with EBUSY: a mount point cannot be replaced by rename.
		// Fall back to an in-place write: truncate + write into the same
		// inode, which is the only way to update a bind-mounted file. Not
		// atomic, but the content already passed full validation above.
		inErr := writeInPlace(path, out)
		os.Remove(tmp)
		if inErr == nil {
			return nil
		}
		if isReadOnlyErr(inErr) {
			return fmt.Errorf("config file is mounted read-only; remount it writable (drop :ro in the volume) to edit from the UI: %w", inErr)
		}
		return fmt.Errorf("replacing config file: %w (in-place write also failed: %v)", err, inErr)
	}
	return nil
}

// writeInPlace rewrites the existing file in place (same inode): truncate +
// write. The only way to update a bind-mounted file, where rename over the
// mount point is impossible (EBUSY).
func writeInPlace(path string, data []byte) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_TRUNC, 0)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// ModelYAMLText returns the YAML text of one model's block as written in the
// config file. found is false when the file has no block for the model ID.
// It is a pure read: the config file on disk is never touched, so opening the
// UI's Conf tab cannot reformat the file, trigger the config watcher, or fail
// against a read-only bind mount.
func ModelYAMLText(path, modelID string) (string, bool, error) {
	var text string
	var found bool
	err := configFileRead(path, func(doc *yaml.Node) error {
		root, err := rootMapping(doc)
		if err != nil {
			return err
		}
		value, ok := findModelNode(root, modelID)
		if !ok {
			return nil
		}
		found = true
		out, err := yaml.Marshal(value)
		if err != nil {
			return fmt.Errorf("marshaling model block: %w", err)
		}
		text = string(out)
		return nil
	})
	return text, found, err
}

// ReplaceModelYAML replaces one model's block in the config file. The model
// must already exist; the new YAML must parse to a mapping and the resulting
// document must pass the full config load pipeline.
func ReplaceModelYAML(path, modelID, modelYAML string) error {
	return ConfigFileEdit(path, func(doc *yaml.Node) error {
		root, err := rootMapping(doc)
		if err != nil {
			return err
		}
		models, err := modelsMapping(root, false)
		if err != nil {
			return err
		}
		value, ok := findModelNode(root, modelID)
		if !ok {
			return fmt.Errorf("model %q not found in config file", modelID)
		}
		node, err := parseModelBlock(modelYAML)
		if err != nil {
			return err
		}
		replaceNode(models, modelID, value, node)
		return nil
	})
}

// AddModelYAML appends a new model's block to the config file. The model ID
// must not already exist in the file.
func AddModelYAML(path, modelID, modelYAML string) error {
	return ConfigFileEdit(path, func(doc *yaml.Node) error {
		root, err := rootMapping(doc)
		if err != nil {
			return err
		}
		models, err := modelsMapping(root, true)
		if err != nil {
			return err
		}
		if _, ok := findModelNode(root, modelID); ok {
			return fmt.Errorf("model %q already exists in config file", modelID)
		}
		node, err := parseModelBlock(modelYAML)
		if err != nil {
			return err
		}
		models.Content = append(models.Content,
			&yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: modelID},
			node,
		)
		return nil
	})
}

// ValidateModelYAML is a cheap, file-free precheck used by the UI before an
// edit: the block must parse as a single YAML mapping that fits ModelConfig.
// The authoritative validation still runs in ConfigFileEdit, which loads the
// whole resulting document through the normal pipeline before writing.
func ValidateModelYAML(modelYAML string) error {
	// parseModelBlock enforces the shape (a single YAML mapping); the node
	// itself is not needed here, only the error it reports.
	if _, err := parseModelBlock(modelYAML); err != nil {
		return err
	}
	var mc ModelConfig
	if err := yaml.Unmarshal([]byte(strings.TrimSpace(modelYAML)), &mc); err != nil {
		return fmt.Errorf("model block does not map to a model config: %w", err)
	}
	return nil
}

// parseModelBlock parses a model's YAML block; it must be a single mapping.
func parseModelBlock(modelYAML string) (*yaml.Node, error) {
	trimmed := strings.TrimSpace(modelYAML)
	if trimmed == "" {
		return nil, fmt.Errorf("model config is empty")
	}
	var doc yaml.Node
	if err := yaml.Unmarshal([]byte(trimmed), &doc); err != nil {
		return nil, fmt.Errorf("model block is not valid YAML: %w", err)
	}
	if doc.Kind != yaml.DocumentNode || len(doc.Content) == 0 {
		return nil, fmt.Errorf("model block is not a YAML document")
	}
	value := doc.Content[0]
	if value.Kind != yaml.MappingNode {
		return nil, fmt.Errorf("model config must be a YAML mapping (key: value pairs)")
	}
	return value, nil
}

// rootMapping returns the document's root mapping node.
func rootMapping(doc *yaml.Node) (*yaml.Node, error) {
	if doc.Kind != yaml.DocumentNode || len(doc.Content) == 0 {
		return nil, fmt.Errorf("config file is not a YAML document")
	}
	if doc.Content[0].Kind != yaml.MappingNode {
		return nil, fmt.Errorf("config root is not a mapping")
	}
	return doc.Content[0], nil
}

// findModelNode locates the model's value node under the models mapping.
func findModelNode(root *yaml.Node, modelID string) (*yaml.Node, bool) {
	models, err := modelsMapping(root, false)
	if err != nil {
		return nil, false
	}
	for i := 0; i+1 < len(models.Content); i += 2 {
		if models.Content[i].Value == modelID {
			return models.Content[i+1], true
		}
	}
	return nil, false
}

// modelsMapping returns the models mapping under the root, optionally
// creating it when createMissing is true.
func modelsMapping(root *yaml.Node, createMissing bool) (*yaml.Node, error) {
	value, ok := findMapKey(root, "models")
	if !ok {
		if !createMissing {
			return nil, fmt.Errorf("config file has no models section")
		}
		created := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
		root.Content = append(root.Content,
			&yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: "models"},
			created,
		)
		return created, nil
	}
	if value.Kind != yaml.MappingNode {
		return nil, fmt.Errorf("models section is not a mapping")
	}
	return value, nil
}

// findMapKey finds a key's value node in a mapping node.
func findMapKey(mapping *yaml.Node, key string) (*yaml.Node, bool) {
	for i := 0; i+1 < len(mapping.Content); i += 2 {
		if mapping.Content[i].Value == key {
			return mapping.Content[i+1], true
		}
	}
	return nil, false
}

// replaceNode swaps out an existing model value node inside the models
// mapping (value must be the node currently paired with modelID).
func replaceNode(models *yaml.Node, modelID string, old, new *yaml.Node) {
	for i := 0; i+1 < len(models.Content); i += 2 {
		if models.Content[i].Value == modelID && models.Content[i+1] == old {
			models.Content[i+1] = new
			return
		}
	}
	// Fall through: drop the old pair and append the new one (keeps the
	// invariant that each key appears once).
	out := make([]*yaml.Node, 0, len(models.Content)+2)
	for i := 0; i+1 < len(models.Content); i += 2 {
		if models.Content[i].Value == modelID {
			continue
		}
		out = append(out, models.Content[i], models.Content[i+1])
	}
	out = append(out,
		&yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: modelID},
		new,
	)
	models.Content = out
}
