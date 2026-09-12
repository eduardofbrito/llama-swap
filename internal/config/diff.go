package config

import (
	"reflect"
	"sort"
)

// DiffModels compares two configs produced by the same load pipeline
// (LoadConfigSources) and reports which model blocks differ.
//
// changed lists the model IDs whose ModelConfig differs between the two
// configs. ok is false when anything other than model blocks changed
// (selectors, profiles, peers, groups/matrix, global settings, ...), in which
// case the caller must do a full reload: the routers' planners and the server
// middleware were built from the old structure and a surgical in-place update
// would leave them inconsistent.
//
// The aliases map is excluded from the comparison on purpose: it is derived
// from the model blocks (explicit aliases plus setParamsByID keys), so a model
// edit legitimately changes it while a global edit never touches it.
func DiffModels(oldCfg, newCfg Config) (changed []string, ok bool) {
	if len(oldCfg.Models) != len(newCfg.Models) {
		return nil, false
	}
	for id, om := range oldCfg.Models {
		nm, found := newCfg.Models[id]
		if !found {
			return nil, false
		}
		if !reflect.DeepEqual(om, nm) {
			changed = append(changed, id)
		}
	}

	gOld, gNew := oldCfg, newCfg
	gOld.Models, gNew.Models = nil, nil
	gOld.aliases, gNew.aliases = nil, nil
	// tailcatEnabled is runtime state (set from -listen-tailcat on the live
	// config, never present in the file) — clear it so a fresh load of the
	// same file compares equal.
	gOld.tailcatEnabled, gNew.tailcatEnabled = false, false
	if !reflect.DeepEqual(gOld, gNew) {
		return nil, false
	}

	sort.Strings(changed)
	return changed, true
}
