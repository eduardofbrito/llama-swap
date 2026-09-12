package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

// Config editing endpoints. They mutate the -config file on disk through the
// config package's node surgery (comments preserved, atomic replace) and the
// result is validated with the full load pipeline before the file is
// written. The UI's model detail "Conf" tab and the add-model dialog consume
// these; both run behind the API auth chain.

func (s *Server) requireConfigFile(w http.ResponseWriter, r *http.Request) bool {
	if s.configPath == "" {
		swaputil.SendResponse(w, r, http.StatusNotImplemented,
			"config editing requires a single -config file (this instance was not started with one)")
		return false
	}
	return true
}

// handleAPIGetModelConfig serves one model's block from the config file.
// GET /api/config/model/{model...}
func (s *Server) handleAPIGetModelConfig(w http.ResponseWriter, r *http.Request) {
	if !s.requireConfigFile(w, r) {
		return
	}
	modelID := r.PathValue("model")
	text, found, err := config.ModelYAMLText(s.configPath, modelID)
	if err != nil {
		swaputil.SendResponse(w, r, http.StatusInternalServerError, err.Error())
		return
	}
	if !found {
		swaputil.SendResponse(w, r, http.StatusNotFound, fmt.Sprintf("model %q not found in config file", modelID))
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"model": modelID, "yaml": text})
}

// modelConfigBody is the request body for replacing/adding a model's block.
type modelConfigBody struct {
	YAML string `json:"yaml"`
}

// handleAPIPutModelConfig replaces one model's block in the config file.
// PUT /api/config/model/{model...}
func (s *Server) handleAPIPutModelConfig(w http.ResponseWriter, r *http.Request) {
	if !s.requireConfigFile(w, r) {
		return
	}
	modelID := r.PathValue("model")
	var body modelConfigBody
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		swaputil.SendResponse(w, r, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	if err := config.ReplaceModelYAML(s.configPath, modelID, body.YAML); err != nil {
		swaputil.SendResponse(w, r, http.StatusUnprocessableEntity, err.Error())
		return
	}
	s.proxylog.Infof("config: model %s updated from UI", modelID)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"msg": "ok", "model": modelID})
}

// addModelBody is the request body for adding a new model.
type addModelBody struct {
	ID   string `json:"id"`
	YAML string `json:"yaml"`
}

// handleAPIAddModel appends a new model's block to the config file.
// POST /api/config/model
func (s *Server) handleAPIAddModel(w http.ResponseWriter, r *http.Request) {
	if !s.requireConfigFile(w, r) {
		return
	}
	var body addModelBody
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		swaputil.SendResponse(w, r, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	id := strings.TrimSpace(body.ID)
	if id == "" {
		swaputil.SendResponse(w, r, http.StatusBadRequest, "model id is required")
		return
	}
	if err := config.AddModelYAML(s.configPath, id, body.YAML); err != nil {
		swaputil.SendResponse(w, r, http.StatusUnprocessableEntity, err.Error())
		return
	}
	s.proxylog.Infof("config: model %s added from UI", id)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"msg": "ok", "model": id})
}

// handleAPIReloadConfig triggers a hot reload so UI edits take effect
// without restarting the process.
//
// POST /api/config/reload          full reload (rebuilds the server; drops
//
//	every running model)
//
// POST /api/config/reload?model=m  surgical reload: only model m's config and
//
//	process are rebuilt, every other model
//	keeps serving. Falls back to a full
//	reload when the diff is not model-only
//	(group/matrix/selector/profile/global
//	changes) or the surgical refresh fails.
func (s *Server) handleAPIReloadConfig(w http.ResponseWriter, r *http.Request) {
	if !s.requireConfigFile(w, r) {
		return
	}
	model := r.URL.Query().Get("model")
	if model == "" {
		if s.reloadFn == nil {
			swaputil.SendResponse(w, r, http.StatusNotImplemented, "config reload is not wired on this instance")
			return
		}
		go s.reloadFn()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"msg": "reload triggered"})
		return
	}
	if s.reloadFn == nil {
		swaputil.SendResponse(w, r, http.StatusNotImplemented, "config reload is not wired on this instance")
		return
	}
	if err := s.SurgicalReload(model); err != nil {
		// Not a model-only edit (selectors/profiles/global) or the refresh
		// failed: fall back to the full reload so the edited config still
		// takes effect. The file on disk was already updated by the PUT.
		s.proxylog.Warnf("config: surgical reload of %s not possible (%v); falling back to full reload", model, err)
		go s.reloadFn()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"msg": "reload triggered", "model": model, "scope": "full"})
		return
	}
	s.proxylog.Infof("config: model %s reloaded surgically (other models untouched)", model)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"msg": "reloaded", "model": model, "scope": "model"})
}
