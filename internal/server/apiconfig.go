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
// these.
//
// Access control matters more here than on the rest of the API: a caller that
// can write a model block can choose that model's `cmd` and then start it with
// a plain GET /upstream/<id>/, which is arbitrary command execution on the
// host. Two gates therefore guard every handler below, on top of the API auth
// chain they are mounted on:
//
//  1. the operator must opt in with -enable-config-api (which also requires a
//     single -config file); without it s.configPath is empty and the endpoints
//     report 501, and
//  2. the config must declare at least one apiKey. The auth middleware is a
//     deliberate pass-through when no keys are configured, so without this
//     check the opt-in alone would publish an unauthenticated write surface.

// configEditingStatus reports whether config editing is available, and why not
// when it is unavailable. The HTTP status is the one a config endpoint answers
// with in that state.
func (s *Server) configEditingStatus() (editable bool, status int, reason string) {
	if s.configPath == "" {
		return false, http.StatusNotImplemented,
			"config editing is disabled; start llama-swap with -enable-config-api and a single -config file to turn it on"
	}
	if len(s.Cfg().RequiredAPIKeys) == 0 {
		return false, http.StatusForbidden,
			"config editing requires authentication; add at least one entry under apiKeys in the config file"
	}
	return true, http.StatusOK, ""
}

// requireConfigEditing reports whether config editing is available for this
// request, writing the appropriate refusal when it is not.
func (s *Server) requireConfigEditing(w http.ResponseWriter, r *http.Request) bool {
	editable, status, reason := s.configEditingStatus()
	if !editable {
		swaputil.SendResponse(w, r, status, reason)
	}
	return editable
}

// handleAPIConfigStatus tells the UI whether the config-editing endpoints are
// usable, so it can hide the Conf tab and the Add Model dialog instead of
// offering controls that can only fail.
// GET /api/config/status
func (s *Server) handleAPIConfigStatus(w http.ResponseWriter, r *http.Request) {
	editable, _, reason := s.configEditingStatus()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"editable": editable, "reason": reason})
}

// handleAPIGetModelConfig serves one model's block from the config file.
// GET /api/config/model/{model...}
func (s *Server) handleAPIGetModelConfig(w http.ResponseWriter, r *http.Request) {
	if !s.requireConfigEditing(w, r) {
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
	if !s.requireConfigEditing(w, r) {
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
	if !s.requireConfigEditing(w, r) {
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
	if !s.requireConfigEditing(w, r) {
		return
	}
	if s.reloadFn == nil {
		swaputil.SendResponse(w, r, http.StatusNotImplemented, "config reload is not wired on this instance")
		return
	}
	model := r.URL.Query().Get("model")
	if model == "" {
		go s.reloadFn()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"msg": "reload triggered"})
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
