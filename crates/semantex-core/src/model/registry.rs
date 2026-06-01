//! `ModelRegistry` — resolves the active model per role from merged specs.
//!
//! Holds built-in + user-manifest specs (validated at construction), and reads
//! the active id per role from `SemantexConfig` (`embedder` / `reranker_model` /
//! `llm_model`, each overridable by env). Resolution is cheap — the model
//! WEIGHTS stay lazy in the existing embedder/reranker singletons; the registry
//! only hands out the `ModelSpec` that tells those singletons WHAT to load.

use crate::config::SemantexConfig;
use crate::model::capabilities::backend_for;
use crate::model::manifest::{builtin_specs, load_user_manifest, merge, user_manifest_path};
use crate::model::spec::{ModelRole, ModelSpec, RoleData};
use crate::search::dense_backend::DenseBackendKind;
use anyhow::Result;
use std::path::Path;

/// Resolved, validated set of model specs + the active selection from config.
pub struct ModelRegistry {
    specs: Vec<ModelSpec>,
    active_embedder_id: String,
    active_reranker_id: String,
    active_llm_id: String,
}

impl ModelRegistry {
    /// Build from `config` + an optional `project_path` (for a project-local
    /// `models.toml`). Merges built-ins with the user manifest (user overrides by
    /// id), validates every spec, and records the active id per role from config.
    pub fn from_config(config: &SemantexConfig, project_path: Option<&Path>) -> Result<Self> {
        let user = match user_manifest_path(config, project_path) {
            Some(path) => load_user_manifest(&path)?,
            None => Vec::new(),
        };
        let specs = merge(builtin_specs(), user);
        for s in &specs {
            s.validate()?;
        }
        Ok(Self {
            specs,
            active_embedder_id: config.embedder.clone(),
            active_reranker_id: config.reranker_model.clone(),
            active_llm_id: config.llm_model.clone(),
        })
    }

    /// Resolve a spec by `(role, id)`. Errors (naming id + role) if no spec has
    /// that id, or if the matched spec has a different role.
    pub fn resolve(&self, role: ModelRole, id: &str) -> Result<&ModelSpec> {
        match self.specs.iter().find(|s| s.id == id) {
            None => anyhow::bail!(
                "no model `{id}` registered for role {role:?} \
                 (check `models.toml` and the {role:?} selection env var)"
            ),
            Some(s) if s.role != role => {
                anyhow::bail!("model `{id}` is registered as {:?}, not {role:?}", s.role)
            }
            Some(s) => Ok(s),
        }
    }

    /// The active embedder spec (from `config.embedder` / `SEMANTEX_EMBEDDER`).
    pub fn active_embedder(&self) -> Result<&ModelSpec> {
        self.resolve(ModelRole::Embedder, &self.active_embedder_id)
    }

    /// The active reranker spec (from `config.reranker_model`).
    pub fn active_reranker(&self) -> Result<&ModelSpec> {
        self.resolve(ModelRole::Reranker, &self.active_reranker_id)
    }

    /// The active LLM spec, or `Ok(None)` when none is selected (the default,
    /// zero-LLM-deps build). Errors only if a non-empty id fails to resolve.
    pub fn active_llm(&self) -> Result<Option<&ModelSpec>> {
        if self.active_llm_id.trim().is_empty() {
            return Ok(None);
        }
        self.resolve(ModelRole::Llm, &self.active_llm_id).map(Some)
    }

    /// The dense backend the active embedder's capabilities select — the value
    /// S1's `hybrid.rs`/`builder.rs` selection will consume in place of
    /// `DenseBackendKind::parse(&config.dense_backend)` (post-S2 reconciliation).
    ///
    /// INTERIM (Phase 1, pre-S2): the underlying `BackendKind::dense_kind` is
    /// fallible because S1's `DenseBackendKind` has only the `ColbertPlaid`
    /// variant today. A multi-vector embedder (e.g. the default `lateon-colbert`)
    /// resolves to `Ok(ColbertPlaid)`; a single-vector embedder (e.g.
    /// `coderank-137m`) errors with "coderank-hnsw backend not available until
    /// S2". S2 + the reconciliation pass make this total.
    pub fn embedder_backend_kind(&self) -> Result<DenseBackendKind> {
        let spec = self.active_embedder()?;
        // Defensive: an embedder spec always carries EmbedderSpec data (validate
        // enforces role agreement), but check explicitly to avoid a panic path.
        match &spec.role_data {
            RoleData::Embedder(_) => {}
            _ => anyhow::bail!("active embedder `{}` is not an embedder spec", spec.id),
        }
        backend_for(&spec.capabilities).dense_kind()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry_from_builtins() -> ModelRegistry {
        // No project path, no user manifest → built-ins only.
        ModelRegistry::from_config(&SemantexConfig::default(), None).unwrap()
    }

    #[test]
    fn resolves_active_embedder_default() {
        // D4: the shipped default embedder is lateon-colbert until cutover.
        let reg = registry_from_builtins();
        let spec = reg.active_embedder().unwrap();
        assert_eq!(spec.id, "lateon-colbert");
        assert_eq!(spec.role, ModelRole::Embedder);
    }

    #[test]
    fn active_embedder_backend_kind_is_plaid_for_default() {
        // lateon-colbert is multi_vector → PLAID, the D4 default dense path.
        let reg = registry_from_builtins();
        assert_eq!(
            reg.embedder_backend_kind().unwrap(),
            DenseBackendKind::ColbertPlaid
        );
    }

    #[test]
    fn coderank_selection_errors_until_s2() {
        // INTERIM Phase-1: the single-vector embedder routes to the coderank-hnsw
        // backend, which S1's DenseBackendKind cannot represent until S2 lands.
        // So `embedder_backend_kind` errors honestly — the spec still resolves.
        let mut cfg = SemantexConfig::default();
        cfg.embedder = "coderank-137m".to_string();
        let reg = ModelRegistry::from_config(&cfg, None).unwrap();
        assert_eq!(reg.active_embedder().unwrap().id, "coderank-137m");
        let err = reg
            .embedder_backend_kind()
            .expect_err("coderank-hnsw must error until S2");
        assert!(err.to_string().contains("S2"), "got: {err}");
    }

    #[test]
    fn resolves_active_reranker_default() {
        let reg = registry_from_builtins();
        assert_eq!(reg.active_reranker().unwrap().id, "bge-reranker-v2-m3");
    }

    #[test]
    fn active_llm_is_none_by_default() {
        let reg = registry_from_builtins();
        assert!(
            reg.active_llm().unwrap().is_none(),
            "no LLM selected by default"
        );
    }

    #[test]
    fn unknown_active_id_errors_naming_the_id_and_role() {
        let mut cfg = SemantexConfig::default();
        cfg.embedder = "does-not-exist".to_string();
        let reg = ModelRegistry::from_config(&cfg, None).unwrap();
        let err = reg.active_embedder().expect_err("unknown id must error");
        let msg = err.to_string();
        assert!(msg.contains("does-not-exist"), "got: {msg}");
        assert!(msg.contains("Embedder"), "got: {msg}");
    }

    #[test]
    fn resolve_wrong_role_errors() {
        let reg = registry_from_builtins();
        // bge is a reranker; asking for it as an embedder must fail.
        let err = reg
            .resolve(ModelRole::Embedder, "bge-reranker-v2-m3")
            .unwrap_err();
        assert!(err.to_string().contains("bge-reranker-v2-m3"));
    }

    #[cfg(feature = "llm")]
    #[test]
    fn llm_builtin_resolves_when_feature_on_and_selected() {
        let mut cfg = SemantexConfig::default();
        cfg.llm_model = "ollama-default".to_string();
        let reg = ModelRegistry::from_config(&cfg, None).unwrap();
        let spec = reg.active_llm().unwrap().expect("llm spec should resolve");
        assert_eq!(spec.id, "ollama-default");
        assert_eq!(spec.role, ModelRole::Llm);
    }
}
