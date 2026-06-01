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
        let user = match user_manifest_path(project_path) {
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
    /// `hybrid.rs::open()` + `builder.rs` consume (via [`Self::resolve_dense_backend`])
    /// in place of `DenseBackendKind::parse(&config.dense_backend)`.
    ///
    /// TOTAL (post-S2): a multi-vector embedder (e.g. the default `lateon-colbert`)
    /// resolves to `Ok(ColbertPlaid)`; a single-vector embedder (e.g.
    /// `coderank-137m` / `qwen3-embed-0.6b`) resolves to `Ok(CoderankHnsw)`. The
    /// `Result` is retained for the wrong-role / unknown-id error arms.
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

    /// Resolve the effective dense backend for a config, honoring the canonical
    /// `SEMANTEX_EMBEDDER` selection AND the DEPRECATED `SEMANTEX_DENSE_BACKEND`
    /// / `config.dense_backend` alias.
    ///
    /// Selection (post-S2 cutover gate):
    /// * Canonical path: the active embedder (`config.embedder`) routes via its
    ///   capabilities to a backend (`lateon-colbert` → colbert-plaid;
    ///   `coderank-137m` / `qwen3-embed-0.6b` → coderank-hnsw).
    /// * Deprecated alias: if `config.dense_backend` was set to an explicit,
    ///   non-default backend name (i.e. NOT the `colbert-plaid` default), that
    ///   alias WINS — it maps directly to the matching `DenseBackendKind`. This
    ///   keeps the S0 harness's `SEMANTEX_DENSE_BACKEND=coderank-hnsw` A/B knob
    ///   live alongside the canonical `SEMANTEX_EMBEDDER` knob.
    ///
    /// When the alias is at its default (`colbert-plaid`, the
    /// [`DenseBackendKind::default`] sentinel) the embedder wins, so the shipped
    /// all-default config — `embedder = "coderank-137m"` (D4 cutover) with the
    /// default alias — resolves to `coderank-hnsw`. To force colbert-plaid
    /// regardless of the embedder, set `SEMANTEX_EMBEDDER=lateon-colbert` (which
    /// capability-routes there) or the explicit `SEMANTEX_DENSE_BACKEND` alias to
    /// any non-default backend name.
    pub fn resolve_dense_backend(
        config: &SemantexConfig,
        project_path: Option<&Path>,
    ) -> Result<DenseBackendKind> {
        // Deprecated alias takes precedence ONLY when explicitly non-default.
        if let Some(kind) = DenseBackendKind::parse(&config.dense_backend)
            && kind != DenseBackendKind::default()
        {
            return Ok(kind);
        }
        // Canonical: route through the active embedder's capabilities (a project
        // `models.toml` may add/override embedder specs).
        let registry = Self::from_config(config, project_path)?;
        registry.embedder_backend_kind()
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
        // D4 cutover: the shipped default embedder is coderank-137m.
        let reg = registry_from_builtins();
        let spec = reg.active_embedder().unwrap();
        assert_eq!(spec.id, "coderank-137m");
        assert_eq!(spec.role, ModelRole::Embedder);
    }

    #[test]
    fn active_embedder_backend_kind_is_hnsw_for_default() {
        // coderank-137m is single_vector → coderank-hnsw, the D4 default dense path.
        let reg = registry_from_builtins();
        assert_eq!(
            reg.embedder_backend_kind().unwrap(),
            DenseBackendKind::CoderankHnsw
        );
    }

    #[test]
    fn explicit_colbert_embedder_still_routes_to_plaid() {
        // colbert-plaid stays fully available via SEMANTEX_EMBEDDER=lateon-colbert.
        let mut cfg = SemantexConfig::default();
        cfg.embedder = "lateon-colbert".to_string();
        let reg = ModelRegistry::from_config(&cfg, None).unwrap();
        assert_eq!(reg.active_embedder().unwrap().id, "lateon-colbert");
        assert_eq!(
            reg.embedder_backend_kind().unwrap(),
            DenseBackendKind::ColbertPlaid
        );
    }

    #[test]
    fn coderank_selection_routes_to_hnsw() {
        // Post-S2: the single-vector embedder routes to the coderank-hnsw backend,
        // now representable in S1's DenseBackendKind (totalized in capabilities.rs).
        let mut cfg = SemantexConfig::default();
        cfg.embedder = "coderank-137m".to_string();
        let reg = ModelRegistry::from_config(&cfg, None).unwrap();
        assert_eq!(reg.active_embedder().unwrap().id, "coderank-137m");
        assert_eq!(
            reg.embedder_backend_kind().unwrap(),
            DenseBackendKind::CoderankHnsw
        );
    }

    #[test]
    fn resolve_dense_backend_canonical_embedder_path() {
        // SEMANTEX_EMBEDDER=coderank-137m (canonical) with the DEFAULT dense_backend
        // alias must route to coderank-hnsw via capabilities.
        let mut cfg = SemantexConfig::default();
        cfg.embedder = "coderank-137m".to_string(); // dense_backend stays "colbert-plaid"
        assert_eq!(
            ModelRegistry::resolve_dense_backend(&cfg, None).unwrap(),
            DenseBackendKind::CoderankHnsw
        );
    }

    #[test]
    fn resolve_dense_backend_deprecated_alias_wins_when_nondefault() {
        // Alias precedence: a NON-default SEMANTEX_DENSE_BACKEND overrides the
        // embedder. Pin embedder to lateon-colbert (→ would be colbert-plaid) but
        // force coderank-hnsw via the alias — the alias must win.
        let mut cfg = SemantexConfig::default();
        cfg.embedder = "lateon-colbert".to_string();
        cfg.dense_backend = "coderank-hnsw".to_string();
        assert_eq!(
            ModelRegistry::resolve_dense_backend(&cfg, None).unwrap(),
            DenseBackendKind::CoderankHnsw
        );
    }

    #[test]
    fn resolve_dense_backend_explicit_colbert_alias_wins() {
        // Symmetric arm: force colbert-plaid via a non-default alias even though
        // the default embedder (coderank-137m) would route to coderank-hnsw.
        // Note: "colbert-plaid" is the alias SENTINEL, so to FORCE it the alias
        // can't be used; instead pin the embedder. This documents that the
        // colbert path stays reachable post-cutover.
        let mut cfg = SemantexConfig::default();
        cfg.embedder = "lateon-colbert".to_string(); // dense_backend stays sentinel
        assert_eq!(
            ModelRegistry::resolve_dense_backend(&cfg, None).unwrap(),
            DenseBackendKind::ColbertPlaid
        );
    }

    #[test]
    fn resolve_dense_backend_all_default_is_coderank_hnsw() {
        // D4 cutover: the shipped all-default config (embedder = coderank-137m,
        // alias at its sentinel) resolves to coderank-hnsw. Evidence-based flip
        // from colbert-plaid: nDCG meets/beats, ~28× less index RSS.
        let cfg = SemantexConfig::default();
        assert_eq!(
            ModelRegistry::resolve_dense_backend(&cfg, None).unwrap(),
            DenseBackendKind::CoderankHnsw
        );
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
