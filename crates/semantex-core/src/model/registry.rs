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
    /// A single-vector embedder (e.g. `coderank-137m` / `qwen3-embed-0.6b`)
    /// resolves to `Ok(CoderankHnsw)`. A multi-vector embedder has no built-in
    /// backend (D4 removed the ColBERT/PLAID path) and errors via `backend_for`.
    /// The `Result` also covers the wrong-role / unknown-id error arms.
    pub fn embedder_backend_kind(&self) -> Result<DenseBackendKind> {
        let spec = self.active_embedder()?;
        // Defensive: an embedder spec always carries EmbedderSpec data (validate
        // enforces role agreement), but check explicitly to avoid a panic path.
        match &spec.role_data {
            RoleData::Embedder(_) => {}
            _ => anyhow::bail!("active embedder `{}` is not an embedder spec", spec.id),
        }
        backend_for(&spec.capabilities)?.dense_kind()
    }

    /// Resolve the effective dense backend for a config, honoring the canonical
    /// `SEMANTEX_EMBEDDER` selection AND the DEPRECATED `SEMANTEX_DENSE_BACKEND`
    /// / `config.dense_backend` alias.
    ///
    /// Selection (D4 — coderank-hnsw is the sole built-in dense backend):
    /// * Deprecated alias: if `config.dense_backend` parses to a known backend
    ///   name, that alias WINS directly (keeps the `SEMANTEX_DENSE_BACKEND` knob
    ///   live for explicit selection / future backends).
    /// * Canonical path: otherwise the active embedder (`config.embedder`) routes
    ///   via its capabilities to a backend (`coderank-137m` / `qwen3-embed-0.6b`
    ///   → coderank-hnsw). The shipped all-default config resolves here.
    ///
    /// An UNKNOWN alias value (e.g. a stale `colbert-plaid` from an old config or
    /// a typo) does NOT match — it falls through to the canonical embedder path,
    /// so a removed backend name degrades to the default rather than erroring.
    pub fn resolve_dense_backend(
        config: &SemantexConfig,
        project_path: Option<&Path>,
    ) -> Result<DenseBackendKind> {
        // Deprecated alias takes precedence when it parses to a known backend.
        if let Some(kind) = DenseBackendKind::parse(&config.dense_backend) {
            return Ok(kind);
        }
        // Canonical: route through the active embedder's capabilities (a project
        // `models.toml` may add/override embedder specs).
        let registry = Self::from_config(config, project_path)?;
        registry.embedder_backend_kind()
    }

    /// The fingerprint of the active embedder (covering id, dims, pooling, quant,
    /// normalization, prefixes, AND the runtime `SEMANTEX_DENSE_CONTEXT` flag).
    /// This is the value `builder.rs` stamps into `meta.json` and that
    /// `state::is_stale_for_embedder` / the S8 versioned-dir selection compare
    /// against: a change here means a different vector space, so the index must be
    /// re-embedded. Backend-agnostic (does not depend on the dense backend arm).
    ///
    /// The `dense_context` flag is read HERE, once, from the environment via
    /// [`dense_context_enabled`] — the SINGLE env-read site for the fingerprint —
    /// so the builder's write-time fingerprint and `detect_for_config`'s
    /// open-time expected fingerprint can never disagree about it.
    pub fn active_embedder_fingerprint(&self) -> Result<String> {
        let spec = self.active_embedder()?;
        let RoleData::Embedder(data) = &spec.role_data else {
            anyhow::bail!("active embedder `{}` is not an embedder spec", spec.id);
        };
        Ok(crate::model::EmbedderFingerprint::compute(
            &spec.id,
            data,
            dense_context_enabled(),
        ))
    }

    /// Resolve the active embedder fingerprint straight from a `config`
    /// (+ optional project path for a local `models.toml`). Convenience wrapper
    /// over `from_config(..).active_embedder_fingerprint()` for the staleness /
    /// builder call sites.
    pub fn resolve_embedder_fingerprint(
        config: &SemantexConfig,
        project_path: Option<&Path>,
    ) -> Result<String> {
        Self::from_config(config, project_path)?.active_embedder_fingerprint()
    }
}

/// Whether the `SEMANTEX_DENSE_CONTEXT` A/B flag is enabled (default OFF). The
/// SINGLE canonical parse of this env var: `"1"` or a case-insensitive `"true"`.
///
/// Both the embedder fingerprint ([`ModelRegistry::active_embedder_fingerprint`])
/// and the builder's document-text selection read the flag through this one
/// function, so the embedded text and the fingerprint that stamps the index can
/// never drift apart (the F5 silent-wrong-index guarantee).
#[must_use]
pub fn dense_context_enabled() -> bool {
    std::env::var("SEMANTEX_DENSE_CONTEXT")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
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
    fn multi_vector_embedder_routes_to_colbert_plaid() {
        // A multi-vector embedder (declared via a user manifest) routes to the
        // opt-in colbert-plaid late-interaction backend via capabilities.
        let tmp = tempfile::TempDir::new().unwrap();
        let project = tmp.path().join("proj");
        let semantex_dir = project.join(".semantex");
        std::fs::create_dir_all(&semantex_dir).unwrap();
        std::fs::write(
            semantex_dir.join("models.toml"),
            r#"
            [[model]]
            id = "some-multivec"
            role = "embedder"
            [model.source]
            kind = "hf"
            repo = "example/some-multivec"
            files = ["model.onnx", "tokenizer.json"]
            [model.capabilities]
            multi_vector = true
            [model.embedder]
            dims = 96
            max_context = 512
            query_prefix = ""
            pooling = "late_interaction"
            quant = "int8_symmetric"
            "#,
        )
        .unwrap();
        let mut cfg = SemantexConfig::default();
        cfg.embedder = "some-multivec".to_string();
        let reg = ModelRegistry::from_config(&cfg, Some(&project)).unwrap();
        assert_eq!(
            reg.embedder_backend_kind().unwrap(),
            DenseBackendKind::ColbertPlaid,
            "multi-vector embedder must route to the colbert-plaid backend"
        );
    }

    #[test]
    fn coderank_selection_routes_to_hnsw() {
        // The single-vector embedder routes to the coderank-hnsw backend.
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
        // SEMANTEX_EMBEDDER=coderank-137m (canonical) with the default dense_backend
        // alias must route to coderank-hnsw via capabilities.
        let mut cfg = SemantexConfig::default();
        cfg.embedder = "coderank-137m".to_string();
        assert_eq!(
            ModelRegistry::resolve_dense_backend(&cfg, None).unwrap(),
            DenseBackendKind::CoderankHnsw
        );
    }

    #[test]
    fn resolve_dense_backend_alias_parses_known_backend() {
        // A SEMANTEX_DENSE_BACKEND alias that names a known backend wins directly.
        let mut cfg = SemantexConfig::default();
        cfg.dense_backend = "coderank-hnsw".to_string();
        assert_eq!(
            ModelRegistry::resolve_dense_backend(&cfg, None).unwrap(),
            DenseBackendKind::CoderankHnsw
        );
    }

    #[test]
    fn resolve_dense_backend_unknown_alias_falls_through_to_embedder() {
        // An unknown/typo alias doesn't parse — it falls through to the canonical
        // embedder path (default coderank-hnsw) rather than erroring. This is how
        // a stale or mistyped config degrades gracefully. (A KNOWN alias like
        // "colbert-plaid" parses and wins directly — covered separately.)
        let mut cfg = SemantexConfig::default();
        cfg.dense_backend = "totally-made-up".to_string();
        assert_eq!(
            ModelRegistry::resolve_dense_backend(&cfg, None).unwrap(),
            DenseBackendKind::CoderankHnsw
        );
    }

    #[test]
    fn resolve_dense_backend_colbert_plaid_alias_parses() {
        // The known colbert-plaid backend alias parses and wins directly (the
        // deprecated SEMANTEX_DENSE_BACKEND path), independent of the embedder.
        let mut cfg = SemantexConfig::default();
        cfg.dense_backend = "colbert-plaid".to_string();
        assert_eq!(
            ModelRegistry::resolve_dense_backend(&cfg, None).unwrap(),
            DenseBackendKind::ColbertPlaid
        );
    }

    #[test]
    fn resolve_dense_backend_all_default_is_coderank_hnsw() {
        // D4: the shipped all-default config resolves to coderank-hnsw.
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
