//! Reranker engine dispatcher held by the hybrid search call site. Wraps either
//! the fastembed cross-encoder (`FastembedReranker`) or the generic ONNX loader
//! (`OnnxReranker`) behind one `rerank` signature.
//!
//! S8 reconciliation: [`RerankerEngine::from_config`] resolves the active
//! reranker through S8's `ModelRegistry` (`active_reranker()` → a single read of
//! `config.reranker_model`, which is the SAME key as `SEMANTEX_RERANKER_MODEL`)
//! and materializes the concrete strategy via `RerankerChoice::from_spec`. The
//! reranker's score strategy, prompt template, and yes/no token ids therefore
//! come from the registry as DATA — never hardcoded here.
//!
//! Off-by-default: every constructor bails when `SEMANTEX_RERANKER` is not
//! enabled, so no weights are downloaded and no ONNX session/tokenizer is built.

use anyhow::Result;

use crate::config::SemantexConfig;
use crate::model::ModelRegistry;
use crate::search::fastembed_reranker::{FastembedReranker, reranker_enabled};
use crate::search::onnx_reranker::OnnxReranker;
use crate::search::reranker_model::{
    OnnxModelSpec, RerankerChoice, select_reranker_choice_from_env,
};

/// Either reranker implementation, behind a uniform `rerank` API.
pub enum RerankerEngine {
    Fastembed(FastembedReranker),
    Onnx(OnnxReranker),
}

impl RerankerEngine {
    /// Build the active reranker resolved through S8's `ModelRegistry` from
    /// `config` (the registry reads `config.reranker_model` ==
    /// `SEMANTEX_RERANKER_MODEL`). Errors (and loads nothing) when the
    /// `SEMANTEX_RERANKER` master switch is off — callers gate on
    /// `config.rerank` first, but we re-check here so weights never load by
    /// accident.
    ///
    /// This is the path the hybrid call site uses; `RerankerChoice::from_spec`
    /// supplies the score strategy / prompt / token ids as DATA.
    pub fn from_config(config: &SemantexConfig, show_download_progress: bool) -> Result<Self> {
        if !reranker_enabled() {
            anyhow::bail!(
                "Refusing to construct reranker: SEMANTEX_RERANKER is not enabled. \
                 Set SEMANTEX_RERANKER=on to load model weights."
            );
        }
        let registry = ModelRegistry::from_config(config, None)?;
        let spec = registry.active_reranker()?;
        match RerankerChoice::from_spec(spec)? {
            RerankerChoice::Fastembed(model) => Ok(Self::Fastembed(FastembedReranker::new(
                model,
                show_download_progress,
            )?)),
            RerankerChoice::Onnx(onnx_spec) => {
                Ok(Self::Onnx(Self::build_onnx(config, &onnx_spec)?))
            }
        }
    }

    /// Build the env-selected reranker WITHOUT a registry (pre-S8 baseline).
    /// Resolves `SEMANTEX_RERANKER_MODEL` via `select_reranker_choice_from_env`.
    /// Kept for standalone use; the hybrid call site prefers `from_config`.
    pub fn new_default(show_download_progress: bool) -> Result<Self> {
        if !reranker_enabled() {
            anyhow::bail!(
                "Refusing to construct reranker: SEMANTEX_RERANKER is not enabled. \
                 Set SEMANTEX_RERANKER=on to load model weights."
            );
        }
        let config = SemantexConfig::default();
        match select_reranker_choice_from_env() {
            RerankerChoice::Fastembed(model) => Ok(Self::Fastembed(FastembedReranker::new(
                model,
                show_download_progress,
            )?)),
            RerankerChoice::Onnx(onnx_spec) => {
                Ok(Self::Onnx(Self::build_onnx(&config, &onnx_spec)?))
            }
        }
    }

    /// Download the ONNX model (per its spec coordinates) and build an
    /// `OnnxReranker` with the concrete score strategy. Threads come from
    /// `SEMANTEX_ORT_THREADS` (query default 4, same as ColbertEmbedder); CoreML
    /// opt-in via `SEMANTEX_COREML`.
    fn build_onnx(config: &SemantexConfig, spec: &OnnxModelSpec) -> Result<OnnxReranker> {
        let model_dir = crate::search::reranker_download::ensure_reranker_model(
            &config.models_dir(),
            &spec.files,
        )?;
        let threads = crate::config::env_usize("SEMANTEX_ORT_THREADS", 4);
        let use_coreml = std::env::var("SEMANTEX_COREML").is_ok_and(|v| v == "1");
        OnnxReranker::new(
            &model_dir,
            &spec.session_file,
            spec.strategy.clone(),
            spec.max_context,
            threads,
            use_coreml,
        )
    }

    /// Rerank — delegates to whichever engine is active. Identical signature to
    /// `FastembedReranker::rerank`, so the hybrid call site is unchanged.
    pub fn rerank(
        &mut self,
        query: &str,
        documents: &[&str],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        match self {
            Self::Fastembed(r) => r.rerank(query, documents, top_k),
            Self::Onnx(r) => r.rerank(query, documents, top_k),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_env<F: FnOnce()>(vars: &[(&str, Option<&str>)], f: F) {
        let _g = crate::search::RERANKER_TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior: Vec<(String, Option<String>)> = vars
            .iter()
            .map(|(k, _)| ((*k).to_string(), std::env::var(*k).ok()))
            .collect();
        // SAFETY: guarded by LOCK.
        unsafe {
            for (k, v) in vars {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        // SAFETY: guarded by LOCK.
        unsafe {
            for (k, v) in &prior {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
        if let Err(e) = r {
            std::panic::resume_unwind(e);
        }
    }

    #[test]
    fn new_default_refuses_when_disabled() {
        use crate::search::fastembed_reranker::{ENV_ENABLE, ENV_MODEL};
        with_env(&[(ENV_ENABLE, Some("off")), (ENV_MODEL, None)], || {
            match RerankerEngine::new_default(false) {
                Ok(_) => panic!("must not construct when SEMANTEX_RERANKER is off"),
                Err(e) => assert!(format!("{e}").contains("SEMANTEX_RERANKER")),
            }
        });
        // Also true for the qwen3 ONNX selection: still refuses, no download.
        with_env(&[(ENV_ENABLE, None), (ENV_MODEL, Some("qwen3"))], || {
            assert!(RerankerEngine::new_default(false).is_err());
        });
    }

    #[test]
    fn from_config_refuses_when_disabled() {
        use crate::search::fastembed_reranker::ENV_ENABLE;
        with_env(&[(ENV_ENABLE, Some("off"))], || {
            let cfg = SemantexConfig::default();
            // Registry-driven path also bails before any download/session build.
            assert!(RerankerEngine::from_config(&cfg, false).is_err());
        });
    }
}
