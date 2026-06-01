//! Reranker model selection: resolves the active reranker to either a
//! fastembed-native model (`RerankerChoice::Fastembed`) or a generic ONNX
//! checkpoint (`RerankerChoice::Onnx`).
//!
//! S8 reconciliation: the AUTHORITATIVE selection is [`RerankerChoice::from_spec`],
//! which consumes a [`ModelSpec`] resolved by S8's `ModelRegistry`
//! (`registry.active_reranker()`). The reranker's score strategy, prompt
//! template, yes/no token ids, download coordinates, and ONNX session filename
//! all come from that spec as DATA — nothing model-specific is hardcoded here.
//! `SEMANTEX_RERANKER_MODEL` (S3) and `config.reranker_model` (S8) are the SAME
//! key, read once through the registry; the engine builds from `from_spec`.
//!
//! [`select_reranker_choice_from_env`] is the pre-S8 env-alias baseline kept for
//! back-compat / standalone use; it routes the new ONNX aliases (`qwen3*`,
//! `bge-onnx`) and otherwise delegates to the existing fastembed selector.
//!
//! Permissive-only (spec §8 / D3): every selectable model is MIT/Apache. There
//! is deliberately NO alias for jina-reranker-v3 (non-commercial license).

use crate::model::spec::{ModelRole, ModelSource, ModelSpec, RoleData, ScoreStrategyKind};
use crate::search::fastembed_reranker::{select_model_from_env, ENV_MODEL};
use crate::search::onnx_reranker::{PromptTemplate, ScoreStrategy};
use crate::search::reranker_download::ModelFiles;
use anyhow::{Context, Result};

/// Static metadata + concrete score strategy for an ONNX reranker model. Carries
/// everything `OnnxReranker` needs to download and run the model. Built from a
/// registry `ModelSpec` (`from_spec`) or a built-in env alias.
#[derive(Debug, Clone)]
pub struct OnnxModelSpec {
    /// Download coordinates (subdir, base url, file list; `files[0]` = sentinel).
    pub files: ModelFiles,
    /// ONNX session filename within the model dir (`model_int8.onnx` /
    /// `model.onnx`). Always one of `files`.
    pub session_file: String,
    /// Concrete score-extraction strategy (with token ids + prompt for yes/no).
    pub strategy: ScoreStrategy,
}

/// The selected reranker model and the engine that runs it.
#[derive(Debug, Clone)]
pub enum RerankerChoice {
    /// A fastembed-native cross-encoder (bge-v2-m3 default, bge-base, jina v1/v2).
    Fastembed(fastembed::RerankerModel),
    /// A generic ONNX cross-encoder loaded by `OnnxReranker`.
    Onnx(OnnxModelSpec),
}

impl RerankerChoice {
    /// Resolve a `RerankerChoice` from an S8 registry [`ModelSpec`]. THE
    /// authoritative path: the score strategy, prompt, token ids, download
    /// coordinates, and session filename are all read from the spec as data.
    ///
    /// The fastembed-native models (bge-v2-m3 / bge-base / jina v1/v2) are still
    /// run through `FastembedReranker`, selected by spec id; any other reranker
    /// spec runs through the generic ONNX loader.
    ///
    /// # Errors
    /// Errors if `spec` is not a reranker, lists no files, or is a `YesNoLogit`
    /// reranker missing its token ids.
    pub fn from_spec(spec: &ModelSpec) -> Result<Self> {
        anyhow::ensure!(
            spec.role == ModelRole::Reranker,
            "model `{}` is not a reranker (role {:?})",
            spec.id,
            spec.role
        );
        let RoleData::Reranker(rspec) = &spec.role_data else {
            anyhow::bail!("reranker `{}` has no reranker role data", spec.id);
        };

        // fastembed-native ids keep going through fastembed (it owns their
        // download + ONNX session). Everything else uses the generic loader.
        if let Some(model) = fastembed_model_for_id(&spec.id) {
            return Ok(Self::Fastembed(model));
        }

        let (subdir, base_url, files) = source_to_coords(&spec.source, &spec.id)?;
        let session_file = files
            .first()
            .cloned()
            .with_context(|| format!("reranker `{}` lists no files", spec.id))?;
        let strategy = match rspec.score_strategy {
            ScoreStrategyKind::ClassifierHead => ScoreStrategy::ClassifierLogit,
            ScoreStrategyKind::YesNoLogit => {
                let yes_id = rspec.yes_token_id.with_context(|| {
                    format!("yes_no reranker `{}` is missing `yes_token_id`", spec.id)
                })?;
                let no_id = rspec.no_token_id.with_context(|| {
                    format!("yes_no reranker `{}` is missing `no_token_id`", spec.id)
                })?;
                ScoreStrategy::YesNoLogit {
                    yes_id,
                    no_id,
                    prompt: PromptTemplate {
                        prefix: rspec.prompt_prefix.clone(),
                        middle: rspec.prompt_middle.clone(),
                        suffix: rspec.prompt_suffix.clone(),
                    },
                }
            }
        };
        Ok(Self::Onnx(OnnxModelSpec {
            files: ModelFiles {
                subdir,
                base_url,
                files,
            },
            session_file,
            strategy,
        }))
    }
}

/// Map a reranker spec id to a fastembed-native `RerankerModel`, or `None` if the
/// id should route through the generic ONNX loader. Mirrors the alias set that
/// `select_model_from_env` understands so the registry and env paths agree.
fn fastembed_model_for_id(id: &str) -> Option<fastembed::RerankerModel> {
    use fastembed::RerankerModel as M;
    match id {
        "bge-reranker-v2-m3" | "bge-v2-m3" => Some(M::BGERerankerV2M3),
        "bge-reranker-base" | "bge-base" => Some(M::BGERerankerBase),
        "jina-reranker-v1-turbo-en" | "jina-v1" => Some(M::JINARerankerV1TurboEn),
        "jina-reranker-v2-base-multilingual" | "jina-v2" => {
            Some(M::JINARerankerV2BaseMultiligual)
        }
        _ => None,
    }
}

/// Turn a [`ModelSource`] into `(subdir, base_url, files)` download coordinates.
/// `Local` sources point `base_url` at the local dir (download is a no-op since
/// files already exist); `Hf`/`Url` build a resolve base.
fn source_to_coords(source: &ModelSource, id: &str) -> Result<(String, String, Vec<String>)> {
    match source {
        ModelSource::Hf { repo, files } => {
            let subdir = repo.rsplit('/').next().unwrap_or(repo).to_string();
            let base_url = format!("https://huggingface.co/{repo}/resolve/main");
            Ok((subdir, base_url, files.clone()))
        }
        ModelSource::Url { base, files } => {
            // Subdir from the spec id (urls have no canonical repo name).
            Ok((id.to_string(), base.clone(), files.clone()))
        }
        ModelSource::Local { dir } => {
            // Hand-placed weights: the "subdir" is the absolute dir itself and we
            // never download. ensure_reranker_model joins models_dir/subdir, so a
            // Local source is only meaningful when used directly; flag it.
            anyhow::bail!(
                "reranker `{id}`: ModelSource::Local ({dir}) is not yet supported by the \
                 generic ONNX reranker download path"
            )
        }
    }
}

/// Resolve `SEMANTEX_RERANKER_MODEL` to a `RerankerChoice` (pre-S8 env baseline).
/// ONNX-only aliases take precedence; everything else delegates to the existing
/// fastembed selector (so unknown values still warn-and-fall-back to bge-v2-m3).
///
/// Prefer [`RerankerChoice::from_spec`] (registry-driven) in the engine; this
/// exists for back-compat and to keep the master `SEMANTEX_RERANKER_MODEL`
/// aliases working when the registry is not threaded through.
#[must_use]
pub fn select_reranker_choice_from_env() -> RerankerChoice {
    let raw = std::env::var(ENV_MODEL).unwrap_or_default();
    match raw.to_ascii_lowercase().as_str() {
        "qwen3-reranker-0.6b" | "qwen3-reranker" | "qwen3" => {
            RerankerChoice::Onnx(qwen3_reranker_0_6b())
        }
        "bge-reranker-v2-m3-onnx" | "bge-v2-m3-onnx" | "bge-onnx" => {
            RerankerChoice::Onnx(bge_v2_m3_onnx())
        }
        // All other values (incl. unknown/empty) -> the existing fastembed map.
        _ => RerankerChoice::Fastembed(select_model_from_env()),
    }
}

/// Qwen3-Reranker-0.6B (Apache-2.0) coordinates for the env-alias path. Mirrors
/// the S8 manifest built-in (`MisterTK/Qwen3-Reranker-0.6B-onnx`, fp16
/// `model.onnx`, yes/no ids 9693/2152). The registry path (`from_spec`) is
/// authoritative; this keeps the env alias self-contained.
fn qwen3_reranker_0_6b() -> OnnxModelSpec {
    OnnxModelSpec {
        files: ModelFiles {
            subdir: "Qwen3-Reranker-0.6B-onnx".to_string(),
            base_url: "https://huggingface.co/MisterTK/Qwen3-Reranker-0.6B-onnx/resolve/main"
                .to_string(),
            files: vec![
                "model.onnx".to_string(),
                "tokenizer.json".to_string(),
                "config.json".to_string(),
            ],
        },
        session_file: "model.onnx".to_string(),
        strategy: ScoreStrategy::YesNoLogit {
            yes_id: 9693,
            no_id: 2152,
            prompt: PromptTemplate {
                prefix:
                    "<Instruct>: Given a code search query, judge whether the document is relevant.\n<Query>: "
                        .to_string(),
                middle: "\n<Document>: ".to_string(),
                suffix: "\n<Relevant>:".to_string(),
            },
        },
    }
}

/// bge-reranker-v2-m3 driven through the GENERIC loader (classifier head). The
/// permissive smoke target proving `ScoreStrategy::ClassifierLogit`.
fn bge_v2_m3_onnx() -> OnnxModelSpec {
    OnnxModelSpec {
        files: ModelFiles {
            subdir: "bge-reranker-v2-m3-onnx".to_string(),
            base_url: "https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/onnx"
                .to_string(),
            files: vec!["model.onnx".to_string(), "tokenizer.json".to_string()],
        },
        session_file: "model.onnx".to_string(),
        strategy: ScoreStrategy::ClassifierLogit,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelRegistry;
    use crate::config::SemantexConfig;

    fn with_env<F: FnOnce()>(key: &str, val: Option<&str>, f: F) {
        use std::sync::Mutex;
        static LOCK: Mutex<()> = Mutex::new(());
        let _g = LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = std::env::var(key).ok();
        // SAFETY: guarded by LOCK.
        unsafe {
            match val {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        // SAFETY: guarded by LOCK.
        unsafe {
            match prior {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
        if let Err(e) = r {
            std::panic::resume_unwind(e);
        }
    }

    // ── env-alias selection ────────────────────────────────────────────────

    #[test]
    fn qwen3_aliases_route_to_onnx_yesno() {
        for alias in ["qwen3", "Qwen3-Reranker-0.6B", "qwen3-reranker"] {
            with_env(ENV_MODEL, Some(alias), || {
                match select_reranker_choice_from_env() {
                    RerankerChoice::Onnx(spec) => {
                        assert!(matches!(spec.strategy, ScoreStrategy::YesNoLogit { .. }));
                        assert_eq!(spec.session_file, "model.onnx");
                        assert_eq!(spec.files.subdir, "Qwen3-Reranker-0.6B-onnx");
                    }
                    other => panic!("expected Onnx for {alias}, got {other:?}"),
                }
            });
        }
    }

    #[test]
    fn bge_onnx_alias_routes_to_onnx_classifier() {
        with_env(ENV_MODEL, Some("bge-onnx"), || {
            match select_reranker_choice_from_env() {
                RerankerChoice::Onnx(spec) => {
                    assert!(matches!(spec.strategy, ScoreStrategy::ClassifierLogit));
                }
                other => panic!("expected Onnx classifier, got {other:?}"),
            }
        });
    }

    #[test]
    fn default_and_unknown_route_to_fastembed_bge() {
        for v in [None, Some(""), Some("garbage"), Some("bge-reranker-v2-m3")] {
            with_env(ENV_MODEL, v, || {
                match select_reranker_choice_from_env() {
                    RerankerChoice::Fastembed(m) => {
                        assert_eq!(m, fastembed::RerankerModel::BGERerankerV2M3);
                    }
                    other => panic!("expected Fastembed bge for {v:?}, got {other:?}"),
                }
            });
        }
    }

    /// D3 guard: there must be no selectable alias for the NC jina-reranker-v3.
    #[test]
    fn jina_v3_is_not_selectable() {
        with_env(ENV_MODEL, Some("jina-reranker-v3"), || {
            assert!(matches!(
                select_reranker_choice_from_env(),
                RerankerChoice::Fastembed(fastembed::RerankerModel::BGERerankerV2M3)
            ));
        });
    }

    // ── registry-driven from_spec (S8 reconciliation) ──────────────────────

    #[test]
    fn from_spec_resolves_bge_to_fastembed() {
        // Default active reranker is bge-reranker-v2-m3 -> fastembed path.
        let reg = ModelRegistry::from_config(&SemantexConfig::default(), None).unwrap();
        let spec = reg.active_reranker().unwrap();
        match RerankerChoice::from_spec(spec).unwrap() {
            RerankerChoice::Fastembed(m) => {
                assert_eq!(m, fastembed::RerankerModel::BGERerankerV2M3);
            }
            other => panic!("expected Fastembed bge from registry spec, got {other:?}"),
        }
    }

    #[test]
    fn from_spec_resolves_qwen3_to_onnx_yesno() {
        // Select qwen3 via config (== SEMANTEX_RERANKER_MODEL) -> ONNX yes/no.
        let mut cfg = SemantexConfig::default();
        cfg.reranker_model = "qwen3-reranker-0.6b".to_string();
        let reg = ModelRegistry::from_config(&cfg, None).unwrap();
        let spec = reg.active_reranker().unwrap();
        match RerankerChoice::from_spec(spec).unwrap() {
            RerankerChoice::Onnx(o) => {
                // Strategy + coordinates come from the registry spec as DATA.
                match o.strategy {
                    ScoreStrategy::YesNoLogit { yes_id, no_id, prompt } => {
                        assert_eq!(yes_id, 9693);
                        assert_eq!(no_id, 2152);
                        assert!(prompt.suffix.contains("Relevant") || !prompt.suffix.is_empty());
                    }
                    other => panic!("expected YesNoLogit, got {other:?}"),
                }
                assert_eq!(o.session_file, "model.onnx");
                assert_eq!(o.files.subdir, "Qwen3-Reranker-0.6B-onnx");
                assert!(o.files.base_url.contains("MisterTK/Qwen3-Reranker-0.6B-onnx"));
            }
            other => panic!("expected Onnx for qwen3 registry spec, got {other:?}"),
        }
    }
}
