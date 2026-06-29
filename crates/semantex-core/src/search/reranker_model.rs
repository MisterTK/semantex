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
use crate::search::fastembed_reranker::{ENV_MODEL, select_model_from_env};
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
    /// Max sequence length (tokens) the model is truncated to. From the reranker
    /// spec; bounds the ONNX sequence so it never exceeds the model's trained
    /// range or CPU O(seq²) budget.
    pub max_context: usize,
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
            max_context: rspec.max_context,
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
        "jina-reranker-v2-base-multilingual" | "jina-v2" => Some(M::JINARerankerV2BaseMultiligual),
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
            // Resolve through the SAME built-in manifest spec the registry uses,
            // so the env alias can never drift from the authoritative Qwen3
            // coordinates / verified chat template / token ids (the Fix-2 hazard).
            choice_from_builtin_id("qwen3-reranker-0.6b")
                .unwrap_or_else(|| RerankerChoice::Fastembed(select_model_from_env()))
        }
        "bge-reranker-v2-m3-onnx" | "bge-v2-m3-onnx" | "bge-onnx" => {
            RerankerChoice::Onnx(bge_v2_m3_onnx())
        }
        // All other values (incl. unknown/empty) -> the existing fastembed map.
        _ => RerankerChoice::Fastembed(select_model_from_env()),
    }
}

/// Look up a built-in manifest spec by id and resolve it via `from_spec`, so the
/// env-alias path shares the manifest's authoritative coordinates/template/ids.
fn choice_from_builtin_id(id: &str) -> Option<RerankerChoice> {
    crate::model::manifest::builtin_specs()
        .into_iter()
        .find(|s| s.id == id)
        .and_then(|spec| RerankerChoice::from_spec(&spec).ok())
}

/// bge-reranker-v2-m3 driven through the GENERIC loader (classifier head). The
/// permissive smoke target proving `ScoreStrategy::ClassifierLogit`. Has no
/// manifest equivalent (the built-in `bge-reranker-v2-m3` routes to fastembed by
/// id), so it stays a dedicated alias spec. `max_context` = 512 (bge's trained
/// context).
fn bge_v2_m3_onnx() -> OnnxModelSpec {
    OnnxModelSpec {
        files: ModelFiles {
            subdir: "bge-reranker-v2-m3-onnx".to_string(),
            base_url: "https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/onnx"
                .to_string(),
            files: vec!["model.onnx".to_string(), "tokenizer.json".to_string()],
        },
        session_file: "model.onnx".to_string(),
        max_context: 512,
        strategy: ScoreStrategy::ClassifierLogit,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SemantexConfig;
    use crate::model::ModelRegistry;

    fn with_env<F: FnOnce()>(key: &str, val: Option<&str>, f: F) {
        let _g = crate::search::RERANKER_TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
            with_env(
                ENV_MODEL,
                Some(alias),
                || match select_reranker_choice_from_env() {
                    RerankerChoice::Onnx(spec) => {
                        assert!(matches!(spec.strategy, ScoreStrategy::YesNoLogit { .. }));
                        assert_eq!(spec.session_file, "model.onnx");
                        assert_eq!(spec.files.subdir, "Qwen3-Reranker-0.6B-onnx");
                    }
                    other @ RerankerChoice::Fastembed(_) => {
                        panic!("expected Onnx for {alias}, got {other:?}")
                    }
                },
            );
        }
    }

    #[test]
    fn bge_onnx_alias_routes_to_onnx_classifier() {
        with_env(
            ENV_MODEL,
            Some("bge-onnx"),
            || match select_reranker_choice_from_env() {
                RerankerChoice::Onnx(spec) => {
                    assert!(matches!(spec.strategy, ScoreStrategy::ClassifierLogit));
                }
                other @ RerankerChoice::Fastembed(_) => {
                    panic!("expected Onnx classifier, got {other:?}")
                }
            },
        );
    }

    #[test]
    fn default_and_unknown_route_to_fastembed_bge() {
        for v in [None, Some(""), Some("garbage"), Some("bge-reranker-v2-m3")] {
            with_env(ENV_MODEL, v, || match select_reranker_choice_from_env() {
                RerankerChoice::Fastembed(m) => {
                    assert_eq!(m, fastembed::RerankerModel::BGERerankerV2M3);
                }
                other @ RerankerChoice::Onnx(_) => {
                    panic!("expected Fastembed bge for {v:?}, got {other:?}")
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
            other @ RerankerChoice::Onnx(_) => {
                panic!("expected Fastembed bge from registry spec, got {other:?}")
            }
        }
    }

    #[test]
    fn from_spec_resolves_qwen3_to_onnx_yesno() {
        // Select qwen3 via config (== SEMANTEX_RERANKER_MODEL) -> ONNX yes/no.
        let cfg = SemantexConfig {
            reranker_model: "qwen3-reranker-0.6b".to_string(),
            ..Default::default()
        };
        let reg = ModelRegistry::from_config(&cfg, None).unwrap();
        let spec = reg.active_reranker().unwrap();
        match RerankerChoice::from_spec(spec).unwrap() {
            RerankerChoice::Onnx(o) => {
                // Strategy + coordinates come from the registry spec as DATA.
                match &o.strategy {
                    ScoreStrategy::YesNoLogit {
                        yes_id,
                        no_id,
                        prompt,
                    } => {
                        assert_eq!(*yes_id, 9693);
                        assert_eq!(*no_id, 2152);
                        // FIX 2: the wired template MUST be the verified FULL chat
                        // form (im_start/im_end control tokens + assistant/<think>
                        // terminator), not the simplified <Relevant> stub.
                        let rendered = prompt.render("binary search", "fn bsearch() {}");
                        assert!(
                            rendered.contains("<|im_start|>system"),
                            "missing system im_start; got:\n{rendered}"
                        );
                        assert!(
                            rendered.contains("<|im_start|>user"),
                            "missing user im_start"
                        );
                        assert!(
                            rendered.contains("<|im_start|>assistant"),
                            "missing assistant im_start"
                        );
                        assert!(rendered.contains("<|im_end|>"), "missing im_end");
                        assert!(rendered.contains("<think>"), "missing <think> terminator");
                        assert!(
                            rendered.contains("binary search") && rendered.contains("fn bsearch"),
                            "query/doc not injected"
                        );
                    }
                    other @ ScoreStrategy::ClassifierLogit => {
                        panic!("expected YesNoLogit, got {other:?}")
                    }
                }
                assert_eq!(o.session_file, "model.onnx");
                // FIX 1: max_context plumbed from the spec (Qwen3 CPU-sane cap).
                assert_eq!(o.max_context, 2048);
                assert_eq!(o.files.subdir, "Qwen3-Reranker-0.6B-onnx");
                assert!(
                    o.files
                        .base_url
                        .contains("MisterTK/Qwen3-Reranker-0.6B-onnx")
                );
            }
            other @ RerankerChoice::Fastembed(_) => {
                panic!("expected Onnx for qwen3 registry spec, got {other:?}")
            }
        }
    }

    /// FIX 2 (env-alias path): the `qwen3` env alias resolves THROUGH the same
    /// built-in manifest spec, so its rendered prompt also carries the verified
    /// chat-control markers (env and registry paths cannot drift).
    #[test]
    fn qwen3_env_alias_renders_full_chat_template() {
        with_env(
            ENV_MODEL,
            Some("qwen3"),
            || match select_reranker_choice_from_env() {
                RerankerChoice::Onnx(o) => match &o.strategy {
                    ScoreStrategy::YesNoLogit { prompt, .. } => {
                        let rendered = prompt.render("q", "d");
                        assert!(rendered.contains("<|im_start|>"), "missing im_start");
                        assert!(rendered.contains("<think>"), "missing <think>");
                        assert_eq!(o.max_context, 2048);
                    }
                    other @ ScoreStrategy::ClassifierLogit => {
                        panic!("expected YesNoLogit, got {other:?}")
                    }
                },
                other @ RerankerChoice::Fastembed(_) => {
                    panic!("expected Onnx for qwen3 alias, got {other:?}")
                }
            },
        );
    }
}
