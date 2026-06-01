//! Cross-encoder reranker for the final ranking stage.
//!
//! # Background (E1 in v0.3 SOTA spec)
//!
//! Prior versions of semantex wired in `JINARerankerV1TurboEn`, but disabled
//! it by default because benchmark ablations showed it slightly *hurt* F1 on
//! the owned 30-query suite. v0.3 replaces that model with a **code-aware
//! cross-encoder** — by default `BAAI/bge-reranker-v2-m3` — and gates the
//! whole stage behind an opt-in env var so we never re-enable it before
//! benchmarks confirm a net win.
//!
//! # Environment variables
//!
//! - `SEMANTEX_RERANKER` — master switch. Accepts `on|1|true` (case-insensitive)
//!   to enable; any other value (including unset) disables. **Default: off.**
//! - `SEMANTEX_RERANKER_MODEL` — override the cross-encoder model. Resolved by
//!   `search::reranker_model` (authoritatively `RerankerChoice::from_spec` via
//!   S8's `ModelRegistry`; `select_reranker_choice_from_env` for the standalone
//!   env path), which routes to one of two engines:
//!     - **fastembed-native** (this module): `bge-reranker-v2-m3` (default when
//!       enabled, multilingual/code-friendly), `bge-reranker-base` (smaller,
//!       EN/ZH only), `jina-reranker-v1-turbo-en` (legacy, kept for A/B),
//!       `jina-reranker-v2-base-multilingual`.
//!     - **generic ONNX loader** (`search::onnx_reranker`): `qwen3-reranker-0.6b`
//!       (Apache-2.0, code-capable, yes/no-logit), `bge-reranker-v2-m3-onnx`
//!       (classifier-logit smoke target).
//!   Unknown values warn and fall back to `bge-reranker-v2-m3` (fastembed).
//!
//! When the master switch is off, `FastembedReranker::rerank` returns an
//! identity ordering (original index, score 0.0) so the caller's pipeline keeps
//! working without code changes. No model is loaded into memory in that case
//! (lazy init).
//!
//! # Cache location
//!
//! Per the v0.3 spec we cache downloaded weights at `~/.fastembed_cache/`
//! (resolved via `dirs::home_dir`). Fastembed's own default is `./.fastembed_cache`
//! relative to cwd, which is unfriendly on per-project daemons. Override with
//! `FASTEMBED_CACHE_DIR` to keep parity with upstream behavior.
//!
//! # Model choice rationale (v0.3 default = bge-reranker-v2-m3)
//!
//! 1. The spec lists `BAAI/bge-reranker-v2-m3` as candidate (a) and accepts
//!    the pre-trained baseline for v0.3 if fine-tuning isn't feasible in scope.
//! 2. `fastembed` 5.9 ships native support (`RerankerModel::BGERerankerV2M3`),
//!    so we avoid wiring `ort` directly.
//! 3. Multilingual BGE handles code identifiers + comments better than the
//!    EN-only Jina turbo path that previously regressed F1.
//! 4. Candidate (b), `nomic-ai/CodeRankEmbed` cross-encoder, is not currently
//!    exposed as a stand-alone reranker checkpoint on HuggingFace at the time
//!    of writing. If/when it ships, add it to `select_model_from_env` and
//!    re-run the ablation.

use anyhow::{Context, Result};
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use ort::ep;
use std::path::PathBuf;

/// Master switch env var — `on|1|true` enables, anything else disables.
pub const ENV_ENABLE: &str = "SEMANTEX_RERANKER";
/// Optional model override env var.
pub const ENV_MODEL: &str = "SEMANTEX_RERANKER_MODEL";

/// Returns `true` iff `SEMANTEX_RERANKER` is set to an enabling value.
///
/// Accepted truthy values (case-insensitive): `on`, `1`, `true`, `yes`.
/// Anything else (including the env var being unset) returns `false`.
#[must_use]
pub fn reranker_enabled() -> bool {
    matches!(
        std::env::var(ENV_ENABLE).ok().map(|v| v.to_ascii_lowercase()),
        Some(ref s) if matches!(s.as_str(), "on" | "1" | "true" | "yes")
    )
}

/// Resolve the cross-encoder model to load, honoring `SEMANTEX_RERANKER_MODEL`.
/// Falls back to `BGERerankerV2M3` (the v0.3 default) on missing or unknown values.
#[must_use]
pub fn select_model_from_env() -> RerankerModel {
    let raw = std::env::var(ENV_MODEL).unwrap_or_default();
    match raw.to_ascii_lowercase().as_str() {
        "" | "bge-reranker-v2-m3" | "bge-v2-m3" | "bge-v2" | "default" => {
            RerankerModel::BGERerankerV2M3
        }
        "bge-reranker-base" | "bge-base" => RerankerModel::BGERerankerBase,
        "jina-reranker-v1-turbo-en" | "jina-v1" | "jina-v1-turbo" => {
            RerankerModel::JINARerankerV1TurboEn
        }
        "jina-reranker-v2-base-multilingual" | "jina-v2" | "jina-v2-base" => {
            RerankerModel::JINARerankerV2BaseMultiligual
        }
        other => {
            tracing::warn!(
                model = other,
                "Unknown {ENV_MODEL} value; falling back to bge-reranker-v2-m3"
            );
            RerankerModel::BGERerankerV2M3
        }
    }
}

/// Resolve the on-disk cache directory for downloaded reranker weights.
///
/// Honors `FASTEMBED_CACHE_DIR` if set (fastembed's native env var); otherwise
/// uses `~/.fastembed_cache` per the v0.3 spec. Falls back to fastembed's
/// default (`.fastembed_cache` in cwd) if the home directory cannot be located.
#[must_use]
pub fn cache_dir() -> PathBuf {
    if let Ok(v) = std::env::var("FASTEMBED_CACHE_DIR") {
        return PathBuf::from(v);
    }
    dirs::home_dir().map_or_else(
        || PathBuf::from(".fastembed_cache"),
        |h| h.join(".fastembed_cache"),
    )
}

/// Cross-encoder reranker using fastembed-rs.
///
/// Instantiate via either `new()` (caller picks the model — used by the
/// legacy code path) or `new_default()` (auto-selects the v0.3 cross-encoder
/// honoring `SEMANTEX_RERANKER_MODEL`). When the `SEMANTEX_RERANKER` master
/// switch is off, `rerank()` becomes an identity pass-through.
pub struct FastembedReranker {
    model: TextRerank,
}

impl FastembedReranker {
    /// Construct a reranker with the specified model and hardware acceleration.
    ///
    /// This path is preserved for the legacy hybrid-search call site and the
    /// `examples/test_reranker.rs` example. New call sites should prefer
    /// `new_default()` so the v0.3 spec-mandated model is used.
    pub fn new(model: RerankerModel, show_download_progress: bool) -> Result<Self> {
        let execution_providers = Self::configure_execution_providers();

        let options = RerankInitOptions::new(model)
            .with_cache_dir(cache_dir())
            .with_show_download_progress(show_download_progress)
            .with_execution_providers(execution_providers);

        let model =
            TextRerank::try_new(options).context("Failed to initialize fastembed reranker")?;

        Ok(Self { model })
    }

    /// Construct a reranker using the v0.3 default model (`bge-reranker-v2-m3`,
    /// overridable via `SEMANTEX_RERANKER_MODEL`).
    ///
    /// Returns an error if the cross-encoder is currently disabled by the
    /// `SEMANTEX_RERANKER` env var — callers should check `reranker_enabled()`
    /// first and only attempt to construct when truly needed, to avoid loading
    /// model weights into RAM unnecessarily.
    pub fn new_default(show_download_progress: bool) -> Result<Self> {
        if !reranker_enabled() {
            anyhow::bail!(
                "Refusing to construct reranker: SEMANTEX_RERANKER is not enabled. \
                 Set SEMANTEX_RERANKER=on to load model weights."
            );
        }
        Self::new(select_model_from_env(), show_download_progress)
    }

    /// Configure execution providers for hardware acceleration.
    ///
    /// CoreML is gated behind `SEMANTEX_COREML=1` (same as the embedder) because
    /// CoreML allocates ~10 GB of persistent ANE/compilation buffers on first
    /// inference, which OOMs machines when the daemon is running. CPU-only
    /// execution uses ~50–200 MB and is safe for always-on daemons.
    #[allow(clippy::vec_init_then_push)] // conditional pushes based on platform/feature flags
    fn configure_execution_providers() -> Vec<ort::ep::ExecutionProviderDispatch> {
        let mut providers = Vec::new();

        #[cfg(target_os = "macos")]
        if std::env::var("SEMANTEX_COREML").is_ok() {
            tracing::debug!("Reranker: CoreML execution provider enabled via SEMANTEX_COREML");
            providers.push(ep::CoreML::default().build());
        } else {
            tracing::debug!(
                "Reranker: CoreML disabled by default (set SEMANTEX_COREML=1 to enable). \
                 CPU-only reranking uses ~50-200 MB vs ~10 GB with CoreML."
            );
        }

        #[cfg(feature = "cuda")]
        {
            providers.push(ep::CUDA::default().build());
        }

        providers.push(ep::CPU::default().build());
        providers
    }

    /// Rerank documents by relevance to the query.
    /// Returns `(original_index, score)` pairs sorted by score descending.
    ///
    /// This is a no-op identity pass-through when `SEMANTEX_RERANKER` is not
    /// enabled, returning the input order with score 0.0. That lets the
    /// hybrid-search caller treat the reranker stage as always-callable without
    /// re-checking the env var.
    pub fn rerank(
        &mut self,
        query: &str,
        documents: &[&str],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        // Master switch: if the reranker is disabled, return identity ordering.
        // Caller (hybrid.rs) treats `Reranked` source as a relevance boost,
        // so we deliberately return score=0.0 to make the no-op visible in logs
        // without re-ordering anything.
        if !reranker_enabled() {
            tracing::debug!(
                "Reranker disabled (SEMANTEX_RERANKER!=on); returning identity ordering"
            );
            let n = documents.len().min(top_k);
            return Ok((0..n).map(|i| (i, 0.0_f32)).collect());
        }

        // Rerank with fastembed - return_documents=false, use default batch size
        let results = self
            .model
            .rerank(query, documents, false, None)
            .context("Fastembed reranking failed")?;

        // Convert to (index, score) pairs and sort by score descending
        let mut scored: Vec<(usize, f32)> = results.iter().map(|r| (r.index, r.score)).collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-k
        scored.truncate(top_k);
        Ok(scored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper for tests that need to scrub & restore env state without
    /// leaking across tests. Tests in the same module run in parallel, so
    /// each helper-using test must hold the env mutex while it manipulates
    /// vars.
    fn with_env<F: FnOnce()>(vars: &[(&str, Option<&str>)], f: F) {
        use std::sync::Mutex;
        static ENV_LOCK: Mutex<()> = Mutex::new(());

        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // Snapshot prior values so we can restore them.
        let prior: Vec<(String, Option<String>)> = vars
            .iter()
            .map(|(k, _)| ((*k).to_string(), std::env::var(*k).ok()))
            .collect();

        // SAFETY: env vars are guarded by ENV_LOCK above.
        unsafe {
            for (k, v) in vars {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }

        // Run the closure; on panic we still restore.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));

        // SAFETY: env vars are guarded by ENV_LOCK above.
        unsafe {
            for (k, v) in &prior {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }

        if let Err(e) = result {
            std::panic::resume_unwind(e);
        }
    }

    #[test]
    fn enabled_only_for_truthy_values() {
        with_env(&[(ENV_ENABLE, Some("on"))], || {
            assert!(reranker_enabled());
        });
        with_env(&[(ENV_ENABLE, Some("ON"))], || {
            assert!(reranker_enabled());
        });
        with_env(&[(ENV_ENABLE, Some("1"))], || {
            assert!(reranker_enabled());
        });
        with_env(&[(ENV_ENABLE, Some("true"))], || {
            assert!(reranker_enabled());
        });
        with_env(&[(ENV_ENABLE, Some("True"))], || {
            assert!(reranker_enabled());
        });
        with_env(&[(ENV_ENABLE, Some("yes"))], || {
            assert!(reranker_enabled());
        });

        with_env(&[(ENV_ENABLE, Some("off"))], || {
            assert!(!reranker_enabled());
        });
        with_env(&[(ENV_ENABLE, Some("0"))], || {
            assert!(!reranker_enabled());
        });
        with_env(&[(ENV_ENABLE, Some("false"))], || {
            assert!(!reranker_enabled());
        });
        with_env(&[(ENV_ENABLE, Some(""))], || {
            assert!(!reranker_enabled());
        });
        with_env(&[(ENV_ENABLE, None)], || {
            assert!(!reranker_enabled());
        });
    }

    #[test]
    fn model_selection_defaults_to_bge_v2_m3() {
        with_env(&[(ENV_MODEL, None)], || {
            assert_eq!(select_model_from_env(), RerankerModel::BGERerankerV2M3);
        });
        with_env(&[(ENV_MODEL, Some(""))], || {
            assert_eq!(select_model_from_env(), RerankerModel::BGERerankerV2M3);
        });
        with_env(&[(ENV_MODEL, Some("default"))], || {
            assert_eq!(select_model_from_env(), RerankerModel::BGERerankerV2M3);
        });
        with_env(&[(ENV_MODEL, Some("garbage-not-a-model"))], || {
            // Unknown values warn-and-fall-back; they don't error.
            assert_eq!(select_model_from_env(), RerankerModel::BGERerankerV2M3);
        });
    }

    #[test]
    fn model_selection_recognizes_explicit_choices() {
        with_env(&[(ENV_MODEL, Some("bge-reranker-v2-m3"))], || {
            assert_eq!(select_model_from_env(), RerankerModel::BGERerankerV2M3);
        });
        with_env(&[(ENV_MODEL, Some("BGE-V2-M3"))], || {
            assert_eq!(select_model_from_env(), RerankerModel::BGERerankerV2M3);
        });
        with_env(&[(ENV_MODEL, Some("bge-reranker-base"))], || {
            assert_eq!(select_model_from_env(), RerankerModel::BGERerankerBase);
        });
        with_env(&[(ENV_MODEL, Some("jina-reranker-v1-turbo-en"))], || {
            assert_eq!(
                select_model_from_env(),
                RerankerModel::JINARerankerV1TurboEn
            );
        });
        with_env(&[(ENV_MODEL, Some("jina-v2"))], || {
            assert_eq!(
                select_model_from_env(),
                RerankerModel::JINARerankerV2BaseMultiligual
            );
        });
    }

    #[test]
    fn cache_dir_resolves_to_home_or_override() {
        with_env(
            &[("FASTEMBED_CACHE_DIR", Some("/tmp/semantex-test-cache"))],
            || {
                assert_eq!(cache_dir(), PathBuf::from("/tmp/semantex-test-cache"));
            },
        );
        with_env(&[("FASTEMBED_CACHE_DIR", None)], || {
            let p = cache_dir();
            // Must not be an absolute hard-coded user path; must end with .fastembed_cache.
            assert!(p.ends_with(".fastembed_cache"), "got {p:?}");
        });
    }

    #[test]
    fn new_default_refuses_when_disabled() {
        with_env(&[(ENV_ENABLE, Some("off")), (ENV_MODEL, None)], || {
            // We can't call `.unwrap_err()` here because `FastembedReranker`
            // does not implement `Debug` (the inner `TextRerank` doesn't),
            // and `unwrap_err` requires `T: Debug`. Match instead.
            match FastembedReranker::new_default(false) {
                Ok(_) => panic!("new_default must not load weights when disabled"),
                Err(e) => {
                    let msg = format!("{e}");
                    assert!(
                        msg.contains("SEMANTEX_RERANKER"),
                        "error must point at the env var; got: {msg}"
                    );
                }
            }
        });
    }

    /// Integration test — exercises the actual model download + inference.
    ///
    /// Requires network access on first run (to fetch the ONNX weights into
    /// `~/.fastembed_cache/`). Subsequent runs use the local cache. Gated with
    /// `#[ignore]` so CI doesn't accidentally pull ~600 MB of weights.
    ///
    /// Run manually with:
    ///   SEMANTEX_RERANKER=on cargo test -p semantex-core \
    ///     -- --ignored fastembed_reranker::tests::reranks_when_enabled
    #[test]
    #[ignore]
    fn reranks_when_enabled() {
        with_env(&[(ENV_ENABLE, Some("on")), (ENV_MODEL, None)], || {
            let mut reranker =
                FastembedReranker::new_default(false).expect("model load failed (offline?)");
            let docs = [
                "The giant panda is a bear endemic to China.",
                "Binary search is an efficient algorithm for finding an item in a sorted slice.",
                "Pizza is a popular Italian dish.",
                "Rust's slice::binary_search returns the index of a matching element.",
            ];
            let docs_ref: Vec<&str> = docs.iter().copied().collect();
            let results = reranker
                .rerank("how does binary search work in Rust?", &docs_ref, 4)
                .expect("rerank failed");
            assert!(!results.is_empty());
            // Top result should be one of the on-topic documents (index 1 or 3).
            let top_idx = results[0].0;
            assert!(
                top_idx == 1 || top_idx == 3,
                "expected on-topic doc at top, got idx {top_idx}"
            );
        });
    }

    #[test]
    fn rerank_is_identity_when_disabled() {
        // We don't construct a real model here because that would download
        // weights; instead we test the env-gate logic by exercising the
        // public `reranker_enabled()` check used at the top of rerank().
        with_env(&[(ENV_ENABLE, None)], || {
            assert!(!reranker_enabled());
        });
        with_env(&[(ENV_ENABLE, Some("off"))], || {
            assert!(!reranker_enabled());
        });
    }
}
