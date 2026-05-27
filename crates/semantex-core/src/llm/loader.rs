// crates/semantex-core/src/llm/loader.rs
// ONNX-backed local-LLM loader (v0.6 Item 9, behind `--features local-llm`).
//
// Loads a quantized small-model ONNX checkpoint from the path supplied via
// the `SEMANTEX_LLM_PATH` env var. No model file is bundled with the
// crate — this is the loader path only (spec §9 R3). When the env var is
// unset or points at a missing/unreadable file, `OnnxLlm::load` returns
// `Ok(None)` and the caller falls back to the keyword classifier without
// the daemon ever attempting to spin up an LLM session.
//
// The full prompt-tokenize-decode pipeline is intentionally stubbed in
// this iteration: the trait impls in `classifier.rs` / `hyde.rs` return
// `Err(_)` when called, which lets the higher-level `classify_with_llm`
// helper exercise its fallback path in tests without requiring an actual
// model artifact. Wiring up the inference loop is follow-up work tracked
// in the v0.6 release notes.

use std::path::PathBuf;

use anyhow::{Context, Result};

/// Environment variable that tells the loader where to find the ONNX
/// model. Intentionally a path (not a HuggingFace handle) so we keep
/// model distribution out of the crate's build graph.
pub const MODEL_PATH_ENV: &str = "SEMANTEX_LLM_PATH";

/// Handle to a loaded local LLM. Holds the ONNX session and any
/// per-session state (tokenizer, sampler config). Cheap to clone via Arc;
/// the daemon shares one instance across all search requests.
///
/// The internals are deliberately minimal in this scaffold iteration —
/// just enough to (a) prove the loader path works when a model is
/// supplied, and (b) let the trait impls compile against a real type.
/// Real inference plumbing will land alongside a model artifact.
#[derive(Clone, Debug)]
pub struct OnnxLlm {
    /// Path the model was loaded from, retained for diagnostics.
    pub model_path: PathBuf,
}

impl OnnxLlm {
    /// Attempt to load the LLM from `SEMANTEX_LLM_PATH`.
    ///
    /// Returns `Ok(None)` when:
    ///   - The env var is unset (the common case in default deployments).
    ///   - The env var points at a file that does not exist.
    ///
    /// Returns `Err` only when the env var is set, the file exists, and
    /// the loader fails to bring up a session — i.e. an operator error
    /// the daemon startup should surface rather than swallow.
    pub fn load() -> Result<Option<Self>> {
        let Some(path) = std::env::var_os(MODEL_PATH_ENV) else {
            return Ok(None);
        };
        let path = PathBuf::from(path);
        if !path.exists() {
            tracing::info!(
                ?path,
                env = MODEL_PATH_ENV,
                "local-llm: model file not found, falling back to keyword classifier",
            );
            return Ok(None);
        }
        // Future work: open an `ort::Session` against `path` with the
        // shared workspace ORT environment (already configured in
        // `crate::embedding::colbert`). For the scaffold we just verify
        // the file is readable.
        let metadata = std::fs::metadata(&path)
            .with_context(|| format!("reading metadata for LLM model at {}", path.display()))?;
        anyhow::ensure!(
            metadata.is_file(),
            "{} is not a regular file",
            path.display()
        );
        tracing::info!(
            ?path,
            size_bytes = metadata.len(),
            "local-llm: model file located (inference pipeline stubbed in scaffold)",
        );
        Ok(Some(Self { model_path: path }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Internal load-from-path helper that mirrors the logic of `load()`
    /// without touching the process env. Exercised here so we can assert
    /// the file-existence and "is a file" branches without racing with
    /// other tests in the same binary.
    fn load_from_path(path: &std::path::Path) -> Result<Option<OnnxLlm>> {
        if !path.exists() {
            return Ok(None);
        }
        let metadata = std::fs::metadata(path)
            .with_context(|| format!("reading metadata for LLM model at {}", path.display()))?;
        anyhow::ensure!(
            metadata.is_file(),
            "{} is not a regular file",
            path.display()
        );
        Ok(Some(OnnxLlm {
            model_path: path.to_path_buf(),
        }))
    }

    #[test]
    fn load_from_path_returns_none_when_missing() {
        let bogus = std::env::temp_dir().join("semantex-llm-does-not-exist-zxq.onnx");
        let result = load_from_path(&bogus).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn load_from_path_returns_handle_when_file_exists() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let result = load_from_path(tmp.path()).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().model_path, tmp.path());
    }

    #[test]
    fn load_from_path_errors_when_path_is_directory() {
        let dir = tempfile::tempdir().unwrap();
        let err = load_from_path(dir.path()).unwrap_err();
        assert!(err.to_string().contains("is not a regular file"));
    }
}
