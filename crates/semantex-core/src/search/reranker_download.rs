//! Download + cache ONNX reranker weights, mirroring
//! `embedding/model_manager::ensure_colbert_model`. Model coordinates are
//! passed in as data (`ModelFiles`) so nothing here is tied to a specific
//! model or path. Coordinates ultimately come from S8's `ModelRegistry`
//! (`ModelSource::Hf`/`Url`) via `RerankerChoice::from_spec`.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Where a reranker's ONNX weights live (HF / mirror) and on disk.
///
/// The FIRST entry of `files` is the download sentinel: if it already exists
/// under `<models_dir>/<subdir>`, the model is considered present and no network
/// is touched. (Exports differ — bge ships `model_int8.onnx`, the hosted
/// Qwen3-Reranker export ships `model.onnx` — so the sentinel is data, not a
/// hardcoded filename.)
#[derive(Debug, Clone)]
pub struct ModelFiles {
    /// On-disk subdirectory name under `models_dir` (e.g. "Qwen3-Reranker-0.6B").
    pub subdir: String,
    /// Resolve base, e.g. "https://huggingface.co/<org>/<repo>/resolve/main".
    pub base_url: String,
    /// Files to fetch; `files[0]` is the download sentinel.
    pub files: Vec<String>,
}

/// Ensure the reranker model's files are present under
/// `<models_dir>/<spec.subdir>`, downloading any missing ones. Returns the
/// model directory. Idempotent: a no-op when the sentinel (`files[0]`) exists.
#[allow(clippy::similar_names)] // model_dir vs models_dir mirrors ensure_colbert_model
pub fn ensure_reranker_model(models_dir: &Path, spec: &ModelFiles) -> Result<PathBuf> {
    let model_dir = models_dir.join(&spec.subdir);
    let sentinel = spec
        .files
        .first()
        .with_context(|| format!("reranker `{}` lists no files", spec.subdir))?;
    if model_dir.join(sentinel).exists() {
        return Ok(model_dir);
    }
    fs::create_dir_all(&model_dir)
        .with_context(|| format!("failed to create reranker dir {}", model_dir.display()))?;
    tracing::info!(
        model = spec.subdir,
        "Downloading ONNX reranker weights (may be several hundred MB)..."
    );
    for file_name in &spec.files {
        let dest = model_dir.join(file_name);
        if dest.exists() {
            continue;
        }
        let url = format!("{}/{file_name}", spec.base_url.trim_end_matches('/'));
        crate::embedding::model_manager::download_file(&url, &dest)
            .with_context(|| format!("failed to download {file_name} for {}", spec.subdir))?;
    }
    Ok(model_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ModelFiles {
        ModelFiles {
            subdir: "Test-Reranker".to_string(),
            base_url: "https://example.invalid/repo/resolve/main".to_string(),
            files: vec!["model_int8.onnx".to_string(), "tokenizer.json".to_string()],
        }
    }

    #[test]
    fn sentinel_short_circuits_download() {
        let tmp = tempfile::TempDir::new().unwrap();
        let s = spec();
        let dir = tmp.path().join(&s.subdir);
        fs::create_dir_all(&dir).unwrap();
        // Sentinel present -> no network, returns dir immediately. (base_url is
        // .invalid, so any download attempt would error; success proves we
        // short-circuited.)
        fs::write(dir.join("model_int8.onnx"), b"stub").unwrap();
        let got = ensure_reranker_model(tmp.path(), &s).expect("short-circuit");
        assert_eq!(got, dir);
    }

    #[test]
    fn missing_sentinel_attempts_download_and_errors_on_bad_url() {
        let tmp = tempfile::TempDir::new().unwrap();
        // No sentinel -> tries to fetch from the .invalid host -> error.
        let err = ensure_reranker_model(tmp.path(), &spec()).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("model_int8.onnx") || msg.contains("Test-Reranker"),
            "error should name the file/model; got: {msg}"
        );
    }
}
