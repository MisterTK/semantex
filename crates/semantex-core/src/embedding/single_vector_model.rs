//! Download-on-first-use provisioning for the CodeRankEmbed single-vector ONNX
//! model. Mirrors `model_manager.rs` (the ColBERT downloader). The HF repo +
//! file list are the values RECORDED in
//! `docs/superpowers/plans/2026-05-31-research-notes.md` (S2 — CodeRankEmbed):
//! the int8 graph ships in ONNX **external-data** format, so the `.onnx` graph
//! AND its co-located `.onnx.data` weights file BOTH must be fetched, alongside
//! the tokenizer + config.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// HF resolve base for the project-hosted CodeRankEmbed int8 export.
/// RECORDED in research-notes (S2): `hf:MisterTK/CodeRankEmbed-onnx-int8` (MIT).
const CODERANK_BASE_URL: &str =
    "https://huggingface.co/MisterTK/CodeRankEmbed-onnx-int8/resolve/main";

/// Files to fetch. RECORDED: the int8 ONNX graph (~1.2 MB) + its external-data
/// weights (`.onnx.data`, ~137 MB — `ort` requires both co-located to load) +
/// the tokenizer + config.
pub(crate) const CODERANK_FILES: &[&str] = &[
    "model_int8.onnx",
    "model_int8.onnx.data",
    "tokenizer.json",
    "config.json",
];

/// On-disk model subdirectory under `models_dir`.
pub(crate) const CODERANK_DIR: &str = "CodeRankEmbed";

/// The ONNX graph leaf the encoder loads (its external-data file sits beside it).
pub(crate) const CODERANK_ONNX: &str = "model_int8.onnx";

/// Download the CodeRankEmbed int8 model if not already cached. Returns the
/// model directory (containing the graph + `.onnx.data` weights + tokenizer).
pub fn ensure_coderank_model(models_dir: &Path) -> Result<PathBuf> {
    let model_dir = models_dir.join(CODERANK_DIR);
    if is_coderank_downloaded(models_dir) {
        return Ok(model_dir);
    }
    fs::create_dir_all(&model_dir)
        .with_context(|| format!("Failed to create model dir: {}", model_dir.display()))?;
    tracing::info!("Downloading CodeRankEmbed single-vector ONNX model...");
    for file_name in CODERANK_FILES {
        let dest = model_dir.join(file_name);
        if !dest.exists() {
            let url = format!("{CODERANK_BASE_URL}/{file_name}");
            crate::embedding::model_manager::download_file(&url, &dest)
                .with_context(|| format!("Failed to download {file_name} for CodeRankEmbed"))?;
        }
    }
    Ok(model_dir)
}

/// True if every CodeRankEmbed file is present under `models_dir`.
pub fn is_coderank_downloaded(models_dir: &Path) -> bool {
    let model_dir = models_dir.join(CODERANK_DIR);
    CODERANK_FILES.iter().all(|f| model_dir.join(f).exists())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coderank_not_downloaded_in_empty_dir() {
        let tmp = tempfile::TempDir::new().unwrap();
        assert!(!is_coderank_downloaded(tmp.path()));
    }

    #[test]
    fn coderank_detected_when_files_present() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path().join(CODERANK_DIR);
        fs::create_dir_all(&dir).unwrap();
        for f in CODERANK_FILES {
            fs::write(dir.join(f), b"stub").unwrap();
        }
        assert!(is_coderank_downloaded(tmp.path()));
    }

    #[test]
    fn coderank_partial_download_is_not_complete() {
        // The external-data `.onnx.data` weights are mandatory: a graph without
        // them must NOT count as downloaded (ort would fail to load).
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path().join(CODERANK_DIR);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("model_int8.onnx"), b"stub").unwrap();
        fs::write(dir.join("tokenizer.json"), b"stub").unwrap();
        fs::write(dir.join("config.json"), b"stub").unwrap();
        // missing model_int8.onnx.data
        assert!(!is_coderank_downloaded(tmp.path()));
    }
}
