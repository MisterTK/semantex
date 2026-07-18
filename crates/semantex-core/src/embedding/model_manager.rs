//! Shared model-download helper. The per-model provisioning lives in each
//! model's own module (`single_vector_model::ensure_coderank_model`,
//! `search/reranker_download.rs`); this module exposes the atomic
//! `download_file` primitive they (and `runtime_manager`) reuse.

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const LATEON_CODE_EDGE_BASE_URL: &str =
    "https://huggingface.co/lightonai/LateOn-Code-edge/resolve/main";
const LATEON_CODE_EDGE_FILES: &[&str] = &["model_int8.onnx", "tokenizer.json", "onnx_config.json"];
const LATEON_CODE_EDGE_DIR: &str = "LateOn-Code-edge";

/// Filename of the Ember Tier-0 static token table inside a model dir.
pub const STATIC_TOKEN_TABLE_FILE: &str = "static_token_table.bin";
/// Filename of the Ember Plan-B frozen universal centroids inside a model dir.
pub const FROZEN_CENTROIDS_FILE: &str = "frozen_centroids.npy";
/// Filename of the Cinder micro-mixer inside a model dir.
pub const CINDER_MIXER_FILE: &str = "cinder_mixer.bin";
/// Filename of the Cinder centroid shortlists (SXCS) inside a model dir.
pub const CINDER_SHORTLISTS_FILE: &str = "cinder_shortlists.bin";

/// Resolve the Ember Tier-0 static token table's path inside `model_dir`.
pub fn static_token_table_path(model_dir: &Path) -> PathBuf {
    model_dir.join(STATIC_TOKEN_TABLE_FILE)
}

/// Resolve the Ember Plan-B frozen universal centroids' path inside `model_dir`.
pub fn frozen_centroids_path(model_dir: &Path) -> PathBuf {
    model_dir.join(FROZEN_CENTROIDS_FILE)
}

/// Resolve the Cinder micro-mixer's path inside `model_dir`.
pub fn cinder_mixer_path(model_dir: &Path) -> PathBuf {
    model_dir.join(CINDER_MIXER_FILE)
}

/// Resolve the Cinder centroid shortlists' path inside `model_dir`.
pub fn cinder_shortlists_path(model_dir: &Path) -> PathBuf {
    model_dir.join(CINDER_SHORTLISTS_FILE)
}

/// Resolve the LateOn-Code-edge model directory under `models_dir` WITHOUT
/// provisioning it. This is the `<models_dir>/LateOn-Code-edge` subdir where
/// the ColBERT files AND all Ember/Cinder artifacts (`static_token_table.bin`,
/// `frozen_centroids.npy`, `cinder_mixer.bin`, `cinder_shortlists.bin`) live —
/// the same dir [`ensure_colbert_model`] returns, but with no download side
/// effect. Offline artifact tooling (`derive-shortlists`) that only reads
/// already-distilled tables should use this rather than `models_dir` directly:
/// the artifacts are in the subdir, not at the `models_dir` root (which is the
/// mistake this helper exists to prevent).
#[must_use]
pub fn colbert_model_dir(models_dir: &Path) -> PathBuf {
    models_dir.join(LATEON_CODE_EDGE_DIR)
}

/// Download LateOn-Code-edge ColBERT model if not already cached. Provisioning for
/// the opt-in `lateon-colbert` late-interaction backend (the only model whose
/// download lives here rather than in its own module — it has no separate
/// `*_model.rs`). Uses the shared [`download_file`] primitive.
#[allow(clippy::similar_names)]
pub fn ensure_colbert_model(models_dir: &Path) -> Result<PathBuf> {
    let model_dir = models_dir.join(LATEON_CODE_EDGE_DIR);
    if model_dir.join("model_int8.onnx").exists() {
        return Ok(model_dir);
    }
    fs::create_dir_all(&model_dir)
        .with_context(|| format!("Failed to create model dir: {}", model_dir.display()))?;
    tracing::info!("Downloading LateOn-Code-edge ColBERT model (~17MB)...");
    for file_name in LATEON_CODE_EDGE_FILES {
        let url = format!("{LATEON_CODE_EDGE_BASE_URL}/{file_name}");
        let dest = model_dir.join(file_name);
        if !dest.exists() {
            download_file(&url, &dest)
                .with_context(|| format!("Failed to download {file_name} for LateOn-Code-edge"))?;
        }
    }
    Ok(model_dir)
}

/// Check if the ColBERT model is already downloaded.
#[allow(clippy::similar_names)]
pub fn is_colbert_downloaded(models_dir: &Path) -> bool {
    let model_dir = models_dir.join(LATEON_CODE_EDGE_DIR);
    LATEON_CODE_EDGE_FILES
        .iter()
        .all(|f| model_dir.join(f).exists())
}

/// Download `url` to `dest` atomically (temp file + rename), showing a progress
/// bar. Shared with `runtime_manager` for fetching the ONNX Runtime archive and
/// with the per-model downloaders.
pub(crate) fn download_file(url: &str, dest: &Path) -> Result<()> {
    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("HTTP GET failed for {url}"))?;

    let total_size: u64 = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .expect("valid progress template")
            .progress_chars("#>-"),
    );

    let file_name = dest
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    pb.set_message(file_name);

    // Write to a temp file first, then rename for atomicity
    let parent = dest.parent().context("dest has no parent directory")?;
    let tmp_path = parent.join(format!(
        ".tmp_{}",
        dest.file_name()
            .map_or_else(|| "download".into(), |n| n.to_string_lossy().to_string())
    ));

    let mut tmp_file = fs::File::create(&tmp_path)
        .with_context(|| format!("Failed to create {}", tmp_path.display()))?;

    let mut reader = resp.into_body().into_reader();
    let mut buf = [0u8; 8192];
    loop {
        let n = reader
            .read(&mut buf)
            .context("Failed to read response body")?;
        if n == 0 {
            break;
        }
        tmp_file.write_all(&buf[..n])?;
        pb.inc(n as u64);
    }

    tmp_file.flush()?;
    drop(tmp_file);

    fs::rename(&tmp_path, dest).with_context(|| {
        format!(
            "Failed to rename {} -> {}",
            tmp_path.display(),
            dest.display()
        )
    })?;

    pb.finish_with_message("done");
    Ok(())
}
