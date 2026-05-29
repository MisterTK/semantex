use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const LATEON_CODE_EDGE_BASE_URL: &str =
    "https://huggingface.co/lightonai/LateOn-Code-edge/resolve/main";
const LATEON_CODE_EDGE_FILES: &[&str] = &["model_int8.onnx", "tokenizer.json", "onnx_config.json"];
const LATEON_CODE_EDGE_DIR: &str = "LateOn-Code-edge";

/// Download LateOn-Code-edge ColBERT model if not already cached.
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
/// bar. Shared with `runtime_manager` for fetching the ONNX Runtime archive.
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
