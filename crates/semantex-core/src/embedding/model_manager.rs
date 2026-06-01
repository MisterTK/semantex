//! Shared model-download helper. The per-model provisioning lives in each
//! model's own module (`single_vector_model::ensure_coderank_model`,
//! `search/reranker_download.rs`); this module exposes the atomic
//! `download_file` primitive they (and `runtime_manager`) reuse.

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

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
