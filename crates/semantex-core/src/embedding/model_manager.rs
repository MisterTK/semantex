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

/// GitHub release that hosts the four distributed encoder-free Ember/Cinder
/// artifacts. Base for [`ember_cinder_artifact_url`].
const EMBER_CINDER_ARTIFACT_BASE_URL: &str =
    "https://github.com/MisterTK/semantex/releases/download";

/// Release tag the Ember/Cinder artifacts are pinned to.
///
/// Deliberately a SPECIFIC historical tag — NOT the current crate version and
/// NOT `latest`. These four files (`static_token_table.bin`,
/// `frozen_centroids.npy`, `cinder_mixer.bin`, `cinder_shortlists.bin`) are
/// trained OFFLINE and change far more rarely than the crate cuts releases.
/// Pinning to a fixed tag means every future crate version keeps pulling the
/// SAME known-good artifacts from the release that first published them, with no
/// need to re-attach them to every subsequent tag — and without the instability
/// of `latest`, which would silently change out from under already-installed
/// clients (and would break reproducibility of an index built at a given tag).
///
/// # Maintainer note — if these artifacts are ever retrained
///
/// The on-disk filenames MUST stay the same (they are what the loaders resolve).
/// To redistribute a NEW training, do ONE of:
///   1. **Preferred (drop-in):** re-upload the new files as assets on the SAME
///      `v1.1.0` release. No code change; every deployed client keeps working
///      and transparently fetches the new bytes. Use this when the retrained
///      artifacts are format/compatibility-equivalent replacements.
///   2. Cut a new release, attach the new artifacts there, and bump this
///      constant to that tag. Use this when old and new clients must diverge
///      (e.g. a breaking artifact-format change) — it ships in a new crate
///      version, so only clients built after the bump fetch the new tag.
const EMBER_CINDER_ARTIFACT_RELEASE_TAG: &str = "v1.1.0";

/// The four encoder-free artifacts distributed as release assets. Order is
/// download order; each is skipped if already present (see
/// [`ensure_ember_cinder_artifacts`]).
const EMBER_CINDER_ARTIFACT_FILES: &[&str] = &[
    STATIC_TOKEN_TABLE_FILE,
    FROZEN_CENTROIDS_FILE,
    CINDER_MIXER_FILE,
    CINDER_SHORTLISTS_FILE,
];

/// Build the pinned GitHub-release download URL for one Ember/Cinder artifact.
fn ember_cinder_artifact_url(file_name: &str) -> String {
    format!("{EMBER_CINDER_ARTIFACT_BASE_URL}/{EMBER_CINDER_ARTIFACT_RELEASE_TAG}/{file_name}")
}

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

/// Download the four encoder-free Ember/Cinder artifacts into `model_dir` if
/// not already present, mirroring [`ensure_colbert_model`]'s per-file
/// "skip if exists" + atomic [`download_file`] pattern exactly.
///
/// `model_dir` is the resolved LateOn-Code-edge model dir (the same dir
/// [`ensure_colbert_model`] returns / [`colbert_model_dir`] resolves) — the very
/// dir the loaders (`CinderEncoder::new`, `StaticTokenEmbedder`,
/// [`frozen_centroids_path`]) read these files from. Call this AFTER
/// `ensure_colbert_model`, which provisions the ColBERT/tokenizer files that
/// share the same dir.
///
/// The files are pulled from the pinned [`EMBER_CINDER_ARTIFACT_RELEASE_TAG`]
/// release (see that constant for why the tag is pinned and how to move the
/// artifacts if they are ever retrained).
///
/// # Errors
///
/// Returns the first per-file download error (all-or-nothing, exactly like
/// `ensure_colbert_model`). This is intentionally NOT swallowed here so the
/// primitive stays honest; the Cinder build path, which MUST stay non-fatal (a
/// failed download has to fall through to the existing tier chain, never fail a
/// build), log-and-continues instead of propagating — see
/// `search::colbert_plaid_backend::build_streaming_ids`.
pub fn ensure_ember_cinder_artifacts(model_dir: &Path) -> Result<()> {
    fs::create_dir_all(model_dir)
        .with_context(|| format!("Failed to create model dir: {}", model_dir.display()))?;
    for file_name in EMBER_CINDER_ARTIFACT_FILES {
        let dest = model_dir.join(file_name);
        if dest.exists() {
            continue;
        }
        tracing::info!(
            "Downloading Ember/Cinder artifact {file_name} from release \
             {EMBER_CINDER_ARTIFACT_RELEASE_TAG}..."
        );
        let url = ember_cinder_artifact_url(file_name);
        download_file(&url, &dest).with_context(|| {
            format!(
                "Failed to download {file_name} from release {EMBER_CINDER_ARTIFACT_RELEASE_TAG}"
            )
        })?;
    }
    Ok(())
}

/// Check whether all four Ember/Cinder artifacts are already present in
/// `model_dir` (mirrors [`is_colbert_downloaded`]). `model_dir` is the resolved
/// LateOn-Code-edge dir, not the `models_dir` root.
#[must_use]
pub fn are_ember_cinder_artifacts_present(model_dir: &Path) -> bool {
    EMBER_CINDER_ARTIFACT_FILES
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ember_cinder_artifact_url_is_pinned_to_release_tag() {
        // Guards the exact distribution URL: base host, the PINNED tag (not the
        // crate version, not `latest`), and the join. A typo here silently
        // 404s → Cinder falls back → the default-on flip becomes a no-op.
        assert_eq!(
            ember_cinder_artifact_url("cinder_mixer.bin"),
            "https://github.com/MisterTK/semantex/releases/download/v1.1.0/cinder_mixer.bin"
        );
        // All four distributed filenames route through the same pinned tag.
        for f in EMBER_CINDER_ARTIFACT_FILES {
            assert!(
                ember_cinder_artifact_url(f)
                    .starts_with("https://github.com/MisterTK/semantex/releases/download/v1.1.0/"),
                "{f} must resolve under the pinned release tag"
            );
        }
    }

    #[test]
    fn ensure_ember_cinder_artifacts_skips_when_all_present() {
        // The "already present → skip download" path (mirrors is_colbert_downloaded's
        // intent): with all four files on disk, ensure_* must return Ok WITHOUT
        // any network access. If it tried to download, download_file would hit the
        // network (and fail offline / in CI); the fact that it returns Ok proves
        // every file was skipped.
        let tmp = tempfile::TempDir::new().unwrap();
        for f in EMBER_CINDER_ARTIFACT_FILES {
            std::fs::write(tmp.path().join(f), b"placeholder").unwrap();
        }
        assert!(are_ember_cinder_artifacts_present(tmp.path()));
        ensure_ember_cinder_artifacts(tmp.path())
            .expect("all artifacts present → no download attempted → Ok");
    }

    #[test]
    fn are_ember_cinder_artifacts_present_requires_all_four() {
        let tmp = tempfile::TempDir::new().unwrap();
        assert!(
            !are_ember_cinder_artifacts_present(tmp.path()),
            "none present"
        );
        // Present only after the LAST of the four lands.
        for (i, f) in EMBER_CINDER_ARTIFACT_FILES.iter().enumerate() {
            std::fs::write(tmp.path().join(f), b"x").unwrap();
            let expect_all = i + 1 == EMBER_CINDER_ARTIFACT_FILES.len();
            assert_eq!(
                are_ember_cinder_artifacts_present(tmp.path()),
                expect_all,
                "presence must require ALL four (after {} of {})",
                i + 1,
                EMBER_CINDER_ARTIFACT_FILES.len()
            );
        }
    }

    // NOTE: the actual network download path (ensure_ember_cinder_artifacts
    // fetching a MISSING file) cannot be unit-tested without network access AND
    // the release assets existing. It is verified end-to-end only after the
    // v1.1.0 release actually hosts the four files as assets — see the report's
    // "manual verification plan". The URL-construction test above locks down the
    // one part that IS statically checkable (the pinned target).
}
