//! Hidden dev/eval subcommand: train frozen universal PLAID centroids from
//! contextual token embeddings over generic corpora (Ember Plan B). The
//! output artifact ships in the model directory and is consumed by the
//! colbert-plaid builder when SEMANTEX_FROZEN_CENTROIDS=1 — see
//! `search/colbert_plaid_backend.rs`.

use anyhow::{Context, Result};
use semantex_core::SemantexConfig;
use semantex_core::embedding::centroid_train::{
    CentroidTrainOptions, load_centroids_npy, save_centroids_npy, train_centroids,
};
use semantex_core::embedding::colbert::ColbertEmbedder;
use semantex_core::embedding::model_manager;
use std::path::{Path, PathBuf};

/// `config` is the same loaded `SemantexConfig` every other CLI command
/// uses — see the parity note on `distill_static_table::run` for why this
/// (rather than `SemantexConfig::default()`) is required to keep
/// walking/chunking consistent with `semantex index`.
pub fn run(
    corpus_dirs: &[PathBuf],
    out: &Path,
    k: usize,
    sample: usize,
    verify: bool,
    config: &SemantexConfig,
) -> Result<()> {
    let corpus_texts = crate::commands::distill_corpus::corpus_chunk_texts(corpus_dirs, config)?;

    let models_dir = config.models_dir();
    let colbert_dir = model_manager::ensure_colbert_model(&models_dir)
        .context("failed to provision the ColBERT model used for centroid training")?;
    let embedder = ColbertEmbedder::for_indexing(&colbert_dir)
        .context("failed to construct ColbertEmbedder for centroid training")?;

    let opts = CentroidTrainOptions {
        num_centroids: k,
        sample_capacity: sample,
        batch: 32,
    };
    let centroids = train_centroids(&embedder, corpus_texts, &opts)?;
    println!(
        "trained {} centroids × {} dims",
        centroids.nrows(),
        centroids.ncols()
    );

    save_centroids_npy(out, &centroids)
        .with_context(|| format!("failed to save centroids to {}", out.display()))?;
    println!("Saved frozen centroids to {}", out.display());

    if verify {
        let back = load_centroids_npy(out)
            .with_context(|| format!("--verify: failed to reload {}", out.display()))?;
        println!(
            "--verify: loaded centroids shape [{}, {}]",
            back.nrows(),
            back.ncols()
        );
    }

    Ok(())
}
