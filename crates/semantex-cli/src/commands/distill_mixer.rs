//! Hidden dev/eval subcommand: train the Cinder micro-mixer (spec §4.1 #3,
//! plan Task 4) against a teacher ColBERT encoder using STUDENT inputs
//! gathered from the Ember Plan A static token table. The output artifact
//! ships in the model directory alongside `static_token_table.bin` and
//! `frozen_centroids.npy`, and is consumed by the Cinder compiled-index
//! encoder — see `embedding/mixer_train.rs` and `embedding/mixer.rs`.

use anyhow::{Context, Result};
use semantex_core::SemantexConfig;
use semantex_core::embedding::colbert::ColbertEmbedder;
use semantex_core::embedding::mixer::MicroMixer;
use semantex_core::embedding::mixer_train::{MixerTrainOptions, train_mixer};
use semantex_core::embedding::model_manager;
use semantex_core::embedding::static_table::StaticTokenTable;
use std::path::{Path, PathBuf};

/// `config` is the same loaded `SemantexConfig` every other CLI command
/// uses — see the parity note on `distill_static_table::run` for why this
/// (rather than `SemantexConfig::default()`) is required to keep
/// walking/chunking consistent with `semantex index`.
// `models_dir` (base) vs `model_dir` (the `LateOn-Code-edge` subdir) are a
// deliberate, meaningful pair — same `#[allow]` `model_manager` uses for it.
#[allow(clippy::similar_names)]
pub fn run(
    corpus_dirs: &[PathBuf],
    out: &Path,
    sample: usize,
    epochs: usize,
    verify: bool,
    config: &SemantexConfig,
) -> Result<()> {
    let corpus_texts = crate::commands::distill_corpus::corpus_chunk_texts(corpus_dirs, config)?;

    let models_dir = config.models_dir();
    // All model artifacts live in the `LateOn-Code-edge` subdir under
    // `models_dir` (the same dir `ensure_colbert_model` resolves), NOT at the
    // `models_dir` root — resolve it so the table is loaded from where
    // `distill-static-table` actually wrote it.
    let model_dir = model_manager::colbert_model_dir(&models_dir);

    // Fail fast on the cheap, local artifact before provisioning/loading the
    // (much heavier) teacher encoder.
    let table_path = model_manager::static_token_table_path(&model_dir);
    let table = StaticTokenTable::load(&table_path).with_context(|| {
        format!(
            "failed to load static token table {} — run distill-static-table first",
            table_path.display()
        )
    })?;

    let colbert_dir = model_manager::ensure_colbert_model(&models_dir)
        .context("failed to provision the ColBERT model used as the mixer's teacher encoder")?;
    let embedder = ColbertEmbedder::for_indexing(&colbert_dir)
        .context("failed to construct ColbertEmbedder for mixer training")?;

    let opts = MixerTrainOptions {
        sample_capacity: sample,
        epochs,
        ..MixerTrainOptions::default()
    };
    let (mixer, report) = train_mixer(&embedder, &table, corpus_texts, &opts)?;
    println!(
        "holdout cosine: mixer={} linear={} pairs={}",
        report.holdout_cosine_mixer, report.holdout_cosine_linear, report.pairs_trained
    );

    mixer
        .save(out)
        .with_context(|| format!("failed to save mixer to {}", out.display()))?;
    println!("Saved mixer to {}", out.display());

    if verify {
        let back = MicroMixer::load(out)
            .with_context(|| format!("--verify: failed to reload {}", out.display()))?;
        println!("--verify: loaded mixer dims={}", back.dims);
    }

    Ok(())
}
