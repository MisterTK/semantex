//! Hidden dev/eval subcommand: derive Cinder centroid shortlists (SXCS
//! format, spec §4.1 #4, plan Task 4) from the Ember Plan A static token
//! table and the Ember Plan B frozen universal centroids. No corpus walk —
//! purely a offline transform of two already-distilled artifacts in the
//! model directory. See `embedding/shortlists.rs` for the format and the
//! union-argmax consumer.

use anyhow::{Context, Result};
use semantex_core::SemantexConfig;
use semantex_core::embedding::centroid_train::load_centroids_npy;
use semantex_core::embedding::cinder::CinderEncoder;
use semantex_core::embedding::model_manager;
use semantex_core::embedding::shortlists::{CentroidShortlists, shortlist_agreement};
use semantex_core::embedding::static_table::StaticTokenTable;
use std::path::{Path, PathBuf};

/// Number of `(mixed embedding, window)` samples drawn for the `--agreement`
/// gate-C4 measurement (spec §5 C4). Reservoir-sampled uniformly across the
/// whole `--agreement` corpus.
const AGREEMENT_SAMPLES: usize = 100_000;
/// Fixed seed for the deterministic reservoir sample, so `--agreement` is
/// reproducible run-to-run given the same corpus.
const AGREEMENT_SEED: u64 = 20_260_718;

/// `config` is the same loaded `SemantexConfig` every other CLI command
/// uses — see the parity note on `distill_static_table::run` for why this
/// (rather than `SemantexConfig::default()`) is required, here to resolve
/// `models_dir()` (honors `SEMANTEX_MODEL_DIR` / `.semantexrc.yaml`)
/// consistently with every other command.
///
/// `agreement`, when non-empty, names one or more corpus dirs over which gate
/// C4's shortlist agreement is measured after the shortlists are derived (the
/// fraction of sampled mixed embeddings whose shortlist-union argmax equals the
/// exhaustive argmax over every centroid). This builds a [`CinderEncoder`] from
/// the model dir, so it requires the static table, `cinder_mixer.bin`, and the
/// just-saved `cinder_shortlists.bin` all present there — i.e. run with `--out`
/// pointing at the model dir's standard `cinder_shortlists.bin` path.
// `models_dir` (base) vs `model_dir` (the `LateOn-Code-edge` subdir) are a
// deliberate, meaningful pair — same `#[allow]` `model_manager` uses for it.
#[allow(clippy::similar_names)]
pub fn run(
    out: &Path,
    m: usize,
    verify: bool,
    agreement: &[PathBuf],
    config: &SemantexConfig,
) -> Result<()> {
    let models_dir = config.models_dir();
    // Artifacts live in the `LateOn-Code-edge` subdir under `models_dir` (the
    // dir `ensure_colbert_model` resolves), NOT at the `models_dir` root.
    let model_dir = model_manager::colbert_model_dir(&models_dir);

    let table_path = model_manager::static_token_table_path(&model_dir);
    let table = StaticTokenTable::load(&table_path).with_context(|| {
        format!(
            "failed to load static token table {} — run distill-static-table first",
            table_path.display()
        )
    })?;

    let centroids_path = model_manager::frozen_centroids_path(&model_dir);
    let centroids = load_centroids_npy(&centroids_path).with_context(|| {
        format!(
            "failed to load frozen centroids {} — run distill-centroids first",
            centroids_path.display()
        )
    })?;

    let shortlists = CentroidShortlists::derive(&table, &centroids.view(), m)?;
    println!(
        "derived shortlists m={} vocab={}",
        shortlists.m, shortlists.vocab_size
    );

    shortlists
        .save(out)
        .with_context(|| format!("failed to save shortlists to {}", out.display()))?;
    println!("Saved shortlists to {}", out.display());

    if verify {
        let back = CentroidShortlists::load(out)
            .with_context(|| format!("--verify: failed to reload {}", out.display()))?;
        println!(
            "--verify: loaded shortlists m={} vocab={}",
            back.m, back.vocab_size
        );
    }

    if !agreement.is_empty() {
        // Gate C4: sample mixed embeddings from a real corpus via the Cinder
        // encoder and compare the shortlist-union argmax against the exhaustive
        // argmax. Uses the just-derived (in-memory) `shortlists` + `centroids`
        // for the comparison so the number reflects exactly the artifact we
        // wrote, not whatever the encoder happened to reload.
        let corpus_texts = crate::commands::distill_corpus::corpus_chunk_texts(agreement, config)?;
        let encoder = CinderEncoder::new(&model_dir).with_context(|| {
            format!(
                "--agreement: failed to build CinderEncoder from {} (needs the static table, \
                 cinder_mixer.bin, and the just-saved cinder_shortlists.bin present there)",
                model_dir.display()
            )
        })?;
        let samples = encoder.agreement_samples(corpus_texts, AGREEMENT_SAMPLES, AGREEMENT_SEED)?;
        let agreement_frac = shortlist_agreement(&samples, &shortlists, &centroids.view());
        println!(
            "shortlist_agreement={agreement_frac:.6} samples={} m={}",
            samples.len(),
            shortlists.m
        );
    }

    Ok(())
}
