//! Hidden dev/eval subcommand: derive Cinder centroid shortlists (SXCS
//! format, spec §4.1 #4, plan Task 4) from the Ember Plan A static token
//! table and the Ember Plan B frozen universal centroids. No corpus walk —
//! purely a offline transform of two already-distilled artifacts in the
//! model directory. See `embedding/shortlists.rs` for the format and the
//! union-argmax consumer.

use anyhow::{Context, Result};
use semantex_core::SemantexConfig;
use semantex_core::embedding::centroid_train::load_centroids_npy;
use semantex_core::embedding::model_manager;
use semantex_core::embedding::shortlists::CentroidShortlists;
use semantex_core::embedding::static_table::StaticTokenTable;
use std::path::Path;

/// `config` is the same loaded `SemantexConfig` every other CLI command
/// uses — see the parity note on `distill_static_table::run` for why this
/// (rather than `SemantexConfig::default()`) is required, here to resolve
/// `models_dir()` (honors `SEMANTEX_MODEL_DIR` / `.semantexrc.yaml`)
/// consistently with every other command.
pub fn run(out: &Path, m: usize, verify: bool, config: &SemantexConfig) -> Result<()> {
    let models_dir = config.models_dir();

    let table_path = model_manager::static_token_table_path(&models_dir);
    let table = StaticTokenTable::load(&table_path).with_context(|| {
        format!(
            "failed to load static token table {} — run distill-static-table first",
            table_path.display()
        )
    })?;

    let centroids_path = model_manager::frozen_centroids_path(&models_dir);
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

    Ok(())
}
