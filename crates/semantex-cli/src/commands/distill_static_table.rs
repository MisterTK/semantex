//! Hidden dev/eval subcommand: distill a static per-token embedding table
//! from the ColBERT-style document encoder (Ember Plan A, Task 4 — see
//! `docs/superpowers/plans/2026-07-17-ember-plan-a-gate1.md`).
//!
//! Walks each corpus directory with the same file walker and per-file
//! chunker-selection logic the real indexing pipeline uses
//! (`semantex_core::index::builder`, mirrored here), so the token
//! distribution the table is distilled from matches what
//! `StaticTokenEmbedder` (Task 5) will see in production. Feeding whole-file
//! reads through instead would skew distillation toward token contexts
//! (whole-file boundaries, no chunk truncation) that indexing never actually
//! produces.

use anyhow::{Context, Result};
use semantex_core::SemantexConfig;
use semantex_core::embedding::colbert::ColbertEmbedder;
use semantex_core::embedding::model_manager;
use semantex_core::embedding::static_distill::{DocTokenEncoder, distill};
use semantex_core::embedding::static_table::StaticTokenTable;
use std::path::{Path, PathBuf};

/// Batch size for encoder calls during distillation. Not user-tunable (this
/// is internal dev tooling) — 32 matches the scale other CPU ONNX batch
/// defaults use elsewhere in this codebase, bounding per-call memory while
/// amortizing session call overhead.
const DISTILL_BATCH: usize = 32;

/// `config` is the SAME `SemantexConfig` every other CLI command uses — built
/// by `main()` via `SemantexConfig::load(Some(&cli.path))` (default CWD),
/// which overlays `.semantexrc.yaml` and env var overrides
/// (`SEMANTEX_MAX_FILE_SIZE`, `SEMANTEX_MODEL_DIR`, ...) on top of defaults.
/// Passing it through (rather than reconstructing `SemantexConfig::default()`
/// here) keeps chunking/walking parity with what `semantex index` would
/// actually do to these files — a `.semantexrc.yaml` with non-default
/// `chunk_size`/`chunk_overlap`, or `SEMANTEX_MODEL_DIR` on the box, must be
/// honored the same way in both places or the distilled table's token
/// distribution silently stops matching production.
///
/// With multiple `--corpus` dirs, this is inherently one config for
/// potentially several directories with their own (possibly differing)
/// `.semantexrc.yaml` files — there's no single right resolution short of
/// walking each dir under its own config. We accept that ambiguity and use
/// the CWD/`--path`-resolved config for all of them, same as every other
/// command that isn't itself per-directory-scoped; this is still a strict
/// improvement over ignoring project config and env overrides entirely.
pub fn run(
    corpus_dirs: &[PathBuf],
    out: &Path,
    verify: bool,
    config: &SemantexConfig,
) -> Result<()> {
    let corpus_texts = crate::commands::distill_corpus::corpus_chunk_texts(corpus_dirs, config)?;

    let models_dir = config.models_dir();
    let colbert_dir = model_manager::ensure_colbert_model(&models_dir)
        .context("failed to provision the ColBERT model used for distillation")?;
    let embedder = ColbertEmbedder::for_indexing(&colbert_dir)
        .context("failed to construct ColbertEmbedder for distillation")?;

    let table = distill(&embedder, corpus_texts, DISTILL_BATCH)?;

    let vocab_size = embedder.vocab_size();
    let seen = (0..vocab_size as u32)
        .filter(|&id| table.lookup(id).is_some())
        .count();
    println!("{seen} of {vocab_size} vocab tokens seen");

    // Coverage against raw vocab undercounts quality when the tokenizer
    // lowercases document input (see `tokenizer_reachable_vocab_count`):
    // uppercase-containing tokens can never be produced from lowered text,
    // so they can never be "seen" regardless of corpus size. Report against
    // the reachable subset too so a low raw-vocab percentage isn't
    // misread as poor corpus coverage.
    let reachable = embedder
        .tokenizer_reachable_vocab_count()
        .unwrap_or(vocab_size);
    // Vocab-sized counts (tens of thousands at most) are exact in f64's
    // 52-bit mantissa; this is a human-readable percentage, not an exact
    // count, so the theoretical precision loss at usize::MAX scale is moot.
    #[allow(clippy::cast_precision_loss)]
    let pct = 100.0 * seen as f64 / reachable.max(1) as f64;
    println!(
        "coverage vs reachable vocab: {seen} of {reachable} ({pct:.1}%) — \
         reachable excludes tokens unproducible under the tokenizer's lowercasing"
    );

    table
        .save(out)
        .with_context(|| format!("failed to save static token table to {}", out.display()))?;
    println!("Saved static token table to {}", out.display());

    if verify {
        let loaded = StaticTokenTable::load(out)
            .with_context(|| format!("--verify: failed to reload {}", out.display()))?;
        println!("--verify: loaded table dims={}", loaded.dims);
    }

    Ok(())
}
