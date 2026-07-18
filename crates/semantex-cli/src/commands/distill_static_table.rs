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
use semantex_core::chunking::Chunker;
use semantex_core::chunking::ast_chunker::AstChunker;
use semantex_core::chunking::pdf_chunker::PdfChunker;
use semantex_core::chunking::text_chunker::TextChunker;
use semantex_core::embedding::colbert::ColbertEmbedder;
use semantex_core::embedding::model_manager;
use semantex_core::embedding::static_distill::{DocTokenEncoder, distill};
use semantex_core::embedding::static_table::StaticTokenTable;
use semantex_core::file::detector::FileType;
use semantex_core::file::walker::FileWalker;
use std::path::{Path, PathBuf};

/// Batch size for encoder calls during distillation. Not user-tunable (this
/// is internal dev tooling) — 32 matches the scale other CPU ONNX batch
/// defaults use elsewhere in this codebase, bounding per-call memory while
/// amortizing session call overhead.
const DISTILL_BATCH: usize = 32;

pub fn run(corpus_dirs: &[PathBuf], out: &Path, verify: bool) -> Result<()> {
    anyhow::ensure!(
        !corpus_dirs.is_empty(),
        "at least one --corpus <dir> is required"
    );

    let config = SemantexConfig::default();
    let walker = FileWalker::new(config.max_file_size, config.max_file_count);

    // Walk (and validate) every corpus dir up front, before touching the
    // model — a typo'd path should fail fast without provisioning anything.
    let mut files: Vec<PathBuf> = Vec::new();
    for dir in corpus_dirs {
        let canon = dir
            .canonicalize()
            .with_context(|| format!("corpus dir does not exist: {}", dir.display()))?;
        let found = walker
            .walk(&canon)
            .with_context(|| format!("failed to walk corpus dir: {}", canon.display()))?;
        println!("{}: {} files", canon.display(), found.len());
        files.extend(found);
    }
    anyhow::ensure!(
        !files.is_empty(),
        "no indexable files found under the given --corpus dir(s)"
    );
    println!(
        "Distilling from {} files across {} corpus dir(s)...",
        files.len(),
        corpus_dirs.len()
    );

    let models_dir = config.models_dir();
    let colbert_dir = model_manager::ensure_colbert_model(&models_dir)
        .context("failed to provision the ColBERT model used for distillation")?;
    let embedder = ColbertEmbedder::for_indexing(&colbert_dir)
        .context("failed to construct ColbertEmbedder for distillation")?;

    let ast_chunker = AstChunker::new(config.chunk_size, config.chunk_overlap);
    let text_chunker = TextChunker::new(config.chunk_size, config.chunk_overlap);
    let pdf_chunker = PdfChunker::new(config.chunk_size, config.chunk_overlap);

    let corpus_texts = files.into_iter().flat_map(move |file_path| {
        chunk_texts(&file_path, &ast_chunker, &text_chunker, &pdf_chunker)
    });

    let table = distill(&embedder, corpus_texts, DISTILL_BATCH)?;

    let vocab_size = embedder.vocab_size();
    let seen = (0..vocab_size as u32)
        .filter(|&id| table.lookup(id).is_some())
        .count();
    println!("{seen} of {vocab_size} vocab tokens seen");

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

/// Split one file's content into the chunk texts the real indexing pipeline
/// would embed, mirroring `semantex_core::index::builder`'s per-file
/// chunker-selection logic: the PDF chunker's own reader for PDFs, the AST
/// chunker for languages it supports, the text-window chunker otherwise.
/// Binary files and unreadable text files are skipped, same as indexing's
/// `files_skipped` path. A file whose chunker panics (a known risk with
/// tree-sitter grammars and `pdf_extract` on malformed input) is skipped
/// rather than aborting the whole distillation run.
fn chunk_texts(
    path: &Path,
    ast_chunker: &AstChunker,
    text_chunker: &TextChunker,
    pdf_chunker: &PdfChunker,
) -> Vec<String> {
    let file_type = FileType::detect(path);
    let content = if file_type == FileType::Pdf {
        String::new() // PDF chunker reads the file directly.
    } else if !file_type.is_text() {
        return Vec::new();
    } else {
        match std::fs::read_to_string(path) {
            Ok(c) => c.replace("\r\n", "\n"),
            Err(_) => return Vec::new(),
        }
    };

    let chunked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if file_type == FileType::Pdf {
            pdf_chunker.chunk(path, "")
        } else if file_type.supports_ast() {
            ast_chunker.chunk(path, &content)
        } else {
            text_chunker.chunk(path, &content)
        }
    }));

    match chunked {
        Ok(Ok(chunks)) => chunks.into_iter().map(|c| c.content).collect(),
        Ok(Err(e)) => {
            tracing::warn!("skipping {}: chunker error: {e}", path.display());
            Vec::new()
        }
        Err(_) => {
            tracing::warn!("skipping {}: chunker panicked", path.display());
            Vec::new()
        }
    }
}
