//! Shared corpus walk + chunk logic for the hidden `distill-static-table`
//! and `distill-centroids` dev/eval subcommands (Ember Plan A Task 4 /
//! Plan B Task 4).
//!
//! Extracted verbatim from `distill_static_table.rs` so both distillation
//! paths walk corpora and select chunkers identically to the real indexing
//! pipeline (`semantex_core::index::builder`) — see the module doc on
//! `distill_static_table` for why that parity matters.

use anyhow::{Context, Result};
use semantex_core::SemantexConfig;
use semantex_core::chunking::Chunker;
use semantex_core::chunking::ast_chunker::AstChunker;
use semantex_core::chunking::pdf_chunker::PdfChunker;
use semantex_core::chunking::text_chunker::TextChunker;
use semantex_core::file::detector::FileType;
use semantex_core::file::walker::FileWalker;
use std::path::{Path, PathBuf};

/// Walk every `corpus_dirs` entry with the same file walker the real
/// indexing pipeline uses, then lazily chunk each discovered file with the
/// same per-file chunker-selection logic (`chunk_texts`), returning a flat
/// iterator of chunk texts across all directories.
///
/// `config` governs walking limits (`max_file_size`, `max_file_count`) and
/// chunking parameters (`chunk_size`, `chunk_overlap`) — see the
/// `distill_static_table::run` doc comment for why callers must pass the
/// SAME loaded `SemantexConfig` every other CLI command uses rather than a
/// freshly constructed default.
///
/// # Errors
///
/// Fails fast (before any model provisioning) if any `--corpus` dir does not
/// exist/canonicalize, if walking a dir fails, or if the combined walk across
/// all dirs finds zero indexable files.
pub fn corpus_chunk_texts(
    corpus_dirs: &[PathBuf],
    config: &SemantexConfig,
) -> Result<impl Iterator<Item = String>> {
    anyhow::ensure!(
        !corpus_dirs.is_empty(),
        "at least one --corpus <dir> is required"
    );

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

    let ast_chunker = AstChunker::new(config.chunk_size, config.chunk_overlap);
    let text_chunker = TextChunker::new(config.chunk_size, config.chunk_overlap);
    let pdf_chunker = PdfChunker::new(config.chunk_size, config.chunk_overlap);

    Ok(files.into_iter().flat_map(move |file_path| {
        chunk_texts(&file_path, &ast_chunker, &text_chunker, &pdf_chunker)
    }))
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
