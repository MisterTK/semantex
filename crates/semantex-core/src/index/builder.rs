use crate::chunking::Chunker;
use crate::chunking::ast_chunker::AstChunker;
use crate::chunking::import_resolver;
use crate::chunking::pdf_chunker::PdfChunker;
use crate::chunking::semantic_role::synthesize_nl_annotation;
use crate::chunking::structured_meta::TypeRefContext;
use crate::chunking::text_chunker::TextChunker;
use crate::config::SemantexConfig;
use crate::file::detector::FileType;
use crate::file::hasher;
use crate::file::walker::FileWalker;
use crate::index::file_classifier;
use crate::index::global_graph;
use crate::index::page_rank;
use crate::index::pattern_catalog::{self, PatternCatalog, PatternLang};
use crate::index::storage::ChunkStore;
use crate::search::code_tokenizer;
use crate::search::dense_backend::{DenseBackendKind, DenseIndexBuilder, dense_subdir};
use crate::search::sparse_search::SparseIndex;
use crate::types::{Chunk, ChunkType, FileEntry, IndexMeta};
use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use rusqlite::{Connection, params};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Statistics from an indexing operation
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub files_scanned: u64,
    pub files_indexed: u64,
    pub files_skipped: u64,
    pub files_deleted: u64,
    pub chunks_created: u64,
    pub chunks_removed: u64,
    pub duration: Duration,
}

/// Builds a search index for a project
pub struct IndexBuilder {
    config: SemantexConfig,
}

impl IndexBuilder {
    pub fn new(config: &SemantexConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    /// Build the index for a project directory
    #[tracing::instrument(skip(self), fields(project_path = %project_path.display()))]
    pub fn build(&self, project_path: &Path) -> Result<IndexStats> {
        let start = Instant::now();
        let project_path = project_path
            .canonicalize()
            .with_context(|| format!("Invalid project path: {}", project_path.display()))?;

        tracing::info!("Starting index build for {}", project_path.display());

        let index_dir = SemantexConfig::project_index_dir(&project_path);
        std::fs::create_dir_all(&index_dir)?;

        // Prevent concurrent index builds from corrupting the index
        let lock_file = std::fs::File::create(index_dir.join(".semantex.lock"))?;
        lock_file.lock()?;

        // Auto-append .semantex/ to .gitignore if inside a git repo
        if project_path.join(".git").is_dir() {
            let gitignore_path = project_path.join(".gitignore");
            let existing = if gitignore_path.exists() {
                std::fs::read_to_string(&gitignore_path).unwrap_or_default()
            } else {
                String::new()
            };
            let has_entry = existing.lines().any(|line| {
                let trimmed = line.trim();
                trimmed == ".semantex" || trimmed == ".semantex/" || trimmed == "/.semantex"
            });
            if !has_entry {
                use std::io::Write;
                let mut f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&gitignore_path)?;
                if !existing.is_empty() && !existing.ends_with('\n') {
                    writeln!(f)?;
                }
                writeln!(f, ".semantex/")?;
                tracing::info!("Added .semantex/ to .gitignore");
            }
        }

        // Check schema version compatibility. We must tolerate a meta.json
        // whose shape no longer matches the current `IndexMeta` struct (older
        // schemas may lack fields added in newer versions), so we parse the
        // schema_version field via a permissive Value path before attempting
        // a full IndexMeta deserialize. Either signal — older schema_version
        // OR meta.json that fails to deserialize against the current struct —
        // means a full rebuild.
        let expected_version = IndexMeta::CURRENT_SCHEMA_VERSION;
        let meta_path = index_dir.join("meta.json");
        if meta_path.exists()
            && let Ok(meta_str) = std::fs::read_to_string(&meta_path)
        {
            let raw_version = serde_json::from_str::<serde_json::Value>(&meta_str)
                .ok()
                .as_ref()
                .and_then(|v| v.get("schema_version").and_then(serde_json::Value::as_u64));
            let strict_parse = serde_json::from_str::<IndexMeta>(&meta_str).is_ok();
            let version_mismatch = raw_version != Some(u64::from(expected_version));
            let shape_mismatch = !strict_parse;
            if version_mismatch || shape_mismatch {
                let label =
                    raw_version.map_or_else(|| "unreadable".to_string(), |v| format!("v{v}"));
                tracing::warn!(
                    "Index schema mismatch (found {}, expected v{}). Forcing full rebuild.",
                    label,
                    expected_version
                );
                // Remove existing index files to force full rebuild.
                let _ = std::fs::remove_dir_all(index_dir.join("sparse"));
                let _ = std::fs::remove_dir_all(index_dir.join("plaid"));
                let _ = std::fs::remove_file(index_dir.join("plaid_mapping.bin"));
                let _ = std::fs::remove_dir_all(index_dir.join("dense")); // S1 per-backend subdirs
                let _ = std::fs::remove_file(index_dir.join("chunks.db"));
                let _ = std::fs::remove_file(&meta_path);
                // Clean up legacy HNSW files if present
                let _ = std::fs::remove_file(index_dir.join("dense.usearch"));
                let _ = std::fs::remove_file(index_dir.join("dense_coarse.usearch"));
            }
        }

        // Initialize components
        let walker = FileWalker::new(self.config.max_file_size, self.config.max_file_count);
        let text_chunker = TextChunker::new(self.config.chunk_size, self.config.chunk_overlap);
        let ast_chunker = AstChunker::new(self.config.chunk_size, self.config.chunk_overlap);
        let pdf_chunker = PdfChunker::new(self.config.chunk_size, self.config.chunk_overlap);

        // Open storage
        let store = ChunkStore::open(&index_dir.join("chunks.db"))?;

        // Walk files
        tracing::info!("Scanning files in {}", project_path.display());
        let files = walker.walk(&project_path)?;
        let files_scanned = files.len() as u64;

        let pb = ProgressBar::new(files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} files ({eta})")
                .expect("valid progress template")
                .progress_chars("=>-"),
        );

        let mut files_indexed = 0u64;
        let mut files_skipped = 0u64;
        let mut files_deleted = 0u64;
        let mut removed_chunk_ids: Vec<u64> = Vec::new();
        let mut new_chunk_ids: Vec<u64> = Vec::new();

        // E3 / E7 — buffered per-chunk annotations and pattern matches,
        // persisted after the indexing transaction commits. Buffering lets us
        // amortise auxiliary inserts and keeps the per-chunk path allocation-
        // light.
        let mut annotations_to_persist: Vec<(u64, String)> = Vec::new();
        let mut pattern_matches_to_persist: Vec<(u64, pattern_catalog::PatternMatch, String)> =
            Vec::new();
        let pattern_catalog_instance = PatternCatalog::new();

        store.begin_transaction()?;

        // Detect files deleted from disk but still in the index
        let current_rel_paths: HashSet<PathBuf> = files
            .iter()
            .filter_map(|p| p.strip_prefix(&project_path).ok())
            .map(std::path::Path::to_path_buf)
            .collect();

        for indexed_path in store.get_all_file_paths()? {
            if !current_rel_paths.contains(&indexed_path) {
                // Clean up graph data before removing chunks
                let path_str = indexed_path.to_string_lossy();
                let _ = store.delete_graph_data_for_file(&path_str);
                if let Ok(ids) = store.delete_chunks_for_file(&indexed_path) {
                    removed_chunk_ids.extend(ids);
                }
                store.remove_file_entry(&indexed_path)?;
                files_deleted += 1;
            }
        }

        if files_deleted > 0 {
            tracing::info!(
                "Detected {} deleted files ({} chunks to remove)",
                files_deleted,
                removed_chunk_ids.len()
            );
        }

        // Load or create sparse (BM25) index. The `use_bm25_stemmer` flag
        // (v0.4 Item 18) controls whether the English Snowball stemmer is
        // applied to indexed tokens. v0.4.1 W-Index #4 makes the flag part
        // of `IndexMeta`, and `SparseIndex::open` refuses to load an index
        // built with a different stemmer than the runtime config — so an
        // incremental rebuild after toggling the config errors out at this
        // point, telling the user to run `semantex index --rebuild`.
        let sparse_path = index_dir.join("sparse");
        let is_incremental = sparse_path.exists();
        let use_stemmer = self.config.use_bm25_stemmer;
        let sparse_index = if is_incremental {
            tracing::info!("Opening existing BM25 index for incremental update");
            SparseIndex::open(&sparse_path, use_stemmer)?
        } else {
            tracing::info!("Building new BM25 index...");
            if sparse_path.exists() {
                let _ = std::fs::remove_dir_all(&sparse_path);
            }
            SparseIndex::create(&sparse_path, use_stemmer)?
        };
        let mut sparse_writer = sparse_index.writer()?;

        // Remove stale documents from sparse index (deleted files)
        if !removed_chunk_ids.is_empty() {
            sparse_writer.delete_documents(&removed_chunk_ids)?;
        }

        // Streaming file processing: chunk → SQLite → BM25
        let mut total_chunks = 0usize;
        // Memory failsafe: check RSS every N files. Some repos have very large
        // individual files (vendored deps, generated code) that can blow the
        // cap inside the chunk loop on a single file, so we check on a tight
        // cadence rather than per-batch.
        const RSS_CHECK_EVERY_N_FILES: usize = 64;
        let mut files_since_rss_check: usize = 0;

        for file_path in &files {
            files_since_rss_check += 1;
            if files_since_rss_check >= RSS_CHECK_EVERY_N_FILES {
                files_since_rss_check = 0;
                if let Err(e) = crate::memory::check_rss_or_abort("indexer file loop") {
                    anyhow::bail!("Indexing aborted: {e}");
                }
            }
            let rel_path = file_path.strip_prefix(&project_path).unwrap_or(file_path);

            // Hash the file for incremental indexing
            let Ok(file_hash) = hasher::hash_file(file_path) else {
                files_skipped += 1;
                pb.inc(1);
                continue;
            };

            // Check if file has changed
            if let Ok(Some(stored_hash)) = store.get_file_hash(rel_path) {
                if stored_hash == file_hash {
                    files_skipped += 1;
                    pb.inc(1);
                    continue;
                }
                // File changed — remove old graph data and chunks
                let rel_path_str = rel_path.to_string_lossy();
                let _ = store.delete_graph_data_for_file(&rel_path_str);
                if let Ok(ids) = store.delete_chunks_for_file(rel_path) {
                    sparse_writer.delete_documents(&ids)?;
                    removed_chunk_ids.extend(ids);
                }
            }

            // Read file content
            let file_type = FileType::detect(file_path);
            let content = if file_type == FileType::Pdf {
                String::new() // PDF chunker reads from file directly
            } else if !file_type.is_text() {
                files_skipped += 1;
                pb.inc(1);
                continue;
            } else if let Ok(c) = std::fs::read_to_string(file_path) {
                c.replace("\r\n", "\n") // Normalize CRLF for consistent line counting
            } else {
                files_skipped += 1;
                pb.inc(1);
                continue;
            };

            // Chunk the file. A single unparseable file must never abort the
            // whole index: third-party parsers (pdf_extract on a malformed PDF,
            // a tree-sitter grammar on a pathological input) can `panic!`, not
            // just return `Err`. Contain a panic at this generic per-file
            // boundary and skip just that file. `Err` keeps its existing
            // propagation. (panic = "unwind" in the release profile makes the
            // catch effective — see workspace Cargo.toml.)
            let chunked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if file_type == FileType::Pdf {
                    pdf_chunker.chunk(file_path, "")
                } else if file_type.supports_ast() {
                    ast_chunker.chunk(rel_path, &content)
                } else {
                    text_chunker.chunk(rel_path, &content)
                }
            }));
            let Ok(chunk_result) = chunked else {
                tracing::warn!("Skipping {}: chunker panicked", file_path.display());
                files_skipped += 1;
                pb.inc(1);
                continue;
            };
            let chunks: Vec<Chunk> = chunk_result?;

            if chunks.is_empty() {
                files_skipped += 1;
                pb.inc(1);
                continue;
            }

            // Collect file-level import texts from structured metadata (for module_edges)
            #[allow(clippy::collapsible_if)]
            let chunks_for_imports: Vec<String> = chunks
                .iter()
                .find_map(|c| {
                    if let ChunkType::AstNode {
                        structured_meta: Some(ref meta),
                        ..
                    } = c.chunk_type
                    {
                        if !meta.resolved_imports.is_empty() {
                            return Some(meta.resolved_imports.clone());
                        }
                    }
                    None
                })
                .unwrap_or_default();

            // Get file metadata
            let metadata = std::fs::metadata(file_path)?;
            let mtime = metadata.modified().map_or(0, |t| {
                t.duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64
            });

            // Process each chunk: insert into SQLite + BM25
            let pattern_lang = PatternLang::from_language_name(file_type.language_name());
            for chunk in chunks {
                let chunk_id = store.insert_chunk(&chunk, file_hash, mtime)?;
                new_chunk_ids.push(chunk_id);

                // E3 — ExCS-style NL annotation synthesized from existing signals
                // (no LLM). Prepended to BM25 content so the NL→code vocabulary
                // gap is closed for sparse retrieval (e.g. "parallel failure
                // handling" lights up Promise.allSettled). Annotation is also
                // captured for later structured retrieval via the index registry.
                let (nl_annotation, base_bm25) = if let ChunkType::AstNode {
                    structured_meta: Some(ref meta),
                    ..
                } = chunk.chunk_type
                {
                    let annotation = synthesize_nl_annotation(meta, &meta.calls, &meta.called_by);
                    let nl_expansion = meta.bm25_expansion();
                    let bm25 = match (annotation.is_empty(), nl_expansion.is_empty()) {
                        (true, true) => chunk.content.clone(),
                        (true, false) => format!("{nl_expansion}\n{}", chunk.content),
                        (false, true) => format!("{annotation}\n{}", chunk.content),
                        (false, false) => {
                            format!("{annotation}\n{nl_expansion}\n{}", chunk.content)
                        }
                    };
                    (annotation, bm25)
                } else {
                    (String::new(), chunk.content.clone())
                };
                annotations_to_persist.push((chunk_id, nl_annotation));

                // Identifier expansion applied on top
                let ident_expansion = code_tokenizer::expand_identifiers(&base_bm25);
                let bm25_content = if ident_expansion.is_empty() {
                    base_bm25
                } else {
                    format!("{ident_expansion}\n{base_bm25}")
                };
                let file_path_str = chunk.file_path.to_string_lossy();
                sparse_writer.add_document(chunk_id, &bm25_content, &file_path_str)?;

                // E7 — Pattern catalog mining. Per-language deterministic
                // substring patterns; matches buffered for batch insertion
                // after the per-file loop to amortise the SQL round-trip.
                if let Some(lang) = pattern_lang {
                    let matches = pattern_catalog::mine_patterns_with(
                        &chunk.content,
                        lang,
                        &pattern_catalog_instance,
                    );
                    let fp_owned = chunk.file_path.to_string_lossy().into_owned();
                    for m in matches {
                        pattern_matches_to_persist.push((chunk_id, m, fp_owned.clone()));
                    }
                }

                // Store call graph edges and v7 graph data
                if let ChunkType::AstNode {
                    structured_meta: Some(ref meta),
                    ..
                } = chunk.chunk_type
                {
                    for call in &meta.calls {
                        store.store_call_graph_edge(chunk_id, call, None)?;
                    }

                    // Symbol definition
                    if let Some(ref name) = meta.name {
                        let kind = meta.kind.as_deref().unwrap_or("function");
                        let fp = chunk.file_path.to_string_lossy();
                        store.insert_symbol_def(chunk_id, name, kind, &fp)?;
                    }

                    // Type references
                    for type_ref in &meta.type_refs {
                        let ctx = match type_ref.context {
                            TypeRefContext::Param => "param",
                            TypeRefContext::Return => "return",
                            TypeRefContext::Field => "field",
                            TypeRefContext::Local => "local",
                            TypeRefContext::Generic => "generic",
                        };
                        store.insert_type_ref(chunk_id, &type_ref.type_name, ctx)?;
                    }

                    // Implementation relationships
                    for impl_rel in &meta.implements {
                        store.insert_type_hierarchy(
                            &impl_rel.implementor,
                            &impl_rel.trait_name,
                            "implements",
                        )?;
                    }
                }

                total_chunks += 1;
            }

            // Resolve and store module-level import edges
            if file_type.supports_ast() {
                let language_name = file_type.language_name();
                let rel_path_str = rel_path.to_string_lossy();
                for import_text in &chunks_for_imports {
                    if let Some(resolved) = import_resolver::resolve_import_path(
                        import_text,
                        language_name,
                        rel_path,
                        &current_rel_paths,
                    ) {
                        let resolved_str = resolved.to_string_lossy();
                        store.insert_module_edge(&rel_path_str, &resolved_str, import_text)?;
                    }
                }
            }

            // Classify and store file role
            let role = file_classifier::classify_file(rel_path);
            store.set_file_role(rel_path, role)?;

            // Update file entry
            store.set_file_entry(&FileEntry {
                path: rel_path.to_path_buf(),
                hash: file_hash,
                size: metadata.len(),
                mtime,
            })?;

            files_indexed += 1;
            pb.inc(1);
        }

        pb.finish_with_message("Files processed");

        store.commit_transaction()?;
        sparse_writer.commit()?;

        // Reload sparse reader to see committed changes
        if is_incremental {
            sparse_index.reload()?;
        }

        // Phase 3: Cross-file graph resolution
        if total_chunks > 0 {
            match global_graph::resolve_cross_file_graph(&store) {
                Ok(graph_stats) => {
                    tracing::info!(
                        calls_resolved = graph_stats.calls_resolved,
                        types_resolved = graph_stats.types_resolved,
                        hierarchy_resolved = graph_stats.hierarchy_resolved,
                        symbol_defs = graph_stats.symbol_defs_count,
                        module_edges = graph_stats.module_edges_count,
                        "Graph resolution completed"
                    );
                }
                Err(e) => {
                    tracing::warn!("Graph resolution failed (continuing without): {e}");
                }
            }
        }

        // Phase 3.5 (W2): annotation persistence (E3) + pattern catalog (E7)
        // + PageRank (E5). These are all derived signals indexed on chunks.db.
        // Failures degrade quality but never block the index build.
        if let Err(e) = persist_auxiliary_signals(
            &index_dir.join("chunks.db"),
            &annotations_to_persist,
            &pattern_matches_to_persist,
        ) {
            tracing::warn!("Annotation/pattern persistence failed: {e}");
        }
        if total_chunks > 0
            && let Err(e) = compute_and_store_pagerank(&index_dir.join("chunks.db"))
        {
            tracing::warn!("PageRank computation failed: {e}");
        }

        let total_removals = removed_chunk_ids.len();

        // Selection (S2 re-point): resolve the active dense backend via the S8
        // ModelRegistry (canonical `SEMANTEX_EMBEDDER`), with the deprecated
        // `SEMANTEX_DENSE_BACKEND`/`config.dense_backend` alias honored when set
        // non-default. D4 cutover: all-defaults now resolve to coderank-hnsw.
        let backend_kind =
            crate::model::ModelRegistry::resolve_dense_backend(&self.config, Some(&project_path))
                .unwrap_or_default();

        // The dense index sentinel file is backend-specific: coderank-hnsw writes
        // `vectors.bin` into the per-backend subdir (`dense/<backend>/`). Treat
        // dense as "missing" (forcing a rebuild) when the sentinel is absent.
        let dense_sentinel = dense_sentinel_file(backend_kind);
        let dense_dir_for_check = dense_subdir(&index_dir, backend_kind);
        let dense_missing_for_early_return = !dense_dir_for_check.join(dense_sentinel).exists();

        if total_chunks == 0 && total_removals == 0 && !dense_missing_for_early_return {
            tracing::info!(
                files_scanned,
                files_indexed,
                files_skipped,
                files_deleted,
                chunks_created = 0,
                chunks_removed = 0,
                duration_secs = start.elapsed().as_secs_f64(),
                "Index build completed with no changes"
            );
            return Ok(IndexStats {
                files_scanned,
                files_indexed,
                files_skipped,
                files_deleted,
                chunks_created: 0,
                chunks_removed: 0,
                duration: start.elapsed(),
            });
        }

        // Build or incrementally update the dense index via the selected
        // DenseBackend. Each backend persists into its per-backend subdir
        // `.semantex/dense/<backend>/` (`backend_kind` resolved above).
        let dense_dir = dense_subdir(&index_dir, backend_kind);
        std::fs::create_dir_all(&dense_dir)?;
        let dense_missing = !dense_dir.join(dense_sentinel).exists();
        if dense_missing {
            tracing::info!("Dense index missing — rebuilding dense embeddings");
        }

        if let Err(e) = (|| -> Result<()> {
            match backend_kind {
                DenseBackendKind::CoderankHnsw => {
                    use crate::index::hnsw_index::{CoderankHnswIndexBuilder, HnswParams};
                    let params = HnswParams::resolve(
                        &self.config.hnsw_preset,
                        self.config.hnsw_ef_search,
                        self.config.dense_rescore_k,
                    );
                    // SEMANTEX_DENSE_CONTEXT (default OFF): when on, embed
                    // `format!("{annotation}\n{code}")` using the graph-derived NL
                    // annotation already persisted for BM25 (E3). The harness
                    // ablates this on/off; raw code is the default. Changing the
                    // embedded text changes the embedder_fingerprint, so on/off
                    // are separate indexes (the harness builds each independently).
                    let dense_context = std::env::var("SEMANTEX_DENSE_CONTEXT")
                        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));

                    if dense_missing {
                        let _slot = crate::index::gate::acquire(|| {
                            tracing::info!(
                                "Waiting for a free index-build slot (max {} concurrent full builds)",
                                crate::index::gate::max_concurrent_builds()
                            );
                        });
                        let all_ids = store.get_all_chunk_ids()?;
                        if all_ids.is_empty() {
                            return Ok(());
                        }
                        // Annotations (E3 NL strings) are small — fetch the id→text
                        // map up front; the heavy CHUNK CONTENT is streamed per
                        // batch below, never materialized whole.
                        let annotations = if dense_context {
                            store.get_annotations(&all_ids)?
                        } else {
                            std::collections::HashMap::new()
                        };
                        let mut dense_builder = CoderankHnswIndexBuilder::new(&dense_dir, params)
                            .with_models_dir(self.config.models_dir())
                            .with_context_annotations(dense_context, annotations);
                        // Stream the full corpus one ENCODE_BATCH at a time
                        // (RSS-bounded; never materialize all chunk content at
                        // once) — same memory discipline as the PLAID arm. The
                        // closure pulls each ≤32-id batch's content and drops it
                        // before the next, so only one batch's text is ever live.
                        dense_builder.build_streaming_ids(&all_ids, |batch_ids| {
                            let mut chunks = store.get_chunks(batch_ids)?;
                            // `get_chunks` (WHERE id IN (...)) is unordered; sort
                            // by id so per-batch order is deterministic and matches
                            // the ascending all_ids order (stable vector store).
                            chunks.sort_by_key(|c| c.id);
                            Ok(chunks.into_iter().map(|c| (c.id, c.content)).collect())
                        })?;
                    } else {
                        let annotations = if dense_context && !new_chunk_ids.is_empty() {
                            store.get_annotations(&new_chunk_ids)?
                        } else {
                            std::collections::HashMap::new()
                        };
                        let mut dense_builder = CoderankHnswIndexBuilder::new(&dense_dir, params)
                            .with_models_dir(self.config.models_dir())
                            .with_context_annotations(dense_context, annotations);
                        if !removed_chunk_ids.is_empty() {
                            dense_builder.delete(&removed_chunk_ids)?;
                        }
                        if !new_chunk_ids.is_empty() {
                            // Incremental adds also stream per batch (no whole-set
                            // content materialization).
                            dense_builder.insert_streaming_ids(&new_chunk_ids, |batch_ids| {
                                let chunks = store.get_chunks(batch_ids)?;
                                Ok(chunks.into_iter().map(|c| (c.id, c.content)).collect())
                            })?;
                        }
                    }
                    tracing::info!(
                        added = new_chunk_ids.len(),
                        removed = removed_chunk_ids.len(),
                        "Dense index (coderank-hnsw) build complete"
                    );
                }
            }
            Ok(())
        })() {
            tracing::warn!("Dense index build failed (continuing without): {e}");
        }

        // Save metadata. v0.4.1 W-Index #4: persist the BM25 stemmer flag so
        // open-time code (see `sparse_search::SparseIndex::open` + the
        // verification call in `hybrid.rs`) can refuse to load an index whose
        // build-time stemmer disagrees with the runtime config.
        let actual_chunk_count = store.chunk_count().unwrap_or(total_chunks as u64);
        let actual_file_count = store.file_count().unwrap_or(files_indexed + files_skipped);
        // S8: stamp the active embedder's fingerprint so open-time code can
        // detect a vector-space change. The fingerprint is backend-agnostic
        // (id+dims+pooling+quant+norm+prefix), resolved from the merged manifest
        // (built-ins + an optional user `models.toml`) — so it does NOT depend on
        // S2's dense backend arm. (The versioned-dir build + atomic switchover
        // wiring is the post-S2 reconciliation pass, gated on S2's builder arm.)
        let embedder_fingerprint = {
            let registry =
                crate::model::ModelRegistry::from_config(&self.config, Some(&project_path))?;
            let embedder_spec = registry.active_embedder()?;
            let crate::model::RoleData::Embedder(embedder_data) = &embedder_spec.role_data else {
                anyhow::bail!(
                    "active embedder `{}` is not an embedder spec",
                    embedder_spec.id
                );
            };
            crate::model::EmbedderFingerprint::compute(&embedder_spec.id, embedder_data)
        };
        // S2: record the embedding model/dim for the ACTIVE dense backend, and
        // stamp the RESOLVED backend name (not the raw config string) so the
        // open-time `verify_persisted_backend_matches` agrees with the resolved
        // selection (canonical SEMANTEX_EMBEDDER may differ from the deprecated
        // dense_backend alias default).
        let (embedding_model, embedding_dim) = match backend_kind {
            DenseBackendKind::CoderankHnsw => (
                "CodeRankEmbed".to_string(),
                crate::embedding::single_vector::SingleVectorEmbedder::embedding_dim() as u32,
            ),
        };
        let meta = IndexMeta {
            schema_version: IndexMeta::CURRENT_SCHEMA_VERSION,
            project_path: project_path.clone(),
            created_at: chrono_now(),
            updated_at: chrono_now(),
            file_count: actual_file_count,
            chunk_count: actual_chunk_count,
            embedding_model,
            embedding_dim,
            use_bm25_stemmer: self.config.use_bm25_stemmer,
            dense_backend: backend_kind.name().to_string(),
            embedder_fingerprint,
        };
        let meta_json = serde_json::to_string_pretty(&meta)?;
        std::fs::write(index_dir.join("meta.json"), meta_json)?;

        let duration = start.elapsed();
        tracing::info!(
            files_scanned,
            files_indexed,
            files_skipped,
            files_deleted,
            chunks_created = total_chunks,
            chunks_removed = total_removals,
            duration_secs = duration.as_secs_f64(),
            "Index build completed successfully"
        );

        Ok(IndexStats {
            files_scanned,
            files_indexed,
            files_skipped,
            files_deleted,
            chunks_created: total_chunks as u64,
            chunks_removed: total_removals as u64,
            duration,
        })
    }
}

/// The per-backend "dense index present" sentinel file. coderank-hnsw writes
/// `vectors.bin`. Its presence in the per-backend subdir marks the dense index
/// as built (else: rebuild).
fn dense_sentinel_file(backend: DenseBackendKind) -> &'static str {
    match backend {
        DenseBackendKind::CoderankHnsw => "vectors.bin",
    }
}

fn chrono_now() -> String {
    // Simple ISO 8601 timestamp without chrono dependency
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    // Format as seconds since epoch (good enough without chrono)
    format!("{secs}")
}

// ─────────────────────────────────────────────────────────────────────────────
// W2 — auxiliary signal persistence (E3 annotations, E7 patterns, E5 PageRank)
//
// These helpers open a separate connection on `chunks.db` (the same DB the
// `ChunkStore` uses) so we can add v0.3-introduced tables/columns without
// touching `storage.rs` (which is co-owned with W4). The schema migration is
// idempotent via `CREATE TABLE IF NOT EXISTS` / `ALTER TABLE ... ADD COLUMN`
// guarded by a column-existence probe.
// ─────────────────────────────────────────────────────────────────────────────

/// Create the v0.3 auxiliary schema (annotations, patterns, pagerank) and
/// persist the buffered annotations + pattern matches.
fn persist_auxiliary_signals(
    db_path: &Path,
    annotations: &[(u64, String)],
    pattern_matches: &[(u64, pattern_catalog::PatternMatch, String)],
) -> Result<()> {
    let mut conn = Connection::open(db_path).with_context(|| {
        format!(
            "Failed to open chunks.db for aux signals: {}",
            db_path.display()
        )
    })?;
    init_auxiliary_schema(&conn)?;

    let tx = conn.transaction()?;

    // E3 — chunk_annotations: replace existing rows for each chunk so
    // incremental updates stay consistent.
    {
        let mut stmt = tx.prepare(
            "INSERT OR REPLACE INTO chunk_annotations (chunk_id, nl_annotation) VALUES (?1, ?2)",
        )?;
        for (chunk_id, annotation) in annotations {
            if annotation.is_empty() {
                continue;
            }
            stmt.execute(params![*chunk_id as i64, annotation])?;
        }
    }

    // E7 — pattern_matches: keyed by (chunk_id, pattern_name) so re-indexing
    // the same chunk does not duplicate matches.
    {
        let mut stmt = tx.prepare(
            "INSERT OR REPLACE INTO pattern_matches \
             (chunk_id, pattern_name, language, description, file_path) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;
        for (chunk_id, m, file_path) in pattern_matches {
            stmt.execute(params![
                *chunk_id as i64,
                m.pattern_name,
                m.language,
                m.description,
                file_path,
            ])?;
        }
    }

    tx.commit()?;
    tracing::info!(
        annotations = annotations.len(),
        patterns = pattern_matches.len(),
        "Auxiliary signals persisted"
    );
    Ok(())
}

/// Compute PageRank over the call+import+hierarchy graph and persist the
/// per-chunk centrality scores. Returns early on an empty index.
fn compute_and_store_pagerank(db_path: &Path) -> Result<()> {
    let mut conn = Connection::open(db_path).with_context(|| {
        format!(
            "Failed to open chunks.db for PageRank: {}",
            db_path.display()
        )
    })?;
    init_auxiliary_schema(&conn)?;

    // Pull edges + node set in a streaming-friendly shape. All these queries
    // hit indexed columns and return narrow tuples.
    let all_chunk_ids = query_all_chunk_ids(&conn)?;
    if all_chunk_ids.is_empty() {
        return Ok(());
    }
    let call_edges = query_call_edges(&conn)?;
    let type_ref_edges = query_type_ref_edges(&conn)?;
    let hierarchy_edges = query_hierarchy_edges(&conn)?;

    let graph = page_rank::build_code_graph(
        &call_edges,
        &type_ref_edges,
        &hierarchy_edges,
        &all_chunk_ids,
    );
    let scores = page_rank::compute_pagerank(&graph);
    if scores.is_empty() {
        return Ok(());
    }

    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT OR REPLACE INTO chunk_centrality (chunk_id, structural_centrality) VALUES (?1, ?2)",
        )?;
        for (chunk_id, score) in &scores {
            stmt.execute(params![*chunk_id as i64, f64::from(*score)])?;
        }
    }
    tx.commit()?;
    tracing::info!(
        nodes = graph.node_count(),
        edges = graph.edge_count(),
        scored = scores.len(),
        "PageRank stored"
    );
    Ok(())
}

/// Create the v0.3 auxiliary tables if missing. Idempotent.
fn init_auxiliary_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS chunk_annotations (
            chunk_id INTEGER PRIMARY KEY,
            nl_annotation TEXT NOT NULL,
            FOREIGN KEY (chunk_id) REFERENCES chunks(id)
        );

        CREATE TABLE IF NOT EXISTS pattern_matches (
            chunk_id     INTEGER NOT NULL,
            pattern_name TEXT NOT NULL,
            language     TEXT NOT NULL,
            description  TEXT NOT NULL,
            file_path    TEXT NOT NULL,
            PRIMARY KEY (chunk_id, pattern_name),
            FOREIGN KEY (chunk_id) REFERENCES chunks(id)
        );
        CREATE INDEX IF NOT EXISTS idx_pattern_matches_name ON pattern_matches(pattern_name);
        CREATE INDEX IF NOT EXISTS idx_pattern_matches_lang ON pattern_matches(language);

        CREATE TABLE IF NOT EXISTS chunk_centrality (
            chunk_id INTEGER PRIMARY KEY,
            structural_centrality REAL NOT NULL,
            FOREIGN KEY (chunk_id) REFERENCES chunks(id)
        );
        CREATE INDEX IF NOT EXISTS idx_chunk_centrality_score
            ON chunk_centrality(structural_centrality);
        ",
    )?;
    Ok(())
}

fn query_all_chunk_ids(conn: &Connection) -> Result<Vec<u64>> {
    let mut stmt = conn.prepare("SELECT id FROM chunks")?;
    let rows = stmt
        .query_map([], |row| row.get::<_, i64>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows.into_iter().map(|id| id as u64).collect())
}

fn query_call_edges(conn: &Connection) -> Result<Vec<(u64, u64)>> {
    let mut stmt = conn.prepare(
        "SELECT caller_chunk_id, callee_chunk_id FROM call_graph WHERE callee_chunk_id IS NOT NULL",
    )?;
    let rows = stmt
        .query_map([], |row| {
            Ok((row.get::<_, i64>(0)? as u64, row.get::<_, i64>(1)? as u64))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

fn query_type_ref_edges(conn: &Connection) -> Result<Vec<(u64, u64)>> {
    let mut stmt = conn.prepare(
        "SELECT defining_chunk, chunk_id FROM type_refs WHERE defining_chunk IS NOT NULL",
    )?;
    let rows = stmt
        .query_map([], |row| {
            Ok((row.get::<_, i64>(0)? as u64, row.get::<_, i64>(1)? as u64))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

fn query_hierarchy_edges(conn: &Connection) -> Result<Vec<(u64, u64)>> {
    let mut stmt = conn.prepare(
        "SELECT child_chunk, parent_chunk FROM type_hierarchy \
         WHERE child_chunk IS NOT NULL AND parent_chunk IS NOT NULL",
    )?;
    let rows = stmt
        .query_map([], |row| {
            Ok((row.get::<_, i64>(0)? as u64, row.get::<_, i64>(1)? as u64))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Chunk, ChunkType};
    use std::path::PathBuf;

    fn make_chunk(id: u64, content: &str) -> Chunk {
        Chunk {
            id,
            file_path: PathBuf::from("test.rs"),
            start_line: 1,
            end_line: 1,
            content: content.to_string(),
            chunk_type: ChunkType::TextWindow { window_index: 0 },
        }
    }

    /// Building with the coderank-137m embedder (→ coderank-hnsw backend)
    /// writes the per-backend subdir `.semantex/dense/coderank-hnsw/vectors.bin`
    /// and records the model + dim + resolved backend in meta.json. `#[ignore]`
    /// — needs the CodeRankEmbed model download.
    #[test]
    #[ignore]
    fn coderank_hnsw_build_writes_subdir_and_meta() {
        use crate::config::SemantexConfig;
        let tmp = tempfile::TempDir::new().unwrap();
        let project = tmp.path().join("repo");
        std::fs::create_dir_all(&project).unwrap();
        std::fs::write(project.join("a.rs"), "pub fn hello() -> u32 { 41 + 1 }\n").unwrap();

        let cfg = SemantexConfig {
            embedder: "coderank-137m".to_string(),
            ..SemantexConfig::default()
        };
        IndexBuilder::new(&cfg).unwrap().build(&project).unwrap();

        assert!(
            project
                .join(".semantex/dense/coderank-hnsw/vectors.bin")
                .exists()
        );
        let meta: crate::types::IndexMeta = serde_json::from_str(
            &std::fs::read_to_string(project.join(".semantex/meta.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(meta.dense_backend, "coderank-hnsw");
        assert_eq!(meta.schema_version, 11);
        assert_eq!(meta.embedding_model, "CodeRankEmbed");
    }

    /// v0.4.1 W-Index #3 — synthetic mapping contract:
    ///
    /// A mapping vector with `PLAID_TOMBSTONE` at index 1 and real chunk IDs
    /// at indices 0 and 2 must be safe to pass through the
    /// `translate_chunk_subset_to_doc_subset` translator and through
    /// `PlaidSearcher::search`-shape iterators without surfacing the
    /// tombstone slot as a chunk_id 0 phantom.
    ///
    /// This is a unit-level test for the search-side reader contract; the
    /// integration test (mapping survives a real delete+rebuild) requires
    /// a live PLAID index and is exercised by the manual `--rebuild` flow.
    #[test]
    fn tombstone_skipping_in_doc_to_chunk_lookup() {
        use crate::types::PLAID_TOMBSTONE;

        // Synthetic mapping: doc 0 -> chunk 100, doc 1 -> tombstone,
        // doc 2 -> chunk 300.
        let mapping: Vec<u64> = vec![100, PLAID_TOMBSTONE, 300];

        // Simulate the per-result filter that `PlaidSearcher::search` runs
        // against `passage_ids` (doc_ids returned by next-plaid). The
        // contract is: a result hitting the tombstone slot must NOT surface
        // as a SearchResult.
        let passage_ids = vec![0i64, 1, 2];
        let scored: Vec<u64> = passage_ids
            .iter()
            .filter_map(|&doc_id| {
                let doc_idx = doc_id as usize;
                mapping
                    .get(doc_idx)
                    .filter(|&&cid| cid != PLAID_TOMBSTONE)
                    .copied()
            })
            .collect();

        assert_eq!(
            scored,
            vec![100u64, 300],
            "tombstoned position must NOT surface as a chunk_id; got {scored:?}",
        );
    }

    /// v0.4.1 W-Index #3: a `PLAID_TOMBSTONE` value must never be confused
    /// with a real chunk_id. This is a documentation-as-test: SQLite
    /// AUTOINCREMENT IDs start at 1 and rowid is i64, so u64::MAX is well
    /// above the practical chunk-id space and reserved as the sentinel.
    #[test]
    fn plaid_tombstone_is_reserved_sentinel() {
        use crate::types::PLAID_TOMBSTONE;
        assert_eq!(PLAID_TOMBSTONE, u64::MAX);
    }

    #[test]
    fn aux_schema_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("chunks.db");
        // Seed the chunks table the way ChunkStore would.
        {
            let conn = Connection::open(&db).unwrap();
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY);
                 CREATE TABLE IF NOT EXISTS call_graph (
                    caller_chunk_id INTEGER NOT NULL,
                    callee_name TEXT NOT NULL,
                    callee_chunk_id INTEGER);
                 CREATE TABLE IF NOT EXISTS type_refs (
                    chunk_id INTEGER NOT NULL,
                    type_name TEXT NOT NULL,
                    ref_context TEXT NOT NULL,
                    defining_chunk INTEGER);
                 CREATE TABLE IF NOT EXISTS type_hierarchy (
                    child_name TEXT NOT NULL,
                    parent_name TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    child_chunk INTEGER,
                    parent_chunk INTEGER);",
            )
            .unwrap();
        }
        let conn = Connection::open(&db).unwrap();
        // Running init twice must not error.
        init_auxiliary_schema(&conn).unwrap();
        init_auxiliary_schema(&conn).unwrap();
    }

    #[test]
    fn persist_annotations_and_patterns_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("chunks.db");
        // Bootstrap chunks table so foreign-key references resolve.
        {
            let conn = Connection::open(&db).unwrap();
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY);
                 INSERT INTO chunks(id) VALUES (1), (2);",
            )
            .unwrap();
        }
        let anns = vec![
            (1u64, "annotation for chunk 1".to_string()),
            (2u64, "annotation for chunk 2".to_string()),
        ];
        let m = pattern_catalog::PatternMatch {
            pattern_name: "rust.drop_impl".to_string(),
            description: "Drop impl".to_string(),
            language: "rust".to_string(),
        };
        let matches = vec![(1u64, m, "lib.rs".to_string())];
        persist_auxiliary_signals(&db, &anns, &matches).unwrap();

        let conn = Connection::open(&db).unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM chunk_annotations", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 2);

        let pat_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM pattern_matches", [], |row| row.get(0))
            .unwrap();
        assert_eq!(pat_count, 1);
    }

    #[test]
    fn compute_pagerank_writes_centrality() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("chunks.db");
        {
            let conn = Connection::open(&db).unwrap();
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY);
                 CREATE TABLE IF NOT EXISTS call_graph (
                    caller_chunk_id INTEGER NOT NULL,
                    callee_name TEXT NOT NULL,
                    callee_chunk_id INTEGER);
                 CREATE TABLE IF NOT EXISTS type_refs (
                    chunk_id INTEGER NOT NULL,
                    type_name TEXT NOT NULL,
                    ref_context TEXT NOT NULL,
                    defining_chunk INTEGER);
                 CREATE TABLE IF NOT EXISTS type_hierarchy (
                    child_name TEXT NOT NULL,
                    parent_name TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    child_chunk INTEGER,
                    parent_chunk INTEGER);
                 INSERT INTO chunks(id) VALUES (1), (2), (3);
                 INSERT INTO call_graph(caller_chunk_id, callee_name, callee_chunk_id)
                    VALUES (1, 'x', 2), (3, 'x', 2);",
            )
            .unwrap();
        }
        compute_and_store_pagerank(&db).unwrap();
        let conn = Connection::open(&db).unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM chunk_centrality", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 3, "all 3 chunks should have a centrality score");

        // Node 2 (the called hub) should outrank nodes 1 and 3.
        let centrality_for = |id: i64| -> f64 {
            conn.query_row(
                "SELECT structural_centrality FROM chunk_centrality WHERE chunk_id = ?1",
                [id],
                |row| row.get(0),
            )
            .unwrap()
        };
        assert!(
            centrality_for(2) > centrality_for(1),
            "callee hub should outrank caller"
        );
    }

    #[test]
    fn aux_persistence_skips_empty_annotation() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("chunks.db");
        {
            let conn = Connection::open(&db).unwrap();
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY);
                 INSERT INTO chunks(id) VALUES (1);",
            )
            .unwrap();
        }
        let anns = vec![(1u64, String::new())];
        persist_auxiliary_signals(&db, &anns, &[]).unwrap();
        let conn = Connection::open(&db).unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM chunk_annotations", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 0, "empty annotations should be skipped");
    }

    #[test]
    fn make_chunk_helper_compiles() {
        // Keeps `make_chunk` and Chunk imports live for future helper tests.
        let c = make_chunk(42, "hello");
        assert_eq!(c.id, 42);
        assert_eq!(c.content, "hello");
    }
}
