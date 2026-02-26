use crate::chunking::Chunker;
use crate::chunking::ast_chunker::AstChunker;
use crate::chunking::import_resolver;
use crate::chunking::pdf_chunker::PdfChunker;
use crate::chunking::structured_meta::TypeRefContext;
use crate::chunking::text_chunker::TextChunker;
use crate::config::SemantexConfig;
use crate::embedding::colbert::ColbertEmbedder;
use crate::embedding::model_manager;
use crate::file::detector::FileType;
use crate::file::hasher;
use crate::file::walker::FileWalker;
use crate::index::file_classifier;
use crate::index::global_graph;
use crate::index::storage::ChunkStore;
use crate::search::code_tokenizer;
use crate::search::sparse_search::SparseIndex;
use crate::types::{Chunk, ChunkType, FileEntry, IndexMeta};
use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
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
        let lock_file = std::fs::File::create(index_dir.join(".sage.lock"))?;
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

        // Check schema version compatibility
        let expected_version = IndexMeta::CURRENT_SCHEMA_VERSION;
        let meta_path = index_dir.join("meta.json");
        if meta_path.exists()
            && let Ok(meta_str) = std::fs::read_to_string(&meta_path)
            && let Ok(existing_meta) = serde_json::from_str::<IndexMeta>(&meta_str)
            && existing_meta.schema_version != expected_version
        {
            tracing::warn!(
                "Index schema version mismatch (found v{}, expected v{}). \
                 Full re-index required.",
                existing_meta.schema_version,
                expected_version
            );
            // Remove existing index files to force full rebuild
            let _ = std::fs::remove_dir_all(index_dir.join("sparse"));
            let _ = std::fs::remove_dir_all(index_dir.join("plaid"));
            let _ = std::fs::remove_file(index_dir.join("plaid_mapping.bin"));
            let _ = std::fs::remove_file(index_dir.join("chunks.db"));
            // Clean up legacy HNSW files if present
            let _ = std::fs::remove_file(index_dir.join("dense.usearch"));
            let _ = std::fs::remove_file(index_dir.join("dense_coarse.usearch"));
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

        // Load or create sparse (BM25) index
        let sparse_path = index_dir.join("sparse");
        let is_incremental = sparse_path.exists();
        let sparse_index = if is_incremental {
            tracing::info!("Opening existing BM25 index for incremental update");
            SparseIndex::open(&sparse_path)?
        } else {
            tracing::info!("Building new BM25 index...");
            if sparse_path.exists() {
                let _ = std::fs::remove_dir_all(&sparse_path);
            }
            SparseIndex::create(&sparse_path)?
        };
        let mut sparse_writer = sparse_index.writer()?;

        // Remove stale documents from sparse index (deleted files)
        if !removed_chunk_ids.is_empty() {
            sparse_writer.delete_documents(&removed_chunk_ids)?;
        }

        // Streaming file processing: chunk → SQLite → BM25
        let mut total_chunks = 0usize;

        for file_path in &files {
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

            // Chunk the file
            let chunks: Vec<Chunk> = if file_type == FileType::Pdf {
                pdf_chunker.chunk(file_path, "")?
            } else if file_type.supports_ast() {
                ast_chunker.chunk(rel_path, &content)?
            } else {
                text_chunker.chunk(rel_path, &content)?
            };

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
            let mtime = metadata
                .modified()
                .map(|t| {
                    t.duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs() as i64
                })
                .unwrap_or(0);

            // Process each chunk: insert into SQLite + BM25
            for chunk in chunks {
                let chunk_id = store.insert_chunk(&chunk, file_hash, mtime)?;

                // Prepend structured NL summary for BM25 enrichment
                let base_bm25 = if let ChunkType::AstNode {
                    structured_meta: Some(ref meta),
                    ..
                } = chunk.chunk_type
                {
                    let nl_expansion = meta.bm25_expansion();
                    if nl_expansion.is_empty() {
                        chunk.content.clone()
                    } else {
                        format!("{}\n{}", nl_expansion, chunk.content)
                    }
                } else {
                    chunk.content.clone()
                };

                // Identifier expansion applied on top
                let ident_expansion = code_tokenizer::expand_identifiers(&base_bm25);
                let bm25_content = if ident_expansion.is_empty() {
                    base_bm25
                } else {
                    format!("{ident_expansion}\n{base_bm25}")
                };
                let file_path_str = chunk.file_path.to_string_lossy();
                sparse_writer.add_document(chunk_id, &bm25_content, &file_path_str)?;

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

        let total_removals = removed_chunk_ids.len();

        if total_chunks == 0 && total_removals == 0 {
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

        // Build PLAID index (ColBERT late-interaction)
        let colbert_model_dir = model_manager::ensure_colbert_model(&self.config.models_dir())?;
        match (|| -> Result<()> {
            use next_plaid::{IndexConfig, MmapIndex, UpdateConfig};

            let embedder = ColbertEmbedder::new(&colbert_model_dir)?;

            // Collect all chunk contents from store for ColBERT encoding
            let mut chunk_ids_all: Vec<u64> = Vec::new();
            let mut chunk_contents: Vec<String> = Vec::new();
            store.for_each_chunk_content(|id, content| {
                chunk_ids_all.push(id);
                chunk_contents.push(content.to_string());
            })?;

            if chunk_contents.is_empty() {
                tracing::info!("No chunks to encode for PLAID index");
                return Ok(());
            }

            let doc_embeddings = embedder.encode_documents(&chunk_contents)?;

            let plaid_dir = index_dir.join("plaid");
            std::fs::create_dir_all(&plaid_dir)?;

            // Build PLAID index via next-plaid
            let plaid_config = IndexConfig {
                nbits: self.config.plaid_nbits,
                ..Default::default()
            };
            let (_index, _doc_ids) = MmapIndex::update_or_create(
                &doc_embeddings,
                &plaid_dir.to_string_lossy(),
                &plaid_config,
                &UpdateConfig::default(),
            )?;

            // Save doc_id -> chunk_id mapping
            let mapping =
                bincode::serde::encode_to_vec(&chunk_ids_all, bincode::config::standard())?;
            std::fs::write(index_dir.join("plaid_mapping.bin"), mapping)?;

            tracing::info!("PLAID index built successfully");
            Ok(())
        })() {
            Ok(()) => {}
            Err(e) => tracing::warn!("PLAID index build failed (continuing without): {e}"),
        }

        // Save metadata
        let actual_chunk_count = store.chunk_count().unwrap_or(total_chunks as u64);
        let actual_file_count = store.file_count().unwrap_or(files_indexed + files_skipped);
        let meta = IndexMeta {
            schema_version: IndexMeta::CURRENT_SCHEMA_VERSION,
            project_path: project_path.clone(),
            created_at: chrono_now(),
            updated_at: chrono_now(),
            file_count: actual_file_count,
            chunk_count: actual_chunk_count,
            embedding_model: "LateOn-Code-edge".to_string(),
            embedding_dim: 48,
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

fn chrono_now() -> String {
    // Simple ISO 8601 timestamp without chrono dependency
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    // Format as seconds since epoch (good enough without chrono)
    format!("{secs}")
}
