//! `colbert-plaid` — the first `DenseBackend`/`DenseIndexBuilder` impl,
//! wrapping the ColBERT late-interaction + vendored next-plaid PLAID path.
//! Behavior is byte-identical to the pre-seam inline PLAID code.

use crate::embedding::colbert::ColbertEmbedder;
use crate::embedding::model_manager;
use crate::search::dense_backend::{DenseBackend, DenseHit, DenseIndexBuilder};
use crate::search::plaid_search::PlaidSearcher;
use crate::types::DENSE_TOMBSTONE;
use anyhow::Result;
use ndarray::Axis;
use std::path::{Path, PathBuf};

/// PLAID UpdateConfig buffer_size — pending-doc threshold below which next-plaid
/// writes to a buffer without full k-means (v0.4_SPEC §6.3). Mirrors the
/// constant previously in `index/builder.rs`.
const PLAID_BUFFER_SIZE: usize = 50;

/// PLAID encode batch size. MUST stay strictly below `PLAID_BUFFER_SIZE` so a
/// single-file incremental refresh skips k-means (v0.4.1 W-Index #12). Mirrors
/// the constant previously in `index/builder.rs`.
const PLAID_BATCH: usize = 32;

/// Query-time `colbert-plaid` backend: owns a `PlaidSearcher` and a reference
/// to the process-global ColBERT encoder.
pub struct ColbertPlaidBackend {
    plaid: PlaidSearcher,
    colbert: &'static ColbertEmbedder,
}

impl ColbertPlaidBackend {
    /// Stable backend identity.
    pub const NAME: &'static str = "colbert-plaid";

    /// Open a `colbert-plaid` backend from a PLAID index directory + its
    /// `plaid_mapping.bin` sidecar, using the process-global ColBERT encoder.
    ///
    /// `plaid_dir` is the on-disk PLAID directory (the per-backend
    /// `dense/colbert-plaid/` going forward, or the legacy `plaid/`).
    /// `mapping_path` is the postcard-encoded doc→chunk mapping.
    /// `model_dir` is the resolved ColBERT model directory.
    pub fn open(plaid_dir: &Path, mapping_path: &Path, model_dir: &Path) -> Result<Self> {
        let plaid = PlaidSearcher::open(plaid_dir, mapping_path)?;
        let colbert = ColbertEmbedder::global(model_dir)?;
        Ok(Self { plaid, colbert })
    }

    /// Borrow the wrapped `PlaidSearcher` (used by `hybrid.rs` to compute the
    /// `file_filter` chunk-ID subset from `doc_to_chunk()` — preserving the
    /// exact pre-seam subset construction).
    pub fn plaid(&self) -> &PlaidSearcher {
        &self.plaid
    }
}

impl DenseBackend for ColbertPlaidBackend {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn search(&self, query: &str, k: usize) -> Result<Vec<DenseHit>> {
        // Delegates verbatim to PlaidSearcher::search — byte-identical to the
        // pre-seam `plaid.search(colbert, &effective_text, retrieval_candidates)`.
        self.plaid.search(self.colbert, query, k)
    }

    fn search_with_subset(&self, query: &str, k: usize, subset: &[u64]) -> Result<Vec<DenseHit>> {
        // The pre-seam call passed `Option<&[u64]>`; the trait takes a `&[u64]`.
        // An empty subset means "no candidates" → empty result, matching
        // PlaidSearcher::search_with_subset's documented short-circuit.
        self.plaid
            .search_with_subset(self.colbert, query, k, Some(subset))
    }

    fn positional_chunk_ids(&self) -> Option<&[u64]> {
        Some(self.plaid.doc_to_chunk())
    }

    /// Single-vector query projection for S7 (MMR / semantic cache): encode the
    /// query with the same ColBERT encoder the search path uses, then
    /// mean-pool + L2-normalize its per-token vectors into one `Vec<f32>`.
    /// `coderank-hnsw` (S2) overrides this with its exact int8-store vectors;
    /// colbert-plaid's mean-pool is an approximate projection of the
    /// late-interaction representation.
    fn embed_text_vector(&self, query: &str) -> Option<Vec<f32>> {
        // `encode_query` returns `[N_query_tokens, dim]` (the same call the
        // PLAID search path makes internally). Mean-pool over tokens → `[dim]`.
        let tokens = self.colbert.encode_query(query).ok()?;
        mean_pool_l2(&tokens)
    }

    fn embed_doc_vectors(&self, _chunk_ids: &[u64]) -> Option<Vec<(u64, Vec<f32>)>> {
        // PLAID stores compressed/quantized residuals, not per-chunk fp32 token
        // vectors, and this backend does not hold the ChunkStore — so a faithful
        // per-doc projection isn't recoverable from the colbert-plaid index.
        // Returns None; S7's doc-side features (MMR over stored doc vectors) are
        // exact only on `coderank-hnsw`, which keeps real int8 vectors. (Per the
        // integration doc D-mmr-cache: validate MMR/semantic-cache on S2.)
        None
    }
}

/// Mean-pool the per-token embedding matrix `[N_tokens, dim]` over the token
/// axis and L2-normalize the result into a single `[dim]` vector. Returns
/// `None` for an empty matrix (no tokens) or a degenerate zero norm.
fn mean_pool_l2(tokens: &crate::embedding::colbert::TokenEmbeddings) -> Option<Vec<f32>> {
    if tokens.nrows() == 0 {
        return None;
    }
    let mean = tokens.mean_axis(Axis(0))?; // `[dim]`
    let norm = mean.dot(&mean).sqrt();
    if norm <= f32::EPSILON {
        return None;
    }
    Some(mean.iter().map(|&x| x / norm).collect())
}

/// Atomically write the postcard-encoded doc→chunk `mapping` to `path`: write a
/// PID-suffixed temp file in the same directory, then rename (atomic on the
/// same filesystem) — mirrors `dense_backend::write_active_pointer`'s pattern
/// so a crash mid-write can never leave a torn `plaid_mapping.bin` for the
/// next open to trip over.
fn write_mapping_atomic(path: &Path, mapping: &[u64]) -> Result<()> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(dir)?;
    let tmp_path = dir.join(format!(".plaid_mapping.{}.tmp", std::process::id()));
    std::fs::write(&tmp_path, postcard::to_stdvec(mapping)?)?;
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Build-time `colbert-plaid` index builder. Owns the PLAID full-rebuild and
/// incremental update logic lifted verbatim from `index/builder.rs` so the
/// dense build path routes through `DenseIndexBuilder`.
///
/// `model_dir` is resolved lazily on first `build`/`insert` (the embedder
/// construction defers the ONNX session). `nbits` is `SemantexConfig.plaid_nbits`.
pub struct ColbertPlaidIndexBuilder {
    plaid_dir: PathBuf,
    mapping_path: PathBuf,
    models_dir: PathBuf,
    nbits: usize,
}

impl ColbertPlaidIndexBuilder {
    /// `index_dir` is the per-backend dense subdir (`dense/colbert-plaid/`).
    /// The mapping sidecar lives at `<index_dir>/plaid_mapping.bin`.
    pub fn new(index_dir: &Path, nbits: usize) -> Self {
        Self {
            plaid_dir: index_dir.to_path_buf(),
            mapping_path: index_dir.join("plaid_mapping.bin"),
            models_dir: crate::config::SemantexConfig::default().models_dir(),
            nbits,
        }
    }

    /// Override the models directory (so the indexer can pass the configured
    /// `SemantexConfig.models_dir()` rather than the global default).
    pub fn with_models_dir(mut self, models_dir: PathBuf) -> Self {
        self.models_dir = models_dir;
        self
    }

    fn next_plaid_configs(&self) -> (next_plaid::IndexConfig, next_plaid::UpdateConfig) {
        let index_config = next_plaid::IndexConfig {
            nbits: self.nbits,
            batch_size: 1024,
            force_cpu: true,
            ..Default::default()
        };
        let update_config = next_plaid::UpdateConfig {
            batch_size: 1024,
            buffer_size: PLAID_BUFFER_SIZE,
            force_cpu: true,
            ..Default::default()
        };
        (index_config, update_config)
    }

    /// Memory-bounded full rebuild that streams chunk content from the store
    /// instead of materializing the entire corpus at once.
    ///
    /// This preserves the hard-won PLAID index-build memory bound (the
    /// 26GB→9GB fix): for each `PLAID_BATCH` (32) slice of `chunk_ids`, the
    /// caller-supplied `fetch` pulls only that batch's content (≤32 ids — well
    /// under SQLite's `SQLITE_MAX_VARIABLE_NUMBER` default of 999), we encode
    /// it, accumulate the compact token embeddings, then DROP the batch content
    /// before the next iteration. Only one batch's content is ever live; the
    /// accumulated embeddings are far smaller than the source text.
    ///
    /// Crucially we still issue exactly ONE `update_or_create` over all
    /// embeddings — identical to the pre-seam single-call strategy — so the
    /// resulting PLAID index is byte-identical (repeated `update_or_create`
    /// calls would re-run k-means per batch, both perturbing centroids and
    /// reintroducing the ~10×-per-call RSS blowup the single call avoids).
    ///
    /// `fetch(batch_ids)` must return `(chunk_id, content)` pairs for the given
    /// ids; order need not match (we read the id back from each pair).
    pub fn build_streaming_ids<F>(&mut self, chunk_ids: &[u64], mut fetch: F) -> Result<()>
    where
        F: FnMut(&[u64]) -> Result<Vec<(u64, String)>>,
    {
        use next_plaid::MmapIndex;

        if chunk_ids.is_empty() {
            tracing::info!("No chunks to encode for PLAID index");
            return Ok(());
        }

        let model_dir = model_manager::ensure_colbert_model(&self.models_dir)?;
        let embedder = ColbertEmbedder::for_indexing(&model_dir)?;
        let (index_config, update_config) = self.next_plaid_configs();
        let plaid_dir_str = self.plaid_dir.to_string_lossy().into_owned();

        if self.plaid_dir.exists() {
            let _ = std::fs::remove_dir_all(&self.plaid_dir);
        }
        std::fs::create_dir_all(&self.plaid_dir)?;

        // Stream content batch-by-batch (memory bound) and accumulate only the
        // compact embeddings; then ONE update_or_create over everything.
        let mut full_mapping: Vec<u64> = Vec::with_capacity(chunk_ids.len());
        let mut all_embeddings: Vec<_> = Vec::with_capacity(chunk_ids.len());
        for batch_ids in chunk_ids.chunks(PLAID_BATCH) {
            if let Err(e) = crate::memory::check_rss_or_abort("PLAID encode batch") {
                anyhow::bail!("Indexing aborted: {e}");
            }
            // Fetch only this batch's content from the store (≤32 ids).
            let batch = fetch(batch_ids)?;
            if batch.is_empty() {
                continue;
            }
            let contents: Vec<String> = batch.iter().map(|(_, c)| c.clone()).collect();
            let embeddings = embedder.encode_documents(&contents)?;
            all_embeddings.extend(embeddings);
            full_mapping.extend(batch.iter().map(|(id, _)| *id));
            // `batch` (and `contents`) drop here → only one batch's content is
            // ever live at a time.
        }

        if all_embeddings.is_empty() {
            tracing::info!("No chunks to encode for PLAID index");
            return Ok(());
        }

        if let Err(e) = crate::memory::check_rss_or_abort("PLAID build (single call)") {
            anyhow::bail!("Indexing aborted: {e}");
        }
        let (_index, plaid_doc_ids) = MmapIndex::update_or_create(
            &all_embeddings,
            &plaid_dir_str,
            &index_config,
            &update_config,
        )?;
        anyhow::ensure!(
            plaid_doc_ids.len() == full_mapping.len(),
            "PLAID returned {} doc IDs for {} chunks — contract violated",
            plaid_doc_ids.len(),
            full_mapping.len(),
        );
        drop(all_embeddings);
        crate::memory::purge_allocator();

        write_mapping_atomic(&self.mapping_path, &full_mapping)?;
        tracing::info!("PLAID index built ({} chunks)", full_mapping.len());
        Ok(())
    }

    /// Memory-bounded incremental insert that streams content from the store.
    ///
    /// Mirrors the per-batch encode + `update_or_create` loop of the trait
    /// `insert`, but pulls each `PLAID_BATCH` (32) slice's content via `fetch`
    /// so a large incremental add never materializes all new content at once
    /// and never passes >32 ids to a single `get_chunks` call.
    pub fn insert_streaming_ids<F>(&mut self, chunk_ids: &[u64], mut fetch: F) -> Result<()>
    where
        F: FnMut(&[u64]) -> Result<Vec<(u64, String)>>,
    {
        use next_plaid::MmapIndex;

        if chunk_ids.is_empty() {
            return Ok(());
        }
        let model_dir = model_manager::ensure_colbert_model(&self.models_dir)?;
        let embedder = ColbertEmbedder::for_indexing(&model_dir)?;
        let (index_config, update_config) = self.next_plaid_configs();
        let plaid_dir_str = self.plaid_dir.to_string_lossy().into_owned();

        let mut mapping: Vec<u64> = if self.mapping_path.exists() {
            postcard::from_bytes::<Vec<u64>>(&std::fs::read(&self.mapping_path)?)?
        } else {
            Vec::new()
        };

        for batch_ids in chunk_ids.chunks(PLAID_BATCH) {
            if let Err(e) = crate::memory::check_rss_or_abort("PLAID incremental batch") {
                anyhow::bail!("Indexing aborted: {e}");
            }
            let batch = fetch(batch_ids)?;
            if batch.is_empty() {
                continue;
            }
            let contents: Vec<String> = batch.iter().map(|(_, c)| c.clone()).collect();
            let embeddings = embedder.encode_documents(&contents)?;
            let (_index, plaid_doc_ids) = MmapIndex::update_or_create(
                &embeddings,
                &plaid_dir_str,
                &index_config,
                &update_config,
            )?;
            anyhow::ensure!(
                plaid_doc_ids.len() == batch.len(),
                "PLAID returned {} doc IDs for {} chunks — contract violated",
                plaid_doc_ids.len(),
                batch.len(),
            );
            for (&doc_id, (chunk_id, _)) in plaid_doc_ids.iter().zip(batch.iter()) {
                anyhow::ensure!(doc_id >= 0, "PLAID returned negative doc_id {doc_id}");
                let idx = doc_id as usize;
                while mapping.len() <= idx {
                    mapping.push(DENSE_TOMBSTONE);
                }
                mapping[idx] = *chunk_id;
            }
        }

        write_mapping_atomic(&self.mapping_path, &mapping)?;
        Ok(())
    }
}

impl DenseIndexBuilder for ColbertPlaidIndexBuilder {
    fn name(&self) -> &'static str {
        ColbertPlaidBackend::NAME
    }

    fn build(&mut self, chunks: &[(u64, &str)]) -> Result<()> {
        // Full rebuild over an already-materialized corpus. Delegate to the
        // streaming core (single `update_or_create`, byte-identical to the
        // pre-seam path) by serving each batch straight from the in-memory
        // slice. Memory-sensitive callers (the indexer) use
        // `build_streaming_ids` so the corpus content is never all live at
        // once; this slice-based path exists for direct/tests callers.
        let ids: Vec<u64> = chunks.iter().map(|(id, _)| *id).collect();
        let by_id: std::collections::HashMap<u64, &str> =
            chunks.iter().map(|(id, c)| (*id, *c)).collect();
        self.build_streaming_ids(&ids, |batch_ids| {
            Ok(batch_ids
                .iter()
                .filter_map(|id| by_id.get(id).map(|c| (*id, (*c).to_string())))
                .collect())
        })
    }

    fn insert(&mut self, chunks: &[(u64, &str)]) -> Result<()> {
        let ids: Vec<u64> = chunks.iter().map(|(id, _)| *id).collect();
        let by_id: std::collections::HashMap<u64, &str> =
            chunks.iter().map(|(id, c)| (*id, *c)).collect();
        self.insert_streaming_ids(&ids, |batch_ids| {
            Ok(batch_ids
                .iter()
                .filter_map(|id| by_id.get(id).map(|c| (*id, (*c).to_string())))
                .collect())
        })
    }

    fn delete(&mut self, chunk_ids: &[u64]) -> Result<()> {
        use next_plaid::MmapIndex;

        if chunk_ids.is_empty() || !self.mapping_path.exists() {
            return Ok(());
        }
        let mut mapping: Vec<u64> =
            postcard::from_bytes::<Vec<u64>>(&std::fs::read(&self.mapping_path)?)?;
        let removed_set: std::collections::HashSet<u64> = chunk_ids.iter().copied().collect();
        let plaid_delete_ids: Vec<i64> = mapping
            .iter()
            .enumerate()
            .filter_map(|(plaid_id, &cid)| {
                if cid != DENSE_TOMBSTONE && removed_set.contains(&cid) {
                    Some(plaid_id as i64)
                } else {
                    None
                }
            })
            .collect();
        if plaid_delete_ids.is_empty() {
            return Ok(());
        }
        let plaid_dir_str = self.plaid_dir.to_string_lossy().into_owned();
        match MmapIndex::load(&plaid_dir_str) {
            Ok(mut index) => {
                if let Err(e) = index.delete(&plaid_delete_ids) {
                    tracing::warn!("PLAID delete failed: {e}");
                }
            }
            Err(e) => tracing::warn!("PLAID load for delete failed: {e}"),
        }
        for plaid_id in &plaid_delete_ids {
            if let Some(slot) = mapping.get_mut(*plaid_id as usize) {
                *slot = DENSE_TOMBSTONE;
            }
        }
        write_mapping_atomic(&self.mapping_path, &mapping)?;
        Ok(())
    }

    fn persist(&self, _dir: &Path) -> Result<()> {
        // PLAID writes its index + mapping eagerly during build/insert/delete
        // (next-plaid persists to `plaid_dir` on each update_or_create; the
        // mapping is written at the end of each op). Nothing extra to flush.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_name_is_colbert_plaid() {
        assert_eq!(ColbertPlaidBackend::NAME, "colbert-plaid");
    }

    #[test]
    fn index_builder_name_is_colbert_plaid() {
        // A builder pointed at a temp dir; we only assert identity here (a full
        // build needs the ColBERT model, exercised by the golden test in
        // tests/dense_backend_golden_test.rs).
        let tmp = tempfile::TempDir::new().unwrap();
        let b = ColbertPlaidIndexBuilder::new(tmp.path(), 4);
        assert_eq!(DenseIndexBuilder::name(&b), "colbert-plaid");
    }

    #[test]
    fn empty_build_writes_nothing_and_is_ok() {
        // Building with zero chunks must be a no-op success (mirrors the
        // `all_ids.is_empty()` early return in the inline PLAID code).
        let tmp = tempfile::TempDir::new().unwrap();
        let mut b = ColbertPlaidIndexBuilder::new(tmp.path(), 4);
        b.build(&[]).unwrap();
        // No mapping file is written for an empty corpus.
        assert!(!tmp.path().join("plaid_mapping.bin").exists());
    }

    #[test]
    fn plaid_batch_strictly_below_buffer_size() {
        // Compile-time invariant: PLAID_BATCH must stay below PLAID_BUFFER_SIZE so
        // single-file incremental updates hit the buffer-only fast path.
        const { assert!(PLAID_BATCH < PLAID_BUFFER_SIZE) };
    }

    /// S1/S7 seam: `embed_text_vector` returns a `Some(Vec<f32>)` of the model
    /// dimension (48) for a non-empty query. Opening a real backend needs a
    /// PLAID index + the ColBERT model, so this is `#[ignore]`'d (run with
    /// `--ignored`); it builds a tiny synthetic repo (repo-agnostic tempdir,
    /// no hardcoded paths) and opens the colbert-plaid backend from the
    /// per-backend dense subdir.
    #[test]
    fn write_mapping_atomic_leaves_no_temp_file_and_round_trips() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("plaid_mapping.bin");
        let mapping: Vec<u64> = vec![10, 20, DENSE_TOMBSTONE, 30];
        write_mapping_atomic(&path, &mapping).unwrap();

        let back: Vec<u64> = postcard::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(back, mapping);

        // No leftover .tmp file in the directory after a successful write.
        let leftovers: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "atomic write must not leave a temp file behind: {leftovers:?}"
        );
    }

    #[test]
    #[ignore = "builds a PLAID index + loads the ColBERT model; run with --ignored"]
    fn embed_text_vector_returns_some_with_model_dim() {
        use crate::config::SemantexConfig;
        use crate::index::builder::IndexBuilder;
        use crate::search::dense_backend::{DenseBackendKind, dense_subdir};

        let tmp = tempfile::TempDir::new().unwrap();
        let project = tmp.path().join("repo");
        std::fs::create_dir_all(&project).unwrap();
        std::fs::write(project.join("a.rs"), "pub fn hello() -> u32 { 41 + 1 }\n").unwrap();

        let cfg = SemantexConfig::default();
        IndexBuilder::new(&cfg).unwrap().build(&project).unwrap();

        let index_dir = project.join(".semantex");
        let dense_dir = dense_subdir(&index_dir, DenseBackendKind::ColbertPlaid);
        let mapping_path = dense_dir.join("plaid_mapping.bin");
        let model_dir =
            crate::embedding::model_manager::ensure_colbert_model(&cfg.models_dir()).unwrap();

        let backend = ColbertPlaidBackend::open(&dense_dir, &mapping_path, &model_dir).unwrap();
        let v = backend
            .embed_text_vector("open a database connection")
            .expect("query projection must be Some for a non-empty query");
        // ColBERT model dim is 48 (see colbert.rs / IndexMeta.embedding_dim).
        assert_eq!(
            v.len(),
            48,
            "mean-pooled query vector must have the model dim"
        );
        // L2-normalized → unit length (within float tolerance).
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "projection must be L2-normalized, got norm={norm}"
        );
    }
}
