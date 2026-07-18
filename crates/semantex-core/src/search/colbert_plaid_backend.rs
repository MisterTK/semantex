//! `colbert-plaid` — the first `DenseBackend`/`DenseIndexBuilder` impl,
//! wrapping the ColBERT late-interaction + vendored next-plaid PLAID path.
//! Behavior is byte-identical to the pre-seam inline PLAID code.

use crate::embedding::colbert::{ColbertEmbedder, TokenEmbeddings};
use crate::embedding::model_manager;
use crate::embedding::static_token::StaticTokenEmbedder;
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

/// Number of chunks accumulated for the initial `update_or_create` call on a
/// fresh build — comfortably above next-plaid's `start_from_scratch` default
/// (999 docs), so real k-means training happens once and `num_documents` is
/// past the from-scratch-retrain threshold before any further batch lands.
/// Every later batch in `build_streaming_ids` streams via `update_append`
/// against the resulting frozen codec instead of retraining, bounding
/// fresh-build peak memory to ~this many chunks' embeddings regardless of
/// total corpus size (see `build_streaming_ids`'s doc comment).
const INITIAL_BUILD_CHUNKS: usize = 2_000;

/// Batch size for Phase B's `update_append` calls in `build_streaming_ids`.
/// Deliberately much larger than `PLAID_BATCH`: next-plaid's
/// `update_append`/`update_index` reloads the ENTIRE on-disk IVF (centroid →
/// doc-postings list) into memory, merges in the new batch, and rewrites it —
/// an O(current-index-size) cost on every single call, not O(batch). At
/// `PLAID_BATCH` (32) granularity a large corpus makes thousands of these
/// read-modify-write cycles back to back with no allocator purge between
/// them; empirically this drove RSS to the soft cap on a real 157k-chunk repo
/// in under 90 seconds despite each call's own logical footprint being
/// modest (pure fragmentation/churn, not a logical memory requirement).
/// Batching this much larger cuts the call count — and the IVF-reload
/// overhead — by ~16x for a corpus that size. Capped comfortably under
/// SQLite's assumed `SQLITE_MAX_VARIABLE_NUMBER` (999; see `PLAID_BATCH`'s
/// sibling comment and `ChunkStore::get_chunks`'s single `IN (...)` query,
/// which does not itself sub-batch) since `fetch` ultimately runs one query
/// per batch of this size.
const PLAID_APPEND_BATCH: usize = 512;

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

/// Document-side encoder selector for index builds (Ember Plan A, Task 6).
///
/// Defaults to the full contextual [`ColbertEmbedder`]. When
/// `SEMANTEX_STATIC_DOC_EMBED` is set AND a distilled static token table
/// loads successfully from the model directory, uses the tier-0
/// encoder-free [`StaticTokenEmbedder`] instead — table lookups + a fixed
/// mixing formula, no ONNX session per document. This ONLY affects how
/// document-side embeddings are produced during an index build
/// (`build_streaming_ids` / `insert_streaming_ids`); **queries and the
/// search path never construct this type and are completely unaffected**.
///
/// See `ModelRegistry::active_embedder_fingerprint`'s doc comment (in
/// `model/registry.rs`) for why `SEMANTEX_STATIC_DOC_EMBED` is deliberately
/// excluded from the embedder fingerprint that gates automatic re-embeds.
enum DocEncoderKind {
    Contextual(ColbertEmbedder),
    Static(StaticTokenEmbedder),
}

impl DocEncoderKind {
    /// Encode documents into per-token embeddings — same output shape
    /// contract regardless of which variant is active, so both build call
    /// sites stay a single code path.
    fn encode_documents(&self, texts: &[String]) -> Result<Vec<TokenEmbeddings>> {
        match self {
            DocEncoderKind::Contextual(embedder) => embedder.encode_documents(texts),
            DocEncoderKind::Static(embedder) => embedder.encode_documents(texts),
        }
    }

    /// Select the document encoder for an index build rooted at `model_dir`.
    ///
    /// `SEMANTEX_STATIC_DOC_EMBED` unset (the default) always returns
    /// [`DocEncoderKind::Contextual`]. When the flag is set, this attempts to
    /// load a [`StaticTokenEmbedder`] from `model_dir`; if that fails for ANY
    /// reason (the distilled `static_token_table.bin` is missing/corrupt, or
    /// the tokenizer files it also needs aren't present), that is **not** a
    /// hard error — it logs `tracing::warn!` and falls back to the
    /// contextual encoder, exactly as if the flag had never been set. This
    /// is a deliberate safety property: flipping an experimental env var
    /// must never turn a working index build into a failed one.
    fn for_indexing(model_dir: &Path) -> Result<Self> {
        if crate::config::env_bool("SEMANTEX_STATIC_DOC_EMBED") {
            match StaticTokenEmbedder::new(model_dir) {
                Ok(embedder) => return Ok(DocEncoderKind::Static(embedder)),
                Err(e) => {
                    tracing::warn!(
                        "SEMANTEX_STATIC_DOC_EMBED is set but the static token table \
                         failed to load from {} ({e:#}); falling back to the contextual \
                         ColBERT encoder for this build",
                        model_dir.display()
                    );
                }
            }
        }
        Ok(DocEncoderKind::Contextual(ColbertEmbedder::for_indexing(
            model_dir,
        )?))
    }
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
    /// Split into two phases so peak memory is bounded by `INITIAL_BUILD_CHUNKS`
    /// instead of total corpus size (a from-scratch build over a very large repo
    /// used to accumulate every chunk's embeddings before a single call — see
    /// git history for the incident this replaced):
    ///
    ///   * Phase A: stream-encode the first `INITIAL_BUILD_CHUNKS` chunk ids
    ///     (batches of `PLAID_BATCH`, content never fully materialized — the
    ///     original 26GB→9GB fix) and issue ONE `update_or_create` call. This
    ///     creates the index with real k-means training and pushes
    ///     `num_documents` past next-plaid's `start_from_scratch` threshold, so
    ///     later batches don't retrain from scratch.
    ///   * Phase B: stream any remaining chunk ids in `PLAID_BATCH`-sized
    ///     batches via `update_append` instead of `update_or_create` — it
    ///     quantizes against the already-trained, now-frozen codec and appends,
    ///     with no retraining and no per-call merged-file regeneration (that
    ///     cost is deferred to the finalize step below; see `update_append`'s
    ///     own doc comment in next-plaid). Memory per call is bounded by one
    ///     batch, independent of corpus size.
    ///   * Finalize: if Phase B ran, one `MmapIndex::load` call forces the
    ///     deferred merge to happen now (at build time, logged) rather than
    ///     silently on the first search.
    ///
    /// For corpora at or below `INITIAL_BUILD_CHUNKS`, Phase B is a no-op — this
    /// is a strict generalization of the old single-call behavior, not a
    /// size-gated special case.
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
        let embedder = DocEncoderKind::for_indexing(&model_dir)?;
        let (index_config, update_config) = self.next_plaid_configs();
        let plaid_dir_str = self.plaid_dir.to_string_lossy().into_owned();

        if self.plaid_dir.exists() {
            let _ = std::fs::remove_dir_all(&self.plaid_dir);
        }
        std::fs::create_dir_all(&self.plaid_dir)?;

        let split = chunk_ids.len().min(INITIAL_BUILD_CHUNKS);
        let (initial_ids, remainder_ids) = chunk_ids.split_at(split);

        // Phase A: bounded initial accumulation + ONE update_or_create call
        // (identical in shape to the old single-call full build, just scoped
        // to a bounded prefix instead of the whole corpus).
        let mut full_mapping: Vec<u64> = Vec::with_capacity(chunk_ids.len());
        let mut all_embeddings: Vec<_> = Vec::with_capacity(initial_ids.len());
        for batch_ids in initial_ids.chunks(PLAID_BATCH) {
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

        if let Err(e) = crate::memory::check_rss_or_abort("PLAID build (initial call)") {
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

        // Phase B: stream any remainder against the now-frozen codec, in
        // PLAID_APPEND_BATCH-sized batches — see that constant's doc comment
        // for why this must NOT use PLAID_BATCH's much smaller granularity.
        let mut appended_so_far: usize = 0;
        for batch_ids in remainder_ids.chunks(PLAID_APPEND_BATCH) {
            if let Err(e) = crate::memory::check_rss_or_abort("PLAID append batch") {
                anyhow::bail!("Indexing aborted: {e}");
            }
            let batch = fetch(batch_ids)?;
            if batch.is_empty() {
                continue;
            }
            let contents: Vec<String> = batch.iter().map(|(_, c)| c.clone()).collect();
            let embeddings = embedder.encode_documents(&contents)?;
            let plaid_doc_ids =
                MmapIndex::update_append(&embeddings, &plaid_dir_str, &update_config)?;
            anyhow::ensure!(
                plaid_doc_ids.len() == batch.len(),
                "PLAID returned {} doc IDs for {} chunks — contract violated",
                plaid_doc_ids.len(),
                batch.len(),
            );
            for (&doc_id, (chunk_id, _)) in plaid_doc_ids.iter().zip(batch.iter()) {
                anyhow::ensure!(doc_id >= 0, "PLAID returned negative doc_id {doc_id}");
                let idx = doc_id as usize;
                while full_mapping.len() <= idx {
                    full_mapping.push(DENSE_TOMBSTONE);
                }
                full_mapping[idx] = *chunk_id;
            }
            // `update_append`'s IVF reload/rewrite churns through a
            // moderately large ephemeral HashMap<usize, Vec<i64>> every call
            // (see PLAID_APPEND_BATCH's doc comment) — purge proactively
            // instead of waiting for `check_rss_or_abort` to catch fragmented
            // growth after the fact.
            crate::memory::purge_allocator();
            appended_so_far += batch.len();
            if let Some(rss_mb) = crate::memory::current_rss_mb() {
                // Diagnostic visibility into update_append's per-call cost,
                // which scales with total corpus size already appended (the
                // on-disk IVF is reloaded/rewritten whole every call — see
                // PLAID_APPEND_BATCH's doc comment). Cheap (one log line per
                // batch); lets a future large-repo failure be root-caused
                // from the growth curve instead of just the final RSS number.
                tracing::info!(
                    appended_so_far,
                    remainder_total = remainder_ids.len(),
                    rss_mb,
                    "PLAID append batch progress"
                );
            }
        }

        // `update_append` intentionally defers merged-code/residual
        // regeneration (potentially 628MB+ on large indices — see its doc
        // comment in next-plaid) to the next `load`/search. Force it now, at
        // build time, so the cost is paid here (logged) instead of silently
        // on the user's first query.
        if !remainder_ids.is_empty() {
            MmapIndex::load(&plaid_dir_str)?;
        }

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
        let embedder = DocEncoderKind::for_indexing(&model_dir)?;
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

    // ── DocEncoderKind::for_indexing selection (Task 6) ─────────────────────

    /// Private to this test module: serializes every test that mutates
    /// `SEMANTEX_STATIC_DOC_EMBED`, mirroring `index::state::DENSE_CONTEXT_ENV_LOCK`'s
    /// pattern for this crate's other single-flag/single-file env vars.
    static STATIC_DOC_EMBED_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Run `f` with `SEMANTEX_STATIC_DOC_EMBED` set to `val` (or unset when
    /// `None`), restoring whatever was there before, serialized by
    /// `STATIC_DOC_EMBED_ENV_LOCK`.
    fn with_static_doc_embed_env<F: FnOnce()>(val: Option<&str>, f: F) {
        let _guard = STATIC_DOC_EMBED_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = std::env::var("SEMANTEX_STATIC_DOC_EMBED").ok();
        // SAFETY: guarded by STATIC_DOC_EMBED_ENV_LOCK.
        unsafe {
            match val {
                Some(v) => std::env::set_var("SEMANTEX_STATIC_DOC_EMBED", v),
                None => std::env::remove_var("SEMANTEX_STATIC_DOC_EMBED"),
            }
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        // SAFETY: guarded by STATIC_DOC_EMBED_ENV_LOCK.
        unsafe {
            match prior {
                Some(v) => std::env::set_var("SEMANTEX_STATIC_DOC_EMBED", v),
                None => std::env::remove_var("SEMANTEX_STATIC_DOC_EMBED"),
            }
        }
        if let Err(e) = result {
            std::panic::resume_unwind(e);
        }
    }

    /// Locate the local LateOn-Code-edge tokenizer files, if the model has
    /// been downloaded. Mirrors `static_token.rs`'s `test_tokenizer_dir`
    /// gating: skip (not fail) the one test that needs a real tokenizer to
    /// build a loadable `StaticTokenEmbedder`, so this file's other tests
    /// always run in CI without the model present.
    fn test_tokenizer_dir() -> Option<std::path::PathBuf> {
        let dir = crate::config::SemantexConfig::default()
            .models_dir()
            .join("LateOn-Code-edge");
        (dir.join("tokenizer.json").exists() && dir.join("onnx_config.json").exists())
            .then_some(dir)
    }

    #[test]
    fn doc_encoder_kind_defaults_to_contextual_when_flag_unset() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_static_doc_embed_env(None, || {
            let kind = DocEncoderKind::for_indexing(tmp.path())
                .expect("contextual construction only requires the dir to exist");
            assert!(
                matches!(kind, DocEncoderKind::Contextual(_)),
                "flag unset must always select the contextual encoder"
            );
        });
    }

    #[test]
    fn doc_encoder_kind_falls_back_to_contextual_when_flag_set_but_table_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_static_doc_embed_env(Some("1"), || {
            // No static_token_table.bin (or tokenizer.json/onnx_config.json)
            // under tmp.path() — StaticTokenEmbedder::new must fail, and
            // for_indexing must fall back rather than propagate that error.
            let kind = DocEncoderKind::for_indexing(tmp.path())
                .expect("a set flag with a missing table must fall back, never error");
            assert!(
                matches!(kind, DocEncoderKind::Contextual(_)),
                "missing table must fall back to the contextual encoder"
            );
        });
    }

    #[test]
    fn doc_encoder_kind_selects_static_when_flag_set_and_table_present() {
        let Some(model_dir) = test_tokenizer_dir() else {
            return;
        };
        let vocab_size = ColbertEmbedder::new(&model_dir)
            .unwrap()
            .tokenizer_vocab_size()
            .unwrap();
        let table = crate::embedding::static_table::StaticTokenTable::new(
            vocab_size,
            4,
            [0.1, 0.2, 0.4, 0.2, 0.1],
        );
        let tmp = tempfile::TempDir::new().unwrap();
        table
            .save(&tmp.path().join("static_token_table.bin"))
            .unwrap();
        std::fs::copy(
            model_dir.join("tokenizer.json"),
            tmp.path().join("tokenizer.json"),
        )
        .unwrap();
        std::fs::copy(
            model_dir.join("onnx_config.json"),
            tmp.path().join("onnx_config.json"),
        )
        .unwrap();

        with_static_doc_embed_env(Some("1"), || {
            let kind = DocEncoderKind::for_indexing(tmp.path())
                .expect("a valid table + flag must succeed");
            assert!(
                matches!(kind, DocEncoderKind::Static(_)),
                "flag set with a loadable table must select the static encoder"
            );
        });
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
            .filter_map(std::result::Result::ok)
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

    /// Exercises both phases of `build_streaming_ids`: a corpus larger than
    /// `INITIAL_BUILD_CHUNKS` forces Phase A (bounded `update_or_create`) AND
    /// Phase B (streamed `update_append` for the remainder) to run across
    /// several `PLAID_APPEND_BATCH`-sized iterations, verifying the resulting
    /// mapping covers every chunk exactly once and the index is searchable
    /// afterward. Multiple Phase B iterations matter here: a real-repo bug
    /// (next-plaid's `update_append` reloading/rewriting the full on-disk IVF
    /// every call) only showed up after many calls, not one. `#[ignore]`'d
    /// like its sibling above — needs the real ColBERT model and is slow.
    #[test]
    #[ignore = "builds a >INITIAL_BUILD_CHUNKS PLAID index + loads the ColBERT model; run with --ignored"]
    fn build_streaming_ids_covers_bounded_initial_and_streamed_remainder() {
        use crate::config::SemantexConfig;

        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = SemantexConfig::default();
        let model_dir =
            crate::embedding::model_manager::ensure_colbert_model(&cfg.models_dir()).unwrap();

        // A few full PLAID_APPEND_BATCH iterations past INITIAL_BUILD_CHUNKS,
        // so Phase B loops multiple times instead of completing in one call.
        let total = INITIAL_BUILD_CHUNKS + PLAID_APPEND_BATCH * 3 + 1;
        let chunk_ids: Vec<u64> = (1..=total as u64).collect();

        let mut builder = ColbertPlaidIndexBuilder::new(tmp.path(), 4);
        builder
            .build_streaming_ids(&chunk_ids, |batch_ids| {
                Ok(batch_ids
                    .iter()
                    .map(|&id| (id, format!("fn chunk_{id}() {{ return {id}; }}")))
                    .collect())
            })
            .unwrap();

        let mapping_path = tmp.path().join("plaid_mapping.bin");
        let mapping: Vec<u64> =
            postcard::from_bytes(&std::fs::read(&mapping_path).unwrap()).unwrap();
        assert_eq!(
            mapping.len(),
            total,
            "mapping must cover every chunk across both phases"
        );
        let mut seen: Vec<u64> = mapping
            .iter()
            .copied()
            .filter(|&c| c != DENSE_TOMBSTONE)
            .collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            seen.len(),
            total,
            "every chunk id must appear exactly once across the phased build"
        );

        let backend = ColbertPlaidBackend::open(tmp.path(), &mapping_path, &model_dir).unwrap();
        let hits = backend.search("chunk_1", 5).unwrap();
        assert!(
            !hits.is_empty(),
            "search must return hits after a phased build"
        );
    }
}
