//! `colbert-plaid` — the first `DenseBackend`/`DenseIndexBuilder` impl,
//! wrapping the ColBERT late-interaction + vendored next-plaid PLAID path.
//! Behavior is byte-identical to the pre-seam inline PLAID code.

use crate::embedding::cinder::CinderEncoder;
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

/// Test-only: serializes every test ACROSS THE CRATE that mutates
/// `SEMANTEX_CINDER` — this file's `cinder_for_build` tests AND
/// `model::spec`'s Cinder fingerprint-invariance test. Deliberately
/// module-scoped and `pub(crate)` (unlike the private-to-`mod tests`
/// `STATIC_DOC_EMBED_ENV_LOCK` / `FROZEN_CENTROIDS_ENV_LOCK`) precisely because
/// a second module needs the SAME lock instance: two separate locks would not
/// serialize the single shared process env var, reintroducing the race.
#[cfg(test)]
pub(crate) static CINDER_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

/// Resolve the frozen-centroids artifact for an index build (Ember Plan B).
///
/// Opt-in and safe by construction: returns `Some` ONLY when
/// `SEMANTEX_FROZEN_CENTROIDS` is set (same `env_bool` contract as
/// `SEMANTEX_STATIC_DOC_EMBED`) AND `frozen_centroids.npy` exists in
/// `model_dir`. Flag set but artifact missing warns and returns `None` — the
/// build proceeds on the current per-repo k-means path, never errors. Load
/// validation (npy parse, dim match) happens inside next-plaid, which falls
/// back to training on any problem, so a corrupt artifact also cannot fail a
/// build. Build-time only: queries never read this; excluded from the
/// embedder fingerprint for the same reason as SEMANTEX_STATIC_DOC_EMBED
/// (see model/registry.rs).
fn frozen_centroids_for_build(model_dir: &Path) -> Option<PathBuf> {
    if !crate::config::env_bool("SEMANTEX_FROZEN_CENTROIDS") {
        return None;
    }
    let path = model_manager::frozen_centroids_path(model_dir);
    if path.exists() {
        tracing::info!("using frozen universal centroids: {}", path.display());
        Some(path)
    } else {
        tracing::warn!(
            "SEMANTEX_FROZEN_CENTROIDS is set but {} is missing; \
             falling back to per-repo k-means for this build",
            path.display()
        );
        None
    }
}

/// Resolve the Cinder compiled-indexing encoder for a fresh index build
/// (spec §4, plan Task 6).
///
/// Opt-in and safe by construction, mirroring `frozen_centroids_for_build` /
/// `DocEncoderKind::for_indexing`: returns `Some` ONLY when `SEMANTEX_CINDER`
/// is set AND all three Cinder artifacts (`static_token_table.bin`,
/// `cinder_mixer.bin`, `cinder_shortlists.bin`) load cleanly from `model_dir`.
/// Flag unset → `None` before any load is attempted. Flag set but any artifact
/// missing/corrupt → `tracing::warn!` (naming the failed artifact, via
/// `CinderEncoder::new`'s contextualized error) and `None`. Either way the
/// existing tier chain in `build_streaming_ids` (contextual / static-embed
/// flag / frozen-centroids flag) proceeds untouched. This NEVER errors — a
/// flipped experimental flag must never turn a working build into a failed one.
///
/// Build-time only: queries never construct this, and `SEMANTEX_CINDER` is
/// excluded from the embedder fingerprint (see `model/registry.rs`), so
/// toggling it never forces a re-embed.
fn cinder_for_build(model_dir: &Path) -> Option<CinderEncoder> {
    if !crate::config::env_bool("SEMANTEX_CINDER") {
        return None;
    }
    match CinderEncoder::new(model_dir) {
        Ok(encoder) => {
            tracing::info!("using cinder compiled indexing");
            Some(encoder)
        }
        Err(e) => {
            tracing::warn!(
                "SEMANTEX_CINDER is set but the Cinder artifacts failed to load from {} \
                 ({e:#}); falling back to the existing PLAID build path for this build",
                model_dir.display()
            );
            None
        }
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

    fn next_plaid_configs(
        &self,
        frozen_centroids: Option<&Path>,
    ) -> (next_plaid::IndexConfig, next_plaid::UpdateConfig) {
        let external_centroids_npy = frozen_centroids.map(|p| p.to_string_lossy().into_owned());
        let index_config = next_plaid::IndexConfig {
            nbits: self.nbits,
            batch_size: 1024,
            force_cpu: true,
            external_centroids_npy: external_centroids_npy.clone(),
            ..Default::default()
        };
        let update_config = next_plaid::UpdateConfig {
            batch_size: 1024,
            buffer_size: PLAID_BUFFER_SIZE,
            force_cpu: true,
            external_centroids_npy,
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

        // Cinder compiled-indexing fast path (fresh builds ONLY; the incremental
        // `insert_streaming_ids` is untouched). Opt-in via `SEMANTEX_CINDER` and
        // gated on all three Cinder artifacts AND frozen centroids. Any missing
        // artifact / failed load falls back to the existing tier chain below,
        // which is left completely unchanged (this is a pure INSERT — no existing
        // line is modified). Frozen centroids are REQUIRED here (the cinder path
        // never runs per-repo k-means); a missing/corrupt centroid file also
        // falls back rather than erroring.
        if let Some(cinder) = cinder_for_build(&model_dir) {
            let centroids_path = model_manager::frozen_centroids_path(&model_dir);
            match crate::embedding::centroid_train::load_centroids_npy(&centroids_path) {
                Ok(centroids) => return self.build_cinder(chunk_ids, fetch, &cinder, centroids),
                Err(e) => tracing::warn!(
                    "SEMANTEX_CINDER is set and Cinder artifacts loaded, but the frozen \
                     centroids the cinder path REQUIRES failed to load from {} ({e:#}); \
                     falling back to the existing PLAID build path",
                    centroids_path.display()
                ),
            }
        }

        let embedder = DocEncoderKind::for_indexing(&model_dir)?;
        let frozen = frozen_centroids_for_build(&model_dir);
        let (index_config, update_config) = self.next_plaid_configs(frozen.as_deref());
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

    /// Cinder single-pass compiled build (spec §4, plan Task 6): encode every
    /// chunk with the encoder-free [`CinderEncoder`] and stream the per-document
    /// f32 embeddings straight into a [`next_plaid::CompiledIndexWriter`] built
    /// on FROZEN centroids — no per-repo k-means, no `update_or_create` /
    /// `update_append` cycle. Called only from `build_streaming_ids` when
    /// `cinder_for_build` returned `Some` and the frozen centroids loaded.
    ///
    /// Memory: the writer needs a residual-statistics sample at construction, so
    /// the first `INITIAL_BUILD_CHUNKS.min(total)` docs' embeddings are held for
    /// `CompiledIndexWriter::new` and then streamed into it; the remainder streams
    /// one `PLAID_APPEND_BATCH` at a time and the sample is dropped first — so
    /// peak is bounded by that same `INITIAL_BUILD_CHUNKS` prefix (the constant
    /// is reused deliberately: identical bound to the contextual Phase A).
    ///
    /// Assignment: every document is encoded via
    /// [`CinderEncoder::encode_documents_with_window_ids`] and streamed into the
    /// writer through [`next_plaid::CompiledIndexWriter::add_document_with_ids`]
    /// with an [`next_plaid::IdAwareCodeAssigner`] installed — so each token's
    /// centroid is chosen by the shortlist-union argmax
    /// ([`crate::embedding::shortlists::shortlist_argmax`], O(~m·|window|)
    /// candidates) rather than the writer's default exhaustive scan over all
    /// centroids. Residual computation and quantization are untouched, so the
    /// on-disk layout stays byte-compatible with the reference PLAID format
    /// (only WHICH centroid each token maps to differs, within the C4
    /// shortlist-agreement tolerance).
    fn build_cinder<F>(
        &mut self,
        chunk_ids: &[u64],
        mut fetch: F,
        cinder: &CinderEncoder,
        centroids: ndarray::Array2<f32>,
    ) -> Result<()>
    where
        F: FnMut(&[u64]) -> Result<Vec<(u64, String)>>,
    {
        use crate::embedding::shortlists::shortlist_argmax;
        use ndarray::Array1;
        use next_plaid::{CompiledIndexWriter, IdAwareCodeAssigner, IndexConfig};

        if chunk_ids.is_empty() {
            tracing::info!("No chunks to encode for PLAID index");
            return Ok(());
        }

        let plaid_dir_str = self.plaid_dir.to_string_lossy().into_owned();
        if self.plaid_dir.exists() {
            let _ = std::fs::remove_dir_all(&self.plaid_dir);
        }
        std::fs::create_dir_all(&self.plaid_dir)?;

        let config = IndexConfig {
            nbits: self.nbits,
            batch_size: 1024,
            force_cpu: true,
            ..Default::default()
        };

        // Bounded residual-stats sample = the first INITIAL_BUILD_CHUNKS.min(total)
        // docs' embeddings (same constant/bound as the contextual Phase A).
        let split = chunk_ids.len().min(INITIAL_BUILD_CHUNKS);
        let (initial_ids, remainder_ids) = chunk_ids.split_at(split);

        let mut full_mapping: Vec<u64> = Vec::with_capacity(chunk_ids.len());
        // Parallel buffers for the residual-stats sample: the embeddings
        // (needed by `CompiledIndexWriter::new`) and the matching per-token
        // window ids (needed by the shortlist assigner when these same docs are
        // streamed back in). Kept split so `new` gets a `&[Array2<f32>]` with no
        // extra clone, and each doc's window ids stay aligned to its rows.
        let mut sample_emb: Vec<TokenEmbeddings> = Vec::with_capacity(initial_ids.len());
        let mut sample_windows: Vec<Vec<Vec<u32>>> = Vec::with_capacity(initial_ids.len());
        for batch_ids in initial_ids.chunks(PLAID_BATCH) {
            if let Err(e) = crate::memory::check_rss_or_abort("cinder encode (sample)") {
                anyhow::bail!("Indexing aborted: {e}");
            }
            let batch = fetch(batch_ids)?;
            if batch.is_empty() {
                continue;
            }
            let contents: Vec<String> = batch.iter().map(|(_, c)| c.clone()).collect();
            for (emb, windows) in cinder.encode_documents_with_window_ids(&contents)? {
                sample_emb.push(emb);
                sample_windows.push(windows);
            }
            full_mapping.extend(batch.iter().map(|(id, _)| *id));
        }

        if sample_emb.is_empty() {
            tracing::info!("No chunks to encode for PLAID index");
            return Ok(());
        }

        // The real Cinder assignment step: a shortlist-union nearest-centroid
        // argmax (O(~m·|window|) candidates per token) in place of the writer's
        // default exhaustive scan over all centroids. The closure owns its own
        // copy of the frozen centroids + the per-vocab shortlists (both bounded,
        // shared read-only) so it can outlive this stack frame inside the writer.
        //
        // Scratch: `IdAwareCodeAssigner` is `Fn`, so we can't thread a `&mut`
        // scratch buffer across calls — but each closure invocation is one
        // FLUSHED CHUNK, and within it we allocate ONE `Vec<u16>` and reuse it
        // across every row (shortlist_argmax clears+refills it). That's one
        // small alloc per chunk (not per row, not via a RefCell), which is
        // negligible next to the chunk's residual/quantize work.
        // Diagnostic (gate C1 ablation arm 2): SEMANTEX_CINDER_EXACT_ASSIGN=1
        // forces the writer's DEFAULT exhaustive per-centroid scan instead of
        // Cinder's shortlist-union assigner, isolating the assignment
        // approximation from the mixer's contribution. Off by default → the
        // shortlist-union assigner is the production path.
        let exact_assign = crate::config::env_bool("SEMANTEX_CINDER_EXACT_ASSIGN");
        let assigner: Option<IdAwareCodeAssigner> = if exact_assign {
            tracing::info!(
                "SEMANTEX_CINDER_EXACT_ASSIGN=1: forcing exhaustive centroid assignment \
                 (diagnostic; Cinder's shortlist-union assigner disabled)"
            );
            None
        } else {
            let assigner_centroids = centroids.clone();
            let assigner_shortlists = cinder.shortlists().clone();
            Some(Box::new(
                move |batch: &ndarray::Array2<f32>, window_ids: &[Vec<u32>]| {
                    use rayon::prelude::*;
                    let cview = assigner_centroids.view();
                    // Each token's assignment is fully independent and written
                    // into its indexed output slot `r`, so parallelizing across
                    // the chunk's token rows is deterministic — the result is
                    // byte-identical to the serial argmax regardless of thread
                    // count/scheduling (rayon's indexed `collect` preserves row
                    // order). This mirrors the exhaustive path's own rayon
                    // batching (`Codec::compress_into_codes_cpu`); without it the
                    // shortlist scan is single-threaded while exhaustive is not,
                    // which is the whole ~8× C2 regression. `map_init` hands each
                    // worker ONE reusable scratch `Vec<u16>` (cleared+refilled per
                    // row inside `shortlist_argmax`) rather than allocating per
                    // token; the per-thread scratch stays tiny (≤ m·|window| u16),
                    // so peak RSS is unaffected.
                    let out: Vec<usize> = (0..batch.nrows())
                        .into_par_iter()
                        .map_init(Vec::<u16>::new, |scratch, r| {
                            // Batch rows are freshly built in standard (C) layout
                            // by the writer, so `as_slice` is always `Some`.
                            let row = batch.row(r);
                            let e = row
                                .as_slice()
                                .expect("compiled-writer batch rows are contiguous");
                            shortlist_argmax(
                                e,
                                &window_ids[r],
                                &assigner_shortlists,
                                &cview,
                                scratch,
                            )
                        })
                        .collect();
                    Array1::from_vec(out)
                },
            ))
        };

        // Build the writer from the frozen centroids + the sample; install the
        // shortlist assigner unless the diagnostic disabled it (then the writer's
        // default exhaustive scan runs and the buffered window ids are ignored).
        // Feed the WHOLE corpus WITH per-token window ids: the held sample first
        // (same order), then the remainder.
        let mut writer = CompiledIndexWriter::new(&plaid_dir_str, centroids, &config, &sample_emb)?;
        if let Some(assigner) = assigner {
            writer = writer.with_id_aware_assigner(assigner);
        }
        for (emb, windows) in sample_emb.iter().zip(sample_windows.iter()) {
            writer.add_document_with_ids(emb, windows)?;
        }
        drop(sample_emb); // free the bounded prefix before streaming the remainder
        drop(sample_windows);
        crate::memory::purge_allocator();

        for batch_ids in remainder_ids.chunks(PLAID_APPEND_BATCH) {
            if let Err(e) = crate::memory::check_rss_or_abort("cinder encode (stream)") {
                anyhow::bail!("Indexing aborted: {e}");
            }
            let batch = fetch(batch_ids)?;
            if batch.is_empty() {
                continue;
            }
            let contents: Vec<String> = batch.iter().map(|(_, c)| c.clone()).collect();
            let encoded = cinder.encode_documents_with_window_ids(&contents)?;
            for ((emb, windows), (chunk_id, _)) in encoded.iter().zip(batch.iter()) {
                writer.add_document_with_ids(emb, windows)?;
                full_mapping.push(*chunk_id);
            }
            crate::memory::purge_allocator();
        }

        let meta = writer.finalize()?;
        anyhow::ensure!(
            meta.num_documents == full_mapping.len(),
            "cinder build wrote {} docs but mapping has {} entries — contract violated",
            meta.num_documents,
            full_mapping.len(),
        );
        write_mapping_atomic(&self.mapping_path, &full_mapping)?;
        tracing::info!(
            "PLAID index built via cinder ({} chunks)",
            full_mapping.len()
        );
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
        let frozen = frozen_centroids_for_build(&model_dir);
        let (index_config, update_config) = self.next_plaid_configs(frozen.as_deref());
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

    // ── frozen_centroids_for_build / next_plaid_configs (Ember Plan B, Task 6) ──

    /// Private to this test module: serializes every test that mutates
    /// `SEMANTEX_FROZEN_CENTROIDS`, mirroring `STATIC_DOC_EMBED_ENV_LOCK`'s
    /// pattern above for this crate's other single-flag/single-file env vars.
    static FROZEN_CENTROIDS_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Run `f` with `SEMANTEX_FROZEN_CENTROIDS` set to `val` (or unset when
    /// `None`), restoring whatever was there before, serialized by
    /// `FROZEN_CENTROIDS_ENV_LOCK`.
    fn with_frozen_centroids_env<F: FnOnce()>(val: Option<&str>, f: F) {
        let _guard = FROZEN_CENTROIDS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = std::env::var("SEMANTEX_FROZEN_CENTROIDS").ok();
        // SAFETY: guarded by FROZEN_CENTROIDS_ENV_LOCK.
        unsafe {
            match val {
                Some(v) => std::env::set_var("SEMANTEX_FROZEN_CENTROIDS", v),
                None => std::env::remove_var("SEMANTEX_FROZEN_CENTROIDS"),
            }
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        // SAFETY: guarded by FROZEN_CENTROIDS_ENV_LOCK.
        unsafe {
            match prior {
                Some(v) => std::env::set_var("SEMANTEX_FROZEN_CENTROIDS", v),
                None => std::env::remove_var("SEMANTEX_FROZEN_CENTROIDS"),
            }
        }
        if let Err(e) = result {
            std::panic::resume_unwind(e);
        }
    }

    #[test]
    fn frozen_centroids_off_without_env_flag() {
        // flag unset → None even when the artifact exists
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("frozen_centroids.npy"), b"x").unwrap();
        with_frozen_centroids_env(None, || {
            assert!(frozen_centroids_for_build(tmp.path()).is_none());
        });
    }

    #[test]
    fn frozen_centroids_on_with_flag_and_artifact() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("frozen_centroids.npy"), b"x").unwrap();
        with_frozen_centroids_env(Some("1"), || {
            let p = frozen_centroids_for_build(tmp.path()).expect("flag + artifact → Some");
            assert!(p.ends_with("frozen_centroids.npy"));
        });
    }

    #[test]
    fn frozen_centroids_flag_without_artifact_warns_and_returns_none() {
        let tmp = tempfile::TempDir::new().unwrap();
        with_frozen_centroids_env(Some("1"), || {
            assert!(frozen_centroids_for_build(tmp.path()).is_none());
        });
    }

    #[test]
    fn configs_carry_the_centroid_path_on_both_structs() {
        let tmp = tempfile::TempDir::new().unwrap();
        let b = ColbertPlaidIndexBuilder::new(tmp.path(), 4);
        let cpath = tmp.path().join("frozen_centroids.npy");
        let (ic, uc) = b.next_plaid_configs(Some(&cpath));
        assert_eq!(ic.external_centroids_npy.as_deref(), cpath.to_str());
        assert_eq!(uc.external_centroids_npy.as_deref(), cpath.to_str());
        let (ic, uc) = b.next_plaid_configs(None);
        assert!(ic.external_centroids_npy.is_none() && uc.external_centroids_npy.is_none());
    }

    // ── cinder_for_build gating (Task 6) ────────────────────────────────────

    /// Run `f` with `SEMANTEX_CINDER` set to `val` (or unset when `None`),
    /// restoring the prior value, serialized by the crate-wide
    /// [`super::CINDER_ENV_LOCK`] — mirrors `with_frozen_centroids_env` /
    /// `with_static_doc_embed_env`, but uses the shared module-scoped lock so it
    /// also serializes against `model::spec`'s fingerprint-invariance test.
    fn with_cinder_env<F: FnOnce()>(val: Option<&str>, f: F) {
        let _guard = super::CINDER_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = std::env::var("SEMANTEX_CINDER").ok();
        // SAFETY: guarded by CINDER_ENV_LOCK.
        unsafe {
            match val {
                Some(v) => std::env::set_var("SEMANTEX_CINDER", v),
                None => std::env::remove_var("SEMANTEX_CINDER"),
            }
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        // SAFETY: guarded by CINDER_ENV_LOCK.
        unsafe {
            match prior {
                Some(v) => std::env::set_var("SEMANTEX_CINDER", v),
                None => std::env::remove_var("SEMANTEX_CINDER"),
            }
        }
        if let Err(e) = result {
            std::panic::resume_unwind(e);
        }
    }

    /// Write the three Cinder artifact files (contents don't matter for the
    /// flag-off test — the load is never reached) so "artifacts present" is
    /// literally true.
    fn write_placeholder_cinder_artifacts(dir: &Path) {
        for name in [
            "static_token_table.bin",
            "cinder_mixer.bin",
            "cinder_shortlists.bin",
        ] {
            std::fs::write(dir.join(name), b"x").unwrap();
        }
    }

    #[test]
    fn cinder_for_build_off_without_env_flag() {
        // (a) flag unset → None even with all three artifacts present (the flag
        // gate short-circuits before any load is attempted).
        let tmp = tempfile::TempDir::new().unwrap();
        write_placeholder_cinder_artifacts(tmp.path());
        with_cinder_env(None, || {
            assert!(cinder_for_build(tmp.path()).is_none());
        });
    }

    #[test]
    fn cinder_for_build_on_with_flag_but_mixer_missing_returns_none() {
        // (b) flag on + a valid table but NO mixer → the constructor fails
        // (naming cinder_mixer.bin, asserted directly in cinder.rs) and
        // cinder_for_build WARNS and returns None rather than propagating —
        // the existing tier chain must proceed untouched.
        let tmp = tempfile::TempDir::new().unwrap();
        crate::embedding::static_table::StaticTokenTable::new(4, 2, [0.0, 0.0, 1.0, 0.0, 0.0])
            .save(&model_manager::static_token_table_path(tmp.path()))
            .unwrap();
        with_cinder_env(Some("1"), || {
            assert!(
                cinder_for_build(tmp.path()).is_none(),
                "a set flag with a missing mixer must fall back (None), never error"
            );
        });
    }

    /// End-to-end Cinder build (spec §4, plan Task 6 Step 4): train the four
    /// REAL Cinder artifacts from the downloaded teacher over a synthetic
    /// corpus, build a small index with `SEMANTEX_CINDER=1`, and verify
    /// (i) the build succeeds, (ii) `MmapIndex::load` works and `num_documents`
    /// matches, (iii) a distinctive token retrieves its chunk top-1 via the
    /// normal query path. `#[ignore]`'d like this file's other model-gated
    /// tests; the real model dir is never mutated (temp overlay symlinks it in,
    /// only the overlay receives the trained artifacts).
    #[test]
    #[ignore = "requires the downloaded LateOn-Code-edge model and real Cinder artifacts"]
    fn cinder_build_produces_searchable_index() {
        use crate::config::SemantexConfig;
        use crate::embedding::centroid_train::{
            CentroidTrainOptions, save_centroids_npy, train_centroids,
        };
        use crate::embedding::mixer_train::{MixerTrainOptions, train_mixer};
        use crate::embedding::shortlists::CentroidShortlists;
        use crate::embedding::static_distill::distill;

        let real_models_dir = SemantexConfig::default().models_dir();
        let real_model_dir = model_manager::ensure_colbert_model(&real_models_dir)
            .expect("LateOn-Code-edge must be downloaded for this ignored test");

        // Overlay model dir: symlink the real files in so ensure_colbert_model
        // is satisfied without touching the real model dir; only the overlay
        // receives the trained artifacts.
        let overlay = tempfile::TempDir::new().unwrap();
        let overlay_model_dir = overlay.path().join("LateOn-Code-edge");
        std::fs::create_dir_all(&overlay_model_dir).unwrap();
        for file in ["model_int8.onnx", "tokenizer.json", "onnx_config.json"] {
            let src = real_model_dir.join(file);
            let dst = overlay_model_dir.join(file);
            #[cfg(unix)]
            std::os::unix::fs::symlink(&src, &dst).unwrap();
            #[cfg(not(unix))]
            std::fs::copy(&src, &dst).unwrap();
        }

        // Real teacher + synthetic corpus; distill table, train mixer, train
        // frozen centroids, derive shortlists — all four Cinder artifacts.
        let teacher = ColbertEmbedder::for_indexing(&overlay_model_dir).unwrap();
        let corpus: Vec<String> = (0..50)
            .map(|i| format!("fn cinder_helper_{i}(x: i32) -> i32 {{ x * {i} + 1 }}"))
            .collect();

        let table = distill(&teacher, corpus.clone().into_iter(), 8).unwrap();
        table
            .save(&model_manager::static_token_table_path(&overlay_model_dir))
            .unwrap();

        let (mixer, _report) = train_mixer(
            &teacher,
            &table,
            corpus.clone().into_iter(),
            &MixerTrainOptions {
                sample_capacity: 50_000,
                epochs: 2,
                batch: 8,
                lr: 1e-3,
                holdout_frac: 0.1,
            },
        )
        .unwrap();
        mixer
            .save(&model_manager::cinder_mixer_path(&overlay_model_dir))
            .unwrap();

        let centroids = train_centroids(
            &teacher,
            corpus.clone().into_iter(),
            &CentroidTrainOptions {
                num_centroids: 16,
                sample_capacity: 50_000,
                batch: 8,
            },
        )
        .unwrap();
        save_centroids_npy(
            &model_manager::frozen_centroids_path(&overlay_model_dir),
            &centroids,
        )
        .unwrap();

        CentroidShortlists::derive(&table, &centroids.view(), 8)
            .unwrap()
            .save(&model_manager::cinder_shortlists_path(&overlay_model_dir))
            .unwrap();

        // Distinctive chunks to index + search.
        let chunks: Vec<(u64, String)> = (0u64..50)
            .map(|i| {
                (
                    i,
                    format!("fn cinder_helper_{i}(x: i32) -> i32 {{ x * {i} + 1 }}"),
                )
            })
            .collect();
        let chunk_ids: Vec<u64> = chunks.iter().map(|(id, _)| *id).collect();
        let fetch = |batch_ids: &[u64]| -> Result<Vec<(u64, String)>> {
            Ok(batch_ids
                .iter()
                .map(|&id| (id, chunks[id as usize].1.clone()))
                .collect())
        };

        // (i) build with SEMANTEX_CINDER=1.
        let idx_dir = tempfile::TempDir::new().unwrap();
        with_cinder_env(Some("1"), || {
            let mut builder = ColbertPlaidIndexBuilder::new(idx_dir.path(), 4)
                .with_models_dir(overlay.path().to_path_buf());
            builder
                .build_streaming_ids(&chunk_ids, fetch)
                .expect("cinder build must succeed");
        });

        // (ii) MmapIndex::load works and num_documents matches.
        let plaid_dir_str = idx_dir.path().to_string_lossy().into_owned();
        let loaded = next_plaid::MmapIndex::load(&plaid_dir_str).unwrap();
        assert_eq!(loaded.metadata.num_documents, chunks.len());

        // (iii) a distinctive token retrieves its chunk top-1.
        let mapping_path = idx_dir.path().join("plaid_mapping.bin");
        let backend =
            ColbertPlaidBackend::open(idx_dir.path(), &mapping_path, &overlay_model_dir).unwrap();
        let hits = backend.search("cinder_helper_7", 5).unwrap();
        assert!(
            !hits.is_empty(),
            "search must return hits after a cinder build"
        );
        assert_eq!(
            hits[0].chunk_id, 7,
            "the distinctive token must retrieve its own chunk top-1"
        );

        // (iv) Task 6b: confirm the SHORTLIST-UNION assigner genuinely ran (not
        // a silent fall-through to the writer's exhaustive default). Recompute
        // the expected codes independently — same encoder, same frozen
        // centroids, same shortlists, same shortlist_argmax — over the corpus in
        // build order (all 50 docs land in the initial sample, one flushed
        // chunk), and assert the on-disk codes match EXACTLY. If build_cinder
        // had used the exhaustive default, these would diverge wherever the
        // shortlist disagrees with the global argmax (the <100% C4 tolerance).
        {
            use crate::embedding::centroid_train::load_centroids_npy;
            use crate::embedding::shortlists::shortlist_argmax;
            use ndarray::Array1;
            use ndarray_npy::ReadNpyExt;

            let cinder = CinderEncoder::new(&overlay_model_dir).unwrap();
            let centroids =
                load_centroids_npy(&model_manager::frozen_centroids_path(&overlay_model_dir))
                    .unwrap();
            let shortlists = cinder.shortlists();
            let cview = centroids.view();

            let all_texts: Vec<String> = chunk_ids
                .iter()
                .map(|&id| chunks[id as usize].1.clone())
                .collect();
            let encoded = cinder.encode_documents_with_window_ids(&all_texts).unwrap();

            let mut scratch: Vec<u16> = Vec::new();
            let mut expected: Vec<i64> = Vec::new();
            for (emb, windows) in &encoded {
                for (r, row) in emb.rows().into_iter().enumerate() {
                    let e = row.as_slice().unwrap();
                    expected.push(
                        shortlist_argmax(e, &windows[r], shortlists, &cview, &mut scratch) as i64,
                    );
                }
            }

            let mut got: Vec<i64> = Vec::new();
            let mut ci = 0usize;
            loop {
                let p = idx_dir.path().join(format!("{ci}.codes.npy"));
                if !p.exists() {
                    break;
                }
                let arr = Array1::<i64>::read_npy(std::fs::File::open(&p).unwrap()).unwrap();
                got.extend(arr.iter().copied());
                ci += 1;
            }
            assert!(!expected.is_empty(), "no codes recomputed");
            assert_eq!(
                got.len(),
                expected.len(),
                "on-disk vs recomputed token count"
            );
            assert_eq!(
                got, expected,
                "on-disk codes must equal direct shortlist_argmax output — proves \
                 build_cinder used the shortlist-union assigner, not the exhaustive default"
            );
        }

        // (v) Task 8 / gate C1 ablation arm 2: SEMANTEX_CINDER_EXACT_ASSIGN=1
        // must force the writer's EXHAUSTIVE default assignment — i.e. NO
        // id-aware shortlist assigner is installed. Rebuild the identical corpus
        // with the diagnostic set and assert the on-disk codes equal the
        // exhaustive full-scan argmax over ALL centroids (same dot-product metric
        // + lowest-id tie-break that `shortlist_argmax` approximates). Had the
        // shortlist assigner still been installed the codes would instead match
        // the (iv) shortlist codes; the two references coincide only where the
        // shortlist already covers the global argmax (the C4 agreement fraction),
        // so an exact match to the exhaustive reference pins the diagnostic path.
        {
            use crate::embedding::centroid_train::load_centroids_npy;
            use ndarray::Array1;
            use ndarray_npy::ReadNpyExt;

            // Local copy of shortlists::exhaustive_argmax (private there): dot
            // product, strict `>` with ascending iteration = lowest-id tie-break,
            // matching the compiled writer's default per-token assignment.
            fn exhaustive_argmax(e: &[f32], centroids: &ndarray::ArrayView2<f32>) -> i64 {
                let mut best_id = 0i64;
                let mut best_dot = f32::NEG_INFINITY;
                for (id, row) in centroids.rows().into_iter().enumerate() {
                    let dot: f32 = e.iter().zip(row.iter()).map(|(&a, &b)| a * b).sum();
                    if dot > best_dot {
                        best_dot = dot;
                        best_id = id as i64;
                    }
                }
                best_id
            }

            let idx_dir2 = tempfile::TempDir::new().unwrap();
            let fetch2 = |batch_ids: &[u64]| -> Result<Vec<(u64, String)>> {
                Ok(batch_ids
                    .iter()
                    .map(|&id| (id, chunks[id as usize].1.clone()))
                    .collect())
            };
            with_cinder_env(Some("1"), || {
                // SAFETY: CINDER_ENV_LOCK is held by with_cinder_env for the
                // duration of this closure, serializing all env mutation.
                unsafe { std::env::set_var("SEMANTEX_CINDER_EXACT_ASSIGN", "1") };
                let mut builder = ColbertPlaidIndexBuilder::new(idx_dir2.path(), 4)
                    .with_models_dir(overlay.path().to_path_buf());
                let build_res = builder.build_streaming_ids(&chunk_ids, fetch2);
                // SAFETY: same held lock; remove BEFORE asserting so a build
                // failure can't leak the diagnostic flag into later tests.
                unsafe { std::env::remove_var("SEMANTEX_CINDER_EXACT_ASSIGN") };
                build_res.expect("cinder build with EXACT_ASSIGN must succeed");
            });

            let cinder = CinderEncoder::new(&overlay_model_dir).unwrap();
            let centroids =
                load_centroids_npy(&model_manager::frozen_centroids_path(&overlay_model_dir))
                    .unwrap();
            let cview = centroids.view();

            let all_texts: Vec<String> = chunk_ids
                .iter()
                .map(|&id| chunks[id as usize].1.clone())
                .collect();
            let encoded = cinder.encode_documents_with_window_ids(&all_texts).unwrap();

            let mut expected_exhaustive: Vec<i64> = Vec::new();
            for (emb, _windows) in &encoded {
                for row in emb.rows() {
                    let e = row.as_slice().unwrap();
                    expected_exhaustive.push(exhaustive_argmax(e, &cview));
                }
            }

            let mut got: Vec<i64> = Vec::new();
            let mut ci = 0usize;
            loop {
                let p = idx_dir2.path().join(format!("{ci}.codes.npy"));
                if !p.exists() {
                    break;
                }
                let arr = Array1::<i64>::read_npy(std::fs::File::open(&p).unwrap()).unwrap();
                got.extend(arr.iter().copied());
                ci += 1;
            }
            assert!(!expected_exhaustive.is_empty(), "no codes recomputed");
            assert_eq!(
                got.len(),
                expected_exhaustive.len(),
                "on-disk vs recomputed token count (EXACT_ASSIGN)"
            );
            assert_eq!(
                got, expected_exhaustive,
                "with SEMANTEX_CINDER_EXACT_ASSIGN=1 the on-disk codes must equal the \
                 EXHAUSTIVE full-scan argmax — proves the shortlist-union assigner was NOT \
                 installed and the writer's default exhaustive scan ran"
            );
        }
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

    /// End-to-end Ember Plan B check: train tiny frozen centroids from a real
    /// `ColbertEmbedder`, then build the SAME corpus twice — once with
    /// `SEMANTEX_FROZEN_CENTROIDS=1` (external centroids) and once without
    /// (per-repo k-means) — and verify:
    ///   1. the flagged build succeeds and its `centroids.npy` row count
    ///      equals the trained k (proving the external artifact, not a
    ///      per-repo k-means result, was actually used);
    ///   2. the unflagged build also succeeds (the fallback path is
    ///      unaffected by the artifact merely existing on disk); and
    ///   3. both indexes agree on the top-1 chunk for a distinctive query
    ///      (frozen centroids don't change *what* is retrieved, only how the
    ///      codec is built).
    ///
    /// Needs the real downloaded LateOn-Code-edge model; `#[ignore]`'d like
    /// this file's other model-gated tests. The real model dir is never
    /// mutated: a temp overlay dir symlinks the model's onnx/tokenizer files
    /// in (so `ensure_colbert_model` sees `model_int8.onnx` and skips
    /// downloading) and only the overlay gets the trained
    /// `frozen_centroids.npy` written into it.
    #[test]
    #[ignore = "requires the downloaded LateOn-Code-edge model and builds a real PLAID index"]
    fn frozen_centroids_build_produces_searchable_index_and_fallback_matches() {
        use crate::config::SemantexConfig;
        use crate::embedding::centroid_train::{
            CentroidTrainOptions, save_centroids_npy, train_centroids,
        };

        let real_models_dir = SemantexConfig::default().models_dir();
        let real_model_dir = model_manager::ensure_colbert_model(&real_models_dir)
            .expect("LateOn-Code-edge must be downloaded for this ignored test");

        // Build an overlay model dir that symlinks in the real model's
        // files, so ensure_colbert_model(&overlay) is satisfied without
        // touching (or downloading into) the real model dir.
        let overlay = tempfile::TempDir::new().unwrap();
        let overlay_model_dir = overlay.path().join("LateOn-Code-edge");
        std::fs::create_dir_all(&overlay_model_dir).unwrap();
        for file in ["model_int8.onnx", "tokenizer.json", "onnx_config.json"] {
            let src = real_model_dir.join(file);
            let dst = overlay_model_dir.join(file);
            #[cfg(unix)]
            std::os::unix::fs::symlink(&src, &dst).unwrap();
            #[cfg(not(unix))]
            std::fs::copy(&src, &dst).unwrap();
        }

        // 1. Train tiny centroids from a few dozen synthetic code strings via
        //    train_centroids + the REAL ColbertEmbedder; save to the overlay.
        let embedder = ColbertEmbedder::for_indexing(&overlay_model_dir).unwrap();
        let corpus: Vec<String> = (0..40)
            .map(|i| format!("fn synthetic_helper_{i}(x: i32) -> i32 {{ x + {i} }}"))
            .collect();
        let k = 8usize;
        let opts = CentroidTrainOptions {
            num_centroids: k,
            sample_capacity: 10_000,
            batch: 8,
        };
        let centroids = train_centroids(&embedder, corpus.into_iter(), &opts).unwrap();
        assert_eq!(centroids.nrows(), k, "trained k must match request");
        let frozen_path = model_manager::frozen_centroids_path(&overlay_model_dir);
        save_centroids_npy(&frozen_path, &centroids).unwrap();

        // Distinctive chunk both indexes must retrieve identically.
        let chunks: Vec<(u64, String)> = (0u64..40)
            .map(|i| {
                (
                    i,
                    format!("fn synthetic_helper_{i}(x: i32) -> i32 {{ x + {i} }}"),
                )
            })
            .collect();
        let chunk_ids: Vec<u64> = chunks.iter().map(|(id, _)| *id).collect();
        let fetch = |batch_ids: &[u64]| -> Result<Vec<(u64, String)>> {
            Ok(batch_ids
                .iter()
                .map(|&id| (id, chunks[id as usize].1.clone()))
                .collect())
        };

        // 2. Build once with SEMANTEX_FROZEN_CENTROIDS=1 → assert build OK and
        //    the index's centroids.npy row count == trained k.
        let flagged_dir = tempfile::TempDir::new().unwrap();
        with_frozen_centroids_env(Some("1"), || {
            let mut builder = ColbertPlaidIndexBuilder::new(flagged_dir.path(), 4)
                .with_models_dir(overlay.path().to_path_buf());
            builder
                .build_streaming_ids(&chunk_ids, fetch)
                .expect("flagged build (external centroids) must succeed");
        });
        let flagged_centroids: ndarray::Array2<f32> = ndarray_npy::ReadNpyExt::read_npy(
            std::fs::File::open(flagged_dir.path().join("centroids.npy")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            flagged_centroids.nrows(),
            k,
            "flagged build's centroids.npy must carry the trained k rows, \
             proving the external artifact (not per-repo k-means) was used"
        );

        // 3. Build again WITHOUT the flag → assert build OK (per-repo path)
        //    and a search for a distinctive token returns the same top-1
        //    chunk in both indexes.
        let unflagged_dir = tempfile::TempDir::new().unwrap();
        with_frozen_centroids_env(None, || {
            let mut builder = ColbertPlaidIndexBuilder::new(unflagged_dir.path(), 4)
                .with_models_dir(overlay.path().to_path_buf());
            builder
                .build_streaming_ids(&chunk_ids, fetch)
                .expect("unflagged build (per-repo k-means) must succeed");
        });

        let flagged_mapping_path = flagged_dir.path().join("plaid_mapping.bin");
        let unflagged_mapping_path = unflagged_dir.path().join("plaid_mapping.bin");
        let flagged_backend = ColbertPlaidBackend::open(
            flagged_dir.path(),
            &flagged_mapping_path,
            &overlay_model_dir,
        )
        .unwrap();
        let unflagged_backend = ColbertPlaidBackend::open(
            unflagged_dir.path(),
            &unflagged_mapping_path,
            &overlay_model_dir,
        )
        .unwrap();

        let query = "synthetic_helper_7";
        let flagged_hits = flagged_backend.search(query, 1).unwrap();
        let unflagged_hits = unflagged_backend.search(query, 1).unwrap();
        assert!(!flagged_hits.is_empty() && !unflagged_hits.is_empty());
        assert_eq!(
            flagged_hits[0].chunk_id, unflagged_hits[0].chunk_id,
            "frozen centroids must not change what is retrieved, only how the codec is built"
        );
    }
}
