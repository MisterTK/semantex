//! `coderank-hnsw` dense backend: CodeRankEmbed single-vector embeddings stored
//! int8-quantized on disk, searched via HNSW ANN (instant-distance) over
//! dequantized f32 → fp32 rescore of the top `rescore_k`, with a brute-force
//! fallback below `HNSW_MIN_VECTORS`. Implements the S1 `DenseBackend` +
//! `DenseIndexBuilder` traits.
//!
//! Persistence is our OWN int8 vector store (postcard) + chunk_id map; the HNSW
//! graph is REBUILT in RAM from the store on open (instant-distance is a
//! build-once index with no native insert/delete/persist — see S2 Spike 2 in
//! `docs/superpowers/plans/2026-05-31-research-notes.md`). Delete compacts the
//! store and re-persists; the graph rebuilds on the next open.
//!
//! Per S1 G5: `positional_chunk_ids()` returns `None` (no positional doc array),
//! so the `hybrid.rs` file_filter subset path degrades to unfiltered dense +
//! result-merge filtering — documented on the trait impl below.

use crate::embedding::single_vector::{
    EMBEDDING_DIM, SingleVectorEmbedder, dequantize_int8, l2_normalize, quantize_int8,
};
use crate::search::dense_backend::{DenseBackend, DenseHit, DenseIndexBuilder};
use crate::search::simd::{dot_f32, dot_i8};
use crate::types::ScoredChunkId;
use anyhow::Result;
use instant_distance::{Builder, HnswMap, Point, Search};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Below this many vectors, skip the HNSW graph and brute-force (exact, and
/// avoids graph overhead on small repos). Overridable via `SEMANTEX_HNSW_MIN_VECTORS`.
const HNSW_MIN_VECTORS_DEFAULT: usize = 2_000;

/// On-disk file name for the int8 vector store + chunk_id map.
const STORE_FILE: &str = "vectors.bin";

/// Streaming-build batch size: at most this many chunks' content is fetched +
/// encoded at once, so only one batch's text is ever resident (the D6 build-RSS
/// bound). Well under SQLite's variable limit.
const ENCODE_BATCH: usize = 32;

/// HNSW + rescore tuning. Presets mirror the oxirs config-preset idea.
///
/// NB on `m`: the chosen ANN crate (`instant-distance`) hardcodes its max-degree
/// (`const M = 32`) and does NOT expose it, so `HnswParams.m` is INFORMATIONAL
/// for this backend (retained for the preset taxonomy + a future crate swap).
/// `ef_construction` and `ef_search` ARE honored (passed to `Builder`).
#[derive(Debug, Clone, Copy)]
pub struct HnswParams {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    /// fp32-rescore the top `rescore_k` ANN candidates. `0` ⇒ derive 4×k at
    /// query time. Effective = rescore_k.max(k).
    pub rescore_k: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 64,
            rescore_k: 0,
        }
    }
}

impl HnswParams {
    /// Resolve a named preset. `rescore_k = 0` means "derive 4×k at query time".
    pub fn preset(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "high_recall" => Self {
                m: 32,
                ef_construction: 400,
                ef_search: 200,
                rescore_k: 0,
            },
            "low_latency" => Self {
                m: 16,
                ef_construction: 200,
                ef_search: 32,
                rescore_k: 0,
            },
            "memory_optimized" => Self {
                m: 8,
                ef_construction: 100,
                ef_search: 48,
                rescore_k: 0,
            },
            _ => Self::default(), // "default" and unknown
        }
    }

    /// Resolve the effective params from a preset name + optional config
    /// overrides. `ef_search_override = 0` means "keep the preset's ef_search"
    /// (so `SEMANTEX_HNSW_PRESET=high_recall` is honored without also setting
    /// `SEMANTEX_HNSW_EF_SEARCH`); a non-zero override wins. `rescore_k_override`
    /// is applied as-is (0 = derive 4×k at query time).
    pub fn resolve(preset: &str, ef_search_override: usize, rescore_k_override: usize) -> Self {
        let mut p = Self::preset(preset);
        if ef_search_override != 0 {
            p.ef_search = ef_search_override;
        }
        p.rescore_k = rescore_k_override;
        p
    }
}

fn hnsw_min_vectors() -> usize {
    crate::config::env_usize("SEMANTEX_HNSW_MIN_VECTORS", HNSW_MIN_VECTORS_DEFAULT)
}

/// A graph point: an L2-normalized f32 vector. `distance` is `1 - dot` (==
/// `1 - cosine` for unit vectors), so smaller = closer — the orientation
/// instant-distance expects. RECORDED in S2 Spike 2.
#[derive(Debug, Clone)]
struct HnswPoint(Vec<f32>);

impl Point for HnswPoint {
    fn distance(&self, other: &Self) -> f32 {
        1.0 - dot_f32(&self.0, &other.0)
    }
}

/// The in-RAM ANN graph: instant-distance map from points → store row indices.
type HnswGraph = HnswMap<HnswPoint, usize>;

/// On-disk int8 vector store. Serialized with postcard alongside the chunk_id
/// map. `vectors[i]` is the int8 quantization of chunk `chunk_ids[i]` with
/// per-vector `scales[i]`. Positional: store row `i` ↔ `chunk_ids[i]` ↔ graph
/// value `i`.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Int8VectorStore {
    pub dim: usize,
    pub scales: Vec<f32>,
    pub vectors: Vec<Vec<i8>>,
    pub chunk_ids: Vec<u64>,
}

impl Int8VectorStore {
    fn len(&self) -> usize {
        self.chunk_ids.len()
    }

    /// Append one quantized vector for `chunk_id`. Returns its row index.
    fn push(&mut self, chunk_id: u64, q: Vec<i8>, scale: f32) -> usize {
        let idx = self.chunk_ids.len();
        self.vectors.push(q);
        self.scales.push(scale);
        self.chunk_ids.push(chunk_id);
        idx
    }

    /// Dequantize store row `i` back to f32. (Stored rows were L2-normalized
    /// before quantization; we re-normalize after dequant so the graph's
    /// `1 - dot` distance is a true `1 - cosine`.)
    fn dequant_row(&self, i: usize) -> Vec<f32> {
        let mut f = dequantize_int8(&self.vectors[i], self.scales[i]);
        l2_normalize(&mut f);
        f
    }

    /// Brute-force top-k over int8 (used below `HNSW_MIN_VECTORS`). Shortlists
    /// the top `rescore_k` by int8 dot, then fp32-rescores down to `k`.
    ///
    /// FIX 4: the int8 shortlist width is `rescore_k.max(k)`, NOT just `k`. int8
    /// dot is only APPROXIMATELY monotonic in cosine across per-vector scales, so
    /// a true top-k item int8-misranked just past `k` would be unrecoverable if
    /// we shortlisted only `k`. Widening to `rescore_k` mirrors the HNSW path's
    /// candidate window and makes the two paths' recall behavior consistent.
    fn brute_force(&self, query: &[f32], k: usize, rescore_k: usize) -> Vec<DenseHit> {
        if self.vectors.is_empty() {
            return Vec::new();
        }
        let shortlist = rescore_k.max(k).max(1);
        let (q8, _qs) = quantize_int8(query);
        let mut scored: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, dot_i8(&q8, v)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(shortlist);
        let positions: Vec<usize> = scored.into_iter().map(|(i, _)| i).collect();
        self.fp32_rescore(query, &positions, k)
    }

    /// fp32-rescore the given row `positions` (dequantize → cosine via dot on
    /// L2-normalized vectors) and return the top `k` as `DenseHit`s sorted desc.
    fn fp32_rescore(&self, query: &[f32], positions: &[usize], k: usize) -> Vec<DenseHit> {
        let mut hits: Vec<DenseHit> = positions
            .iter()
            .filter_map(|&i| {
                let scale = *self.scales.get(i)?;
                let mut f = dequantize_int8(self.vectors.get(i)?, scale);
                // Re-normalize: int8 round-trip leaves the dequantized vector at
                // norm ≈0.99, so re-unit it (FIX 5) → dot == cosine exactly. The
                // query is already unit from the encoder.
                l2_normalize(&mut f);
                let score = dot_f32(query, &f);
                Some(ScoredChunkId::new(self.chunk_ids[i], score))
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k.max(1));
        hits
    }

    /// Dequantize the stored int8 vectors for the given `chunk_ids` back to f32,
    /// preserving the requested order and SKIPPING any id not present (no faked
    /// rows). Powers `DenseBackend::embed_doc_vectors` (S7 MMR + semantic cache).
    ///
    /// FIX 5: the returned vectors are RE-NORMALIZED to unit length (via
    /// `dequant_row`), so they match `embed_text_vector`'s exactly-unit query
    /// vector — S7's cosine(query, doc) is symmetric (both sides unit, dot ==
    /// cosine), not skewed by the int8 round-trip's ≈0.99 norm.
    fn vectors_for(&self, chunk_ids: &[u64]) -> Vec<(u64, Vec<f32>)> {
        use std::collections::HashMap;
        let pos: HashMap<u64, usize> = self
            .chunk_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();
        chunk_ids
            .iter()
            .filter_map(|id| {
                let &i = pos.get(id)?;
                Some((*id, self.dequant_row(i)))
            })
            .collect()
    }
}

/// The persisted dense index: the int8 vector store (+ rebuilt HNSW graph in RAM).
pub struct HnswDenseIndex {
    store: Int8VectorStore,
    params: HnswParams,
    /// In-RAM HNSW graph, built from `store` on open when len ≥ HNSW_MIN_VECTORS.
    /// `None` ⇒ brute-force path.
    graph: Option<HnswGraph>,
    /// The ef_search actually BAKED into `graph` at build time. instant-distance
    /// bakes ef_search into the index, so its search iterator yields at most this
    /// many candidates — the rescore window cannot exceed it (FIX 3). We raise
    /// the baked value to cover an explicit `rescore_k`, and clamp+warn at query
    /// time when a derived `4×k` window would overrun it.
    baked_ef_search: usize,
}

impl HnswDenseIndex {
    /// The ef_search to bake into the graph: the configured `ef_search`, raised
    /// to cover an explicit `rescore_k` so the candidate pool actually spans the
    /// rescore window. (`rescore_k = 0` ⇒ derived 4×k at query time, which is
    /// clamped to the baked value in `search`, since k is not known here.)
    fn baked_ef_search(params: HnswParams) -> usize {
        params.ef_search.max(params.rescore_k).max(1)
    }

    /// Build the in-RAM graph from the store (dequantized + renormalized f32
    /// vectors), or return `None` for a small store (brute-force path). The
    /// graph value at row `i` is the store row index `i`.
    fn build_graph(store: &Int8VectorStore, params: HnswParams) -> Option<HnswGraph> {
        if store.len() < hnsw_min_vectors() {
            return None;
        }
        let points: Vec<HnswPoint> = (0..store.len())
            .map(|i| HnswPoint(store.dequant_row(i)))
            .collect();
        let values: Vec<usize> = (0..store.len()).collect();
        let graph = Builder::default()
            .ef_construction(params.ef_construction.max(1))
            .ef_search(Self::baked_ef_search(params))
            .build(points, values);
        Some(graph)
    }

    fn load(dir: &Path, params: HnswParams) -> Result<Self> {
        let bytes = std::fs::read(dir.join(STORE_FILE))?;
        let store: Int8VectorStore = postcard::from_bytes(&bytes)?;
        let graph = Self::build_graph(&store, params);
        Ok(Self {
            store,
            params,
            graph,
            baked_ef_search: Self::baked_ef_search(params),
        })
    }

    fn save(store: &Int8VectorStore, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join(STORE_FILE), postcard::to_stdvec(store)?)?;
        Ok(())
    }

    /// Top-k search: ANN (if graph present) → fp32 rescore; else brute-force.
    fn search(&self, query: &[f32], k: usize) -> Vec<DenseHit> {
        if self.store.len() == 0 {
            return Vec::new();
        }
        let rescore_k = if self.params.rescore_k == 0 {
            4 * k
        } else {
            self.params.rescore_k
        }
        .max(k);
        match &self.graph {
            None => self.store.brute_force(query, k, rescore_k),
            Some(graph) => {
                // The graph's search yields at most `baked_ef_search` candidates,
                // so the rescore window can't exceed it. Clamp (and warn once when
                // a derived 4×k window would overrun the baked ef_search — raise
                // SEMANTEX_HNSW_EF_SEARCH / use a higher preset for larger k).
                let effective = rescore_k.min(self.baked_ef_search);
                if rescore_k > self.baked_ef_search {
                    tracing::debug!(
                        rescore_k,
                        baked_ef_search = self.baked_ef_search,
                        "coderank-hnsw rescore window capped by baked ef_search; \
                         raise SEMANTEX_HNSW_EF_SEARCH or use a higher-recall preset"
                    );
                }
                let mut search = Search::default();
                let qpoint = HnswPoint(query.to_vec());
                let positions: Vec<usize> = graph
                    .search(&qpoint, &mut search)
                    .take(effective)
                    .map(|item| *item.value)
                    .collect();
                self.store.fp32_rescore(query, &positions, k)
            }
        }
    }
}

/// Query-time backend.
pub struct CoderankHnswBackend {
    index: HnswDenseIndex,
    embedder: &'static SingleVectorEmbedder,
}

impl CoderankHnswBackend {
    pub const NAME: &'static str = "coderank-hnsw";

    pub fn open(dir: &Path, model_dir: &Path, params: HnswParams) -> Result<Self> {
        let index = HnswDenseIndex::load(dir, params)?;
        let embedder = SingleVectorEmbedder::global(model_dir)?;
        Ok(Self { index, embedder })
    }

    #[cfg(test)]
    fn open_for_test(dir: &Path) -> Self {
        // Test-only: load the store without a real model. The only test calling
        // this asserts `positional_chunk_ids()`, which never touches the
        // embedder — so a lazy global against a temp dir is fine (no session
        // built; the dir merely needs to exist).
        let index = HnswDenseIndex::load(dir, HnswParams::default()).expect("test store loads");
        let tmp = std::env::temp_dir();
        let embedder =
            SingleVectorEmbedder::global(&tmp).expect("global embedder (lazy; no session built)");
        Self { index, embedder }
    }
}

impl DenseBackend for CoderankHnswBackend {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn search(&self, query: &str, k: usize) -> Result<Vec<DenseHit>> {
        let q = self.embedder.encode_query(query)?;
        Ok(self.index.search(&q, k))
    }

    fn search_with_subset(&self, query: &str, k: usize, subset: &[u64]) -> Result<Vec<DenseHit>> {
        // HNSW has no positional doc array (positional_chunk_ids → None), so the
        // hybrid.rs subset path normally won't call this with a real subset. If a
        // caller does pass one, honor it by post-filtering the unfiltered top-N
        // (N widened to absorb the filter). This is the documented G5 degradation.
        if subset.is_empty() {
            return Ok(Vec::new());
        }
        let q = self.embedder.encode_query(query)?;
        let wide = self.index.search(&q, (k * 8).max(64));
        let allow: std::collections::HashSet<u64> = subset.iter().copied().collect();
        let mut filtered: Vec<DenseHit> = wide
            .into_iter()
            .filter(|h| allow.contains(&h.chunk_id))
            .collect();
        filtered.truncate(k);
        Ok(filtered)
    }

    /// S1 G5: no positional doc→chunk array. Returning `None` makes the
    /// `hybrid.rs` file_filter subset path degrade to unfiltered dense + the
    /// result-merge file_filter — which is correct, just not pre-filtered.
    fn positional_chunk_ids(&self) -> Option<&[u64]> {
        None
    }

    /// EXACT query vector for S7's MMR + semantic cache (integration §3 item 2 +
    /// §4 D-mmr-cache). Encodes the query via the single-vector encoder
    /// (CodeRankEmbed), already L2-normalized by `encode_query`. `Some` always on
    /// this backend; `None` only if the encoder errors (e.g. model unavailable).
    fn embed_text_vector(&self, query: &str) -> Option<Vec<f32>> {
        self.embedder.encode_query(query).ok()
    }

    /// EXACT stored doc vectors for `chunk_ids`: dequantize the int8 vectors held
    /// in the `Int8VectorStore` back to f32 (integration §3 item 2 + §4
    /// D-mmr-cache). Ids not present are skipped. `Some` always on this backend.
    fn embed_doc_vectors(&self, chunk_ids: &[u64]) -> Option<Vec<(u64, Vec<f32>)>> {
        Some(self.index.store.vectors_for(chunk_ids))
    }
}

/// Build-time backend. Streams encode→insert: each batch is encoded, quantized,
/// pushed into the store, and the RSS failsafe is checked — never collect all
/// fp32 embeddings then one big call (the D6 build-memory property).
pub struct CoderankHnswIndexBuilder {
    dir: PathBuf,
    /// Tuning params. Build-time only uses `ef_construction`/`m` indirectly via
    /// the open-time graph rebuild (instant-distance is build-once + we rebuild
    /// on open), so the builder keeps them for forward-compat / a future crate
    /// that builds the graph at index time. Retained on the struct so callers
    /// can pass the resolved preset through one constructor.
    #[allow(dead_code)]
    params: HnswParams,
    models_dir: PathBuf,
    /// When set, encode `format!("{annotation}\n{code}")` instead of raw code
    /// (the `SEMANTEX_DENSE_CONTEXT` A/B; default off). The annotation is passed
    /// in via [`with_context_annotations`] keyed by chunk id.
    dense_context: bool,
    context_by_id: std::collections::HashMap<u64, String>,
    store: Int8VectorStore,
}

impl CoderankHnswIndexBuilder {
    pub fn new(dir: &Path, params: HnswParams) -> Self {
        Self {
            dir: dir.to_path_buf(),
            params,
            models_dir: crate::config::SemantexConfig::default().models_dir(),
            dense_context: false,
            context_by_id: std::collections::HashMap::new(),
            store: Int8VectorStore {
                dim: EMBEDDING_DIM,
                ..Default::default()
            },
        }
    }

    pub fn with_models_dir(mut self, models_dir: PathBuf) -> Self {
        self.models_dir = models_dir;
        self
    }

    /// Enable the `SEMANTEX_DENSE_CONTEXT` A/B: per-chunk graph-derived NL
    /// annotations to prepend to the embedded text. Default off (raw code).
    pub fn with_context_annotations(
        mut self,
        enabled: bool,
        annotations: std::collections::HashMap<u64, String>,
    ) -> Self {
        self.dense_context = enabled;
        self.context_by_id = annotations;
        self
    }

    /// Encode + quantize one already-fetched batch into the store. The caller is
    /// responsible for the RSS check + dropping the batch's content before the
    /// next fetch (see [`Self::encode_streaming_ids`]). Kept for the in-memory
    /// `build`/`insert` trait entry points (small corpora / tests).
    fn encode_into_store(
        &mut self,
        embedder: &SingleVectorEmbedder,
        chunks: &[(u64, &str)],
    ) -> Result<()> {
        for batch in chunks.chunks(ENCODE_BATCH) {
            if let Err(e) = crate::memory::check_rss_or_abort("HNSW encode batch") {
                anyhow::bail!("Indexing aborted: {e}");
            }
            for &(chunk_id, content) in batch {
                let emb = if self.dense_context {
                    let ctx = self.context_by_id.get(&chunk_id).map_or("", String::as_str);
                    embedder.encode_document_with_context(ctx, content)?
                } else {
                    embedder.encode_document(content)?
                };
                let (q, scale) = quantize_int8(&emb);
                self.store.push(chunk_id, q, scale);
            }
        }
        Ok(())
    }

    /// Stream content batch-by-batch and encode→quantize→push into the store,
    /// checking the RSS failsafe at each batch boundary. `fetch` pulls only one
    /// `ENCODE_BATCH`-sized slice of `(id, content)` from the store at a time, so
    /// only ONE batch's chunk text is ever resident — the D6 build-memory bound.
    /// The whole corpus is NEVER materialized.
    fn encode_streaming_ids<F>(
        &mut self,
        embedder: &SingleVectorEmbedder,
        chunk_ids: &[u64],
        mut fetch: F,
    ) -> Result<()>
    where
        F: FnMut(&[u64]) -> Result<Vec<(u64, String)>>,
    {
        for batch_ids in chunk_ids.chunks(ENCODE_BATCH) {
            if let Err(e) = crate::memory::check_rss_or_abort("HNSW encode batch") {
                anyhow::bail!("Indexing aborted: {e}");
            }
            // Fetch only this batch's content (≤ENCODE_BATCH ids).
            let batch = fetch(batch_ids)?;
            for (chunk_id, content) in &batch {
                let emb = if self.dense_context {
                    let ctx = self.context_by_id.get(chunk_id).map_or("", String::as_str);
                    embedder.encode_document_with_context(ctx, content)?
                } else {
                    embedder.encode_document(content)?
                };
                let (q, scale) = quantize_int8(&emb);
                self.store.push(*chunk_id, q, scale);
            }
            // `batch` (and its content Strings) drop here → only one batch's
            // content is ever live at a time.
        }
        Ok(())
    }

    /// Fresh full build over `chunk_ids`, streaming content via `fetch` (one
    /// batch resident at a time). This is the RSS-bounded entry the builder uses
    /// on a fresh dense build — it never holds the whole corpus in RAM.
    pub fn build_streaming_ids<F>(&mut self, chunk_ids: &[u64], mut fetch: F) -> Result<()>
    where
        F: FnMut(&[u64]) -> Result<Vec<(u64, String)>>,
    {
        if chunk_ids.is_empty() {
            tracing::info!("No chunks to encode for coderank-hnsw index");
            return Ok(());
        }
        let embedder = self.make_embedder()?;
        // Fresh full build.
        self.store = Int8VectorStore {
            dim: EMBEDDING_DIM,
            ..Default::default()
        };
        self.encode_streaming_ids(&embedder, chunk_ids, &mut fetch)?;
        crate::memory::purge_allocator();
        let dir = self.dir.clone();
        HnswDenseIndex::save(&self.store, &dir)?;
        tracing::info!("coderank-hnsw index built ({} vectors)", self.store.len());
        Ok(())
    }

    /// Incremental add of `chunk_ids`, streaming content via `fetch` (one batch
    /// resident at a time). RSS-bounded counterpart to `insert`.
    pub fn insert_streaming_ids<F>(&mut self, chunk_ids: &[u64], mut fetch: F) -> Result<()>
    where
        F: FnMut(&[u64]) -> Result<Vec<(u64, String)>>,
    {
        if chunk_ids.is_empty() {
            return Ok(());
        }
        let embedder = self.make_embedder()?;
        self.store = self.load_existing_store();
        self.encode_streaming_ids(&embedder, chunk_ids, &mut fetch)?;
        let dir = self.dir.clone();
        HnswDenseIndex::save(&self.store, &dir)?;
        Ok(())
    }

    fn load_existing_store(&self) -> Int8VectorStore {
        std::fs::read(self.dir.join(STORE_FILE))
            .ok()
            .and_then(|b| postcard::from_bytes::<Int8VectorStore>(&b).ok())
            .unwrap_or_else(|| Int8VectorStore {
                dim: EMBEDDING_DIM,
                ..Default::default()
            })
    }

    fn make_embedder(&self) -> Result<SingleVectorEmbedder> {
        let model_dir =
            crate::embedding::single_vector_model::ensure_coderank_model(&self.models_dir)?;
        SingleVectorEmbedder::for_indexing(&model_dir)
    }
}

impl DenseIndexBuilder for CoderankHnswIndexBuilder {
    fn name(&self) -> &'static str {
        CoderankHnswBackend::NAME
    }

    fn build(&mut self, chunks: &[(u64, &str)]) -> Result<()> {
        if chunks.is_empty() {
            tracing::info!("No chunks to encode for coderank-hnsw index");
            return Ok(());
        }
        let embedder = self.make_embedder()?;
        // Fresh full build.
        self.store = Int8VectorStore {
            dim: EMBEDDING_DIM,
            ..Default::default()
        };
        self.encode_into_store(&embedder, chunks)?;
        crate::memory::purge_allocator();
        let dir = self.dir.clone();
        HnswDenseIndex::save(&self.store, &dir)?;
        tracing::info!("coderank-hnsw index built ({} vectors)", self.store.len());
        Ok(())
    }

    fn insert(&mut self, chunks: &[(u64, &str)]) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }
        let embedder = self.make_embedder()?;
        self.store = self.load_existing_store();
        self.encode_into_store(&embedder, chunks)?;
        let dir = self.dir.clone();
        HnswDenseIndex::save(&self.store, &dir)?;
        Ok(())
    }

    fn delete(&mut self, chunk_ids: &[u64]) -> Result<()> {
        if chunk_ids.is_empty() {
            return Ok(());
        }
        self.store = self.load_existing_store();
        let remove: std::collections::HashSet<u64> = chunk_ids.iter().copied().collect();
        // Rebuild the store without the removed ids (compact; no tombstones —
        // the HNSW graph is rebuilt on open from the compacted store).
        let mut compact = Int8VectorStore {
            dim: self.store.dim,
            ..Default::default()
        };
        for i in 0..self.store.len() {
            if remove.contains(&self.store.chunk_ids[i]) {
                continue;
            }
            compact.push(
                self.store.chunk_ids[i],
                std::mem::take(&mut self.store.vectors[i]),
                self.store.scales[i],
            );
        }
        self.store = compact;
        let dir = self.dir.clone();
        HnswDenseIndex::save(&self.store, &dir)?;
        Ok(())
    }

    fn persist(&self, dir: &Path) -> Result<()> {
        HnswDenseIndex::save(&self.store, dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_names_resolve() {
        assert_eq!(HnswParams::preset("default").m, 16);
        assert_eq!(HnswParams::preset("high_recall").ef_search, 200);
        assert_eq!(HnswParams::preset("low_latency").ef_search, 32);
        assert_eq!(HnswParams::preset("memory_optimized").m, 8);
        assert_eq!(HnswParams::preset("garbage").m, 16); // unknown → default
    }

    #[test]
    fn resolve_keeps_preset_ef_search_when_override_is_zero() {
        // FIX 2: preset=high_recall with no explicit ef_search override (0) must
        // KEEP the preset's ef_search (200), not silently fall back to 64.
        let p = HnswParams::resolve("high_recall", 0, 0);
        assert_eq!(
            p.ef_search, 200,
            "preset ef_search must survive a 0 override"
        );
        assert_eq!(p.rescore_k, 0);
        // An explicit non-zero override wins.
        let p2 = HnswParams::resolve("high_recall", 96, 0);
        assert_eq!(p2.ef_search, 96, "explicit ef_search override must win");
        // default preset with no override keeps 64.
        assert_eq!(HnswParams::resolve("default", 0, 0).ef_search, 64);
        // rescore_k override is applied as-is.
        assert_eq!(HnswParams::resolve("default", 0, 256).rescore_k, 256);
    }

    #[test]
    fn baked_ef_search_covers_explicit_rescore_k() {
        // FIX 3: an explicit rescore_k larger than ef_search must widen the
        // graph's baked ef_search so the candidate pool spans the rescore window.
        let params = HnswParams {
            ef_search: 64,
            rescore_k: 300,
            ..HnswParams::default()
        };
        assert_eq!(HnswDenseIndex::baked_ef_search(params), 300);
        // When rescore_k ≤ ef_search, ef_search stands.
        let params2 = HnswParams {
            ef_search: 200,
            rescore_k: 50,
            ..HnswParams::default()
        };
        assert_eq!(HnswDenseIndex::baked_ef_search(params2), 200);
    }

    #[test]
    fn brute_force_ranks_by_cosine_descending() {
        let store = Int8VectorStore {
            dim: 2,
            scales: vec![1.0, 1.0, 1.0],
            vectors: vec![
                quantize_int8(&[1.0, 0.0]).0, // id 10
                // unit vector at 45°: cos45° = 1/√2
                quantize_int8(&[
                    std::f32::consts::FRAC_1_SQRT_2,
                    std::f32::consts::FRAC_1_SQRT_2,
                ])
                .0, // id 20
                quantize_int8(&[0.0, 1.0]).0, // id 30
            ],
            chunk_ids: vec![10, 20, 30],
        };
        let q = vec![1.0f32, 0.0];
        let hits = store.brute_force(&q, 3, 0);
        assert_eq!(
            hits.iter().map(|h| h.chunk_id).collect::<Vec<_>>(),
            vec![10, 20, 30]
        );
        assert!(hits[0].score >= hits[1].score && hits[1].score >= hits[2].score);
    }

    #[test]
    fn empty_store_returns_no_hits() {
        let store = Int8VectorStore {
            dim: 4,
            scales: vec![],
            vectors: vec![],
            chunk_ids: vec![],
        };
        assert!(store.brute_force(&[0.0, 0.0, 0.0, 0.0], 5, 0).is_empty());
    }

    #[test]
    fn rescore_reorders_int8_candidates_by_fp32() {
        let store = Int8VectorStore {
            dim: 3,
            scales: vec![1.0, 1.0],
            vectors: vec![
                quantize_int8(&[0.9, 0.1, 0.0]).0,
                quantize_int8(&[0.8, 0.2, 0.0]).0,
            ],
            chunk_ids: vec![1, 2],
        };
        let q = vec![1.0f32, 0.0, 0.0];
        let hits = store.fp32_rescore(&q, &[0, 1], 2);
        assert_eq!(hits[0].chunk_id, 1);
    }

    #[test]
    fn brute_force_wide_shortlist_recovers_int8_misranked_topk() {
        // FIX 4: int8 dot is only APPROXIMATELY monotone in cosine across rows
        // with different per-vector scales, so it can INVERT the fp32 top-1.
        //
        // These vectors (verified to reproduce the inversion under the exact
        // quantize_int8/dot_i8/dot_f32 math) make id=1 the true fp32 top-1 but
        // the int8 LOSER vs id=2:
        //   v1=[0.952,0.218,0.505] — large axis-2 outlier ⇒ COARSE int8 scale
        //       (max|comp|/127) ⇒ its int8 dot is DEFLATED.
        //   v2=[0.605,0.394,0.096] — no outlier ⇒ fine scale ⇒ inflated int8 dot.
        // fp32 cosine(query): id1=0.8711 > id2=0.8455 (id1 wins, margin ~0.026).
        // int8 dot(query):    id1=16245 < id2=16461   (id2 wins by 216).
        // Stored rows are L2-normalized before quant (as the builder does).
        let mut v1 = [0.952f32, 0.218, 0.505];
        let mut v2 = [0.605f32, 0.394, 0.096];
        l2_normalize(&mut v1);
        l2_normalize(&mut v2);
        let (q1, s1) = quantize_int8(&v1);
        let (q2, s2) = quantize_int8(&v2);
        let store = Int8VectorStore {
            dim: 3,
            scales: vec![s1, s2],
            vectors: vec![q1, q2],
            chunk_ids: vec![1, 2],
        };
        let mut query = vec![1.0f32, 0.028, 0.0];
        l2_normalize(&mut query);

        // CONTRAST — the assertion that genuinely guards FIX 4:
        // (a) the PRE-FIX behavior (shortlist = k = 1) int8-shortlists only the
        //     wrong winner (id=2), so fp32 rescore can never recover id=1.
        let narrow = store.brute_force(&query, 1, 1);
        assert_eq!(
            narrow[0].chunk_id, 2,
            "with shortlist == k the int8 inversion is unrecoverable (returns the WRONG id) \
             — this is exactly the bug FIX 4 addresses"
        );
        // (b) the FIXED behavior (rescore_k ≥ 2) widens the int8 shortlist so the
        //     true fp32 top-1 (id=1) is included and fp32 rescore recovers it.
        let wide = store.brute_force(&query, 1, 4);
        assert_eq!(
            wide[0].chunk_id, 1,
            "wide int8 shortlist + fp32 rescore must recover the true fp32 top-1"
        );
    }

    #[test]
    fn builder_name_and_empty_build_is_ok() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut b = CoderankHnswIndexBuilder::new(tmp.path(), HnswParams::default());
        assert_eq!(DenseIndexBuilder::name(&b), "coderank-hnsw");
        b.build(&[]).unwrap();
        assert!(!tmp.path().join("vectors.bin").exists());
    }

    #[test]
    fn store_persist_and_reload_round_trips() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut store = Int8VectorStore {
            dim: 2,
            ..Default::default()
        };
        let (q, s) = quantize_int8(&[0.6, 0.8]);
        store.push(42, q, s);
        let path = tmp.path().join("vectors.bin");
        std::fs::write(&path, postcard::to_stdvec(&store).unwrap()).unwrap();
        let back: Int8VectorStore = postcard::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(back.chunk_ids, vec![42]);
        assert_eq!(back.dim, 2);
    }

    #[test]
    fn backend_positional_chunk_ids_is_none() {
        // S1 G5: HNSW has no positional doc array; subset path degrades to merge.
        let tmp = tempfile::TempDir::new().unwrap();
        let store = Int8VectorStore {
            dim: 4,
            ..Default::default()
        };
        std::fs::write(
            tmp.path().join("vectors.bin"),
            postcard::to_stdvec(&store).unwrap(),
        )
        .unwrap();
        let backend = CoderankHnswBackend::open_for_test(tmp.path());
        assert!(backend.positional_chunk_ids().is_none());
    }

    #[test]
    fn vectors_for_dequantizes_stored_ids_and_skips_missing() {
        let mut store = Int8VectorStore {
            dim: 3,
            ..Default::default()
        };
        let (q10, s10) = quantize_int8(&[0.6, 0.8, 0.0]);
        let (q20, s20) = quantize_int8(&[0.0, 0.6, 0.8]);
        store.push(10, q10, s10);
        store.push(20, q20, s20);

        // Requested order is honored; the absent id (99) is skipped, not faked.
        let got = store.vectors_for(&[20, 99, 10]);
        assert_eq!(
            got.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![20, 10]
        );
        for (id, v) in &got {
            assert_eq!(v.len(), store.dim);
            let row = store.chunk_ids.iter().position(|c| c == id).unwrap();
            // FIX 5: vectors_for re-normalizes the dequantized row to unit norm
            // (== dequant_row), so it must equal the re-normalized stored row and
            // be unit-length (matching embed_text_vector's unit query vector).
            let expected = store.dequant_row(row);
            assert_eq!(
                v, &expected,
                "vectors_for must return the re-normalized stored int8 row"
            );
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "embed_doc_vectors row must be unit-norm (FIX 5), got {norm}"
            );
        }
    }

    /// End-to-end on a tiny built index: BOTH accessor methods return `Some`
    /// with `EMBEDDING_DIM`-length vectors, and `embed_doc_vectors` round-trips
    /// the stored ids. `#[ignore]` — needs the CodeRankEmbed model download.
    #[test]
    #[ignore = "needs the CodeRankEmbed model download"]
    fn vector_accessor_methods_return_some_on_built_index() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path().join("dense");
        let models = tmp.path().join("models");
        let model_dir = crate::embedding::single_vector_model::ensure_coderank_model(&models)
            .expect("model provisions");

        let mut b = CoderankHnswIndexBuilder::new(&dir, HnswParams::default())
            .with_models_dir(models.clone());
        b.build(&[
            (7, "fn parse_int(s:&str)->i64 { s.parse().unwrap() }"),
            (9, "def add(a,b): return a+b"),
        ])
        .unwrap();

        let backend = CoderankHnswBackend::open(&dir, &model_dir, HnswParams::default()).unwrap();

        let qv = backend
            .embed_text_vector("parse an integer")
            .expect("Some on coderank-hnsw");
        assert_eq!(qv.len(), SingleVectorEmbedder::embedding_dim());
        let norm: f32 = qv.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "query vector must be L2-normalized, got {norm}"
        );

        let dv = backend
            .embed_doc_vectors(&[9, 7, 404])
            .expect("Some on coderank-hnsw");
        assert_eq!(dv.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![9, 7]);
        assert!(
            dv.iter()
                .all(|(_, v)| v.len() == SingleVectorEmbedder::embedding_dim())
        );
    }

    #[test]
    fn hnsw_graph_and_brute_force_agree_on_tiny_corpus() {
        // Force the HNSW graph path on a tiny corpus by lowering the threshold,
        // then assert HNSW top-k == brute-force top-k (exact agreement is
        // expected because we fp32-rescore the ANN candidates).
        // SAFETY: single-threaded test; restore the env var after.
        // (The min-vectors knob is read per-call, so set it just for this build.)
        unsafe {
            std::env::set_var("SEMANTEX_HNSW_MIN_VECTORS", "4");
        }
        let mut store = Int8VectorStore {
            dim: 4,
            ..Default::default()
        };
        // 12 deterministic L2-normalized vectors.
        for i in 0..12u64 {
            let mut v = vec![
                ((i * 7 % 11) as f32) - 5.0,
                ((i * 13 % 11) as f32) - 5.0,
                ((i * 17 % 11) as f32) - 5.0,
                ((i * 19 % 11) as f32) - 5.0,
            ];
            l2_normalize(&mut v);
            let (q, s) = quantize_int8(&v);
            store.push(100 + i, q, s);
        }
        let params = HnswParams::default();
        let graph = HnswDenseIndex::build_graph(&store, params);
        assert!(
            graph.is_some(),
            "graph must build above the lowered threshold"
        );
        let idx = HnswDenseIndex {
            store: store.clone(),
            params,
            graph,
            baked_ef_search: HnswDenseIndex::baked_ef_search(params),
        };
        let mut query = vec![1.0f32, 0.5, -0.3, 0.2];
        l2_normalize(&mut query);
        let hnsw_hits: Vec<u64> = idx.search(&query, 5).iter().map(|h| h.chunk_id).collect();
        // Mirror search()'s rescore window (rescore_k = 4×k for the default
        // params) so the two paths shortlist the same candidate set.
        let bf_hits: Vec<u64> = store
            .brute_force(&query, 5, 4 * 5)
            .iter()
            .map(|h| h.chunk_id)
            .collect();
        assert_eq!(
            hnsw_hits, bf_hits,
            "HNSW (with fp32 rescore) must agree with brute-force on a tiny corpus"
        );
        unsafe {
            std::env::remove_var("SEMANTEX_HNSW_MIN_VECTORS");
        }
    }
}
