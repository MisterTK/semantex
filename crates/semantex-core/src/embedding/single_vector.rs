//! Single-vector dense encoder for the `coderank-hnsw` backend.
//!
//! Wraps CodeRankEmbed (int8 ONNX) via a raw `ort::Session`. Lazy session init,
//! CPU execution provider pinned, threads from `SEMANTEX_(INDEX_)ORT_THREADS` —
//! the same rationale as `colbert.rs` (a single multi-thread CPU session; `Auto`
//! would collapse to 1 thread on non-macOS, see colbert.rs).
//!
//! Pipeline: tokenize → forward → **mean-pool over the attention mask** (RECORDED
//! in research-notes, S2 — CodeRankEmbed) → L2-normalize. Documents embed RAW
//! code (no prefix); queries get the RECORDED query prefix
//! `Represent this query for searching relevant code: `. The BM25 enrichment
//! (NL annotation + expansion) is NOT fed here by default — dense sees only the
//! raw chunk content (the `SEMANTEX_DENSE_CONTEXT` A/B switch can prepend the
//! graph-derived annotation; see [`encode_document_with_context`]). No
//! Matryoshka (fixed 768-dim).

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::search::simd::dot_f32;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use parking_lot::Mutex;
use tokenizers::Tokenizer;

// ── RECORDED from research-notes (S2 — CodeRankEmbed embedder) ──────────────
/// CodeRankEmbed embedding dimension.
pub(crate) const EMBEDDING_DIM: usize = 768;
/// Max context the encoder accepts; we truncate token ids to this.
const MAX_CTX: usize = 8192;
/// Query prefix (documents get none). EXACT, trailing space included.
pub(crate) const QUERY_PREFIX: &str = "Represent this query for searching relevant code: ";
/// ONNX input tensor names (int64).
const INPUT_IDS: &str = "input_ids";
const ATTENTION_MASK: &str = "attention_mask";
/// ONNX output tensor name holding `[batch, seq, dim]` token embeddings.
const OUTPUT_NAME: &str = "last_hidden_state";
// ─────────────────────────────────────────────────────────────────────────────

/// L2-normalize in place. A zero vector is left as zeros (no div-by-zero).
pub(crate) fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Symmetric per-vector int8 quantization. Returns `(quantized, scale)` where
/// `dequantized[i] ≈ quantized[i] as f32 * scale`. The scale is `max(|v|) / 127`,
/// floored to a tiny positive value so dequant never divides by — or multiplies
/// into — zero. Symmetric (zero-point 0) per integration §4 D-int8, so the
/// `dot_i8`/`cosine_i8` kernels apply directly. L2-normalized inputs keep
/// `max(|v|)` ≈ O(1), so int8 resolution is ample.
pub(crate) fn quantize_int8(v: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let scale = (max_abs / 127.0).max(1e-8);
    let q = v
        .iter()
        .map(|&x| {
            let scaled = (x / scale).round();
            scaled.clamp(-127.0, 127.0) as i8
        })
        .collect();
    (q, scale)
}

/// Inverse of [`quantize_int8`].
pub(crate) fn dequantize_int8(q: &[i8], scale: f32) -> Vec<f32> {
    q.iter().map(|&x| f32::from(x) * scale).collect()
}

/// Apply the query prefix (search side).
pub(crate) fn prefix_query(q: &str) -> String {
    format!("{QUERY_PREFIX}{q}")
}
/// Documents are embedded raw (no prefix).
pub(crate) fn prefix_document(d: &str) -> &str {
    d
}

/// Query-tuned ORT intra-op threads (low; many concurrent daemons). Mirrors
/// `colbert.rs` query default.
fn query_threads() -> usize {
    crate::config::env_usize("SEMANTEX_ORT_THREADS", 4)
}
/// Indexing-tuned ORT intra-op threads (throughput-bound; gated by index::gate).
/// Mirrors `colbert.rs::index_ort_threads`.
fn index_threads() -> usize {
    let cores = std::thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get);
    crate::config::env_usize("SEMANTEX_INDEX_ORT_THREADS", (cores / 2).clamp(2, 8))
}

/// Lazy single-vector encoder. The ONNX session is built on first encode
/// (double-checked locking), exactly like `ColbertEmbedder`.
pub struct SingleVectorEmbedder {
    model_dir: PathBuf,
    threads: usize,
    use_coreml: bool,
    session: OnceLock<Mutex<Session>>,
    tokenizer: OnceLock<Tokenizer>,
    build_lock: std::sync::Mutex<()>,
}

static GLOBAL: OnceLock<SingleVectorEmbedder> = OnceLock::new();
static GLOBAL_INIT_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

impl SingleVectorEmbedder {
    /// CodeRankEmbed embedding dimension (RECORDED).
    pub fn embedding_dim() -> usize {
        EMBEDDING_DIM
    }

    /// Construct a lazy embedder (query thread profile). Cheap; no session.
    pub fn new(model_dir: &Path) -> Result<Self> {
        Self::with_threads(model_dir, query_threads())
    }

    /// Construct a lazy embedder tuned for indexing (more threads).
    pub fn for_indexing(model_dir: &Path) -> Result<Self> {
        Self::with_threads(model_dir, index_threads())
    }

    fn with_threads(model_dir: &Path, threads: usize) -> Result<Self> {
        if !model_dir.exists() {
            anyhow::bail!(
                "CodeRankEmbed model dir does not exist: {}",
                model_dir.display()
            );
        }
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            threads: threads.max(1),
            use_coreml: std::env::var("SEMANTEX_COREML").is_ok_and(|v| v == "1"),
            session: OnceLock::new(),
            tokenizer: OnceLock::new(),
            build_lock: std::sync::Mutex::new(()),
        })
    }

    /// Process-global singleton (query path), mirroring `ColbertEmbedder::global`.
    /// The first caller's `model_dir` wins; subsequent calls return the same
    /// instance regardless of their `model_dir` argument.
    pub fn global(model_dir: &Path) -> Result<&'static SingleVectorEmbedder> {
        if let Some(e) = GLOBAL.get() {
            return Ok(e);
        }
        let _g = GLOBAL_INIT_LOCK.lock();
        if let Some(e) = GLOBAL.get() {
            return Ok(e);
        }
        let e = Self::new(model_dir)?;
        let _ = GLOBAL.set(e);
        Ok(GLOBAL.get().expect("just set"))
    }

    pub fn is_initialized(&self) -> bool {
        self.session.get().is_some()
    }

    /// CPU execution provider pinned on every platform (CoreML only via
    /// `SEMANTEX_COREML` on macOS) — same rationale as `colbert.rs`: a single
    /// multi-thread CPU session; `Auto` would silently collapse to 1 thread on
    /// Linux/Windows where no GPU EP is compiled.
    fn execution_providers(&self) -> Vec<ort::ep::ExecutionProviderDispatch> {
        let mut providers = Vec::new();
        #[cfg(target_os = "macos")]
        if self.use_coreml {
            providers.push(ort::ep::CoreML::default().build());
        }
        #[cfg(not(target_os = "macos"))]
        let _ = self.use_coreml;
        providers.push(ort::ep::CPU::default().build());
        providers
    }

    fn build_session(&self) -> Result<Session> {
        // Defensive: ensure the ONNX Runtime shared lib is provisioned (the CLI
        // already sets ORT_DYLIB_PATH at startup; this is a no-op when present).
        let runtime_root = crate::config::SemantexConfig::semantex_home().join("runtime");
        let _ = crate::embedding::runtime_manager::ensure_onnxruntime(&runtime_root);

        let model_path = self
            .model_dir
            .join(crate::embedding::single_vector_model::CODERANK_ONNX);
        let session = Session::builder()?
            .with_execution_providers(self.execution_providers())?
            .with_intra_threads(self.threads)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(&model_path)?;
        Ok(session)
    }

    fn session(&self) -> Result<&Mutex<Session>> {
        if let Some(s) = self.session.get() {
            return Ok(s);
        }
        let _guard = self
            .build_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(s) = self.session.get() {
            return Ok(s);
        }
        let built = self.build_session()?;
        let _ = self.session.set(Mutex::new(built));
        Ok(self.session.get().expect("session set above"))
    }

    fn tokenizer(&self) -> Result<&Tokenizer> {
        if let Some(t) = self.tokenizer.get() {
            return Ok(t);
        }
        let path = self.model_dir.join("tokenizer.json");
        let tok = Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer {}: {e}", path.display()))?;
        let _ = self.tokenizer.set(tok);
        Ok(self.tokenizer.get().expect("tokenizer set above"))
    }

    /// Encode a document (raw code, no prefix) → L2-normalized embedding.
    pub fn encode_document(&self, text: &str) -> Result<Vec<f32>> {
        self.encode_text(prefix_document(text))
    }

    /// Encode a document with the `SEMANTEX_DENSE_CONTEXT` A/B prefix already
    /// applied by the caller (builder.rs). `context` is the graph-derived NL
    /// annotation; the embedded text is `format!("{context}\n{code}")`. When the
    /// switch is OFF the caller uses [`encode_document`] (raw code) instead.
    pub fn encode_document_with_context(&self, context: &str, code: &str) -> Result<Vec<f32>> {
        if context.is_empty() {
            return self.encode_document(code);
        }
        self.encode_text(&format!("{context}\n{code}"))
    }

    /// Encode a query (RECORDED prefix prepended) → L2-normalized embedding.
    pub fn encode_query(&self, text: &str) -> Result<Vec<f32>> {
        self.encode_text(&prefix_query(text))
    }

    fn encode_text(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer()?
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;

        // Truncate to the model's max context (RECORDED).
        let ids_u32 = encoding.get_ids();
        let mask_u32 = encoding.get_attention_mask();
        let len = ids_u32.len().min(MAX_CTX);
        let ids: Vec<i64> = ids_u32[..len].iter().map(|&x| i64::from(x)).collect();
        let mask: Vec<i64> = mask_u32[..len].iter().map(|&x| i64::from(x)).collect();

        let seq = ids.len() as i64;
        let id_tensor = Tensor::from_array((vec![1_i64, seq], ids))?;
        let mask_tensor = Tensor::from_array((vec![1_i64, seq], mask.clone()))?;

        let session = self.session()?;
        let mut guard = session.lock();
        let outputs = guard.run(ort::inputs![
            INPUT_IDS => id_tensor,
            ATTENTION_MASK => mask_tensor,
        ])?;

        // Output is [1, seq, dim] f32 token embeddings.
        let (shape, data) = outputs[OUTPUT_NAME].try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            shape.len() == 3 && shape[2] as usize == EMBEDDING_DIM,
            "unexpected encoder output shape {shape:?} (expected [1, seq, {EMBEDDING_DIM}])"
        );
        let seq_len = shape[1] as usize;
        let dim = EMBEDDING_DIM;

        // Mean pooling over masked tokens (RECORDED: mean, mask-weighted).
        let mut pooled = vec![0.0f32; dim];
        let mut count = 0.0f32;
        for t in 0..seq_len {
            if mask.get(t).copied().unwrap_or(0) == 0 {
                continue;
            }
            count += 1.0;
            let row = &data[t * dim..(t + 1) * dim];
            for (p, &x) in pooled.iter_mut().zip(row) {
                *p += x;
            }
        }
        if count > 0.0 {
            for p in pooled.iter_mut() {
                *p /= count;
            }
        }
        l2_normalize(&mut pooled);
        // Keep `dot_f32` referenced for the rescore path (imported for callers).
        debug_assert!(dot_f32(&pooled, &pooled) <= 1.0 + 1e-3);
        Ok(pooled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_normalize_unit_length() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "expected unit norm, got {norm}");
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector_is_safe() {
        let mut v = vec![0.0f32, 0.0, 0.0];
        l2_normalize(&mut v); // must not divide by zero / NaN
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn int8_round_trip_preserves_direction() {
        let mut v = vec![0.1f32, -0.5, 0.8, 0.2, -0.3];
        l2_normalize(&mut v);
        let (q, scale) = quantize_int8(&v);
        assert_eq!(q.len(), v.len());
        assert!(scale > 0.0);
        let back = dequantize_int8(&q, scale);
        let dot: f32 = v.iter().zip(&back).map(|(a, b)| a * b).sum();
        let nb: f32 = back.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / nb.max(1e-12);
        assert!(cos > 0.99, "int8 round-trip cosine too low: {cos}");
    }

    #[test]
    fn int8_all_zero_vector_has_nonzero_scale() {
        let (q, scale) = quantize_int8(&[0.0, 0.0, 0.0]);
        assert!(q.iter().all(|&x| x == 0));
        assert!(
            scale > 0.0,
            "scale must never be zero (avoids div-by-zero on dequant)"
        );
    }

    #[test]
    fn embedder_new_is_lazy_no_session_at_construction() {
        let tmp = tempfile::TempDir::new().unwrap();
        let emb = SingleVectorEmbedder::new(tmp.path())
            .expect("constructor must succeed for any existing dir");
        assert!(
            !emb.is_initialized(),
            "session must not build at construction"
        );
    }

    #[test]
    fn embedder_rejects_missing_dir() {
        let res = SingleVectorEmbedder::new(Path::new("/nonexistent/s2/model/dir"));
        assert!(res.is_err(), "missing model dir must fail at construction");
    }

    #[test]
    fn query_prefix_is_applied_document_is_raw() {
        let doc = "fn add(a:i32,b:i32)->i32{a+b}";
        assert_eq!(prefix_document(doc), doc, "documents get NO prefix");
        let q = "add two integers";
        assert_eq!(prefix_query(q), format!("{QUERY_PREFIX}{q}"));
        assert!(
            QUERY_PREFIX.ends_with(' '),
            "RECORDED prefix keeps its trailing space"
        );
    }

    #[test]
    fn embedding_dim_is_recorded_constant() {
        assert_eq!(SingleVectorEmbedder::embedding_dim(), EMBEDDING_DIM);
        assert!(EMBEDDING_DIM > 0);
    }
}
