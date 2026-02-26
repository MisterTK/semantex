//! ColBERT multi-vector encoder for late-interaction retrieval.
//!
//! Uses LateOn-Code-edge via `next-plaid-onnx` for per-token embeddings (48d).

use anyhow::Result;
use ndarray::Array2;
use next_plaid_onnx::Colbert;
use parking_lot::Mutex;
use std::path::Path;
use std::sync::OnceLock;

/// Per-token embedding matrix: rows = tokens, cols = embedding dimension (48).
pub type TokenEmbeddings = Array2<f32>;

/// ColBERT encoder wrapping LateOn-Code-edge via `next-plaid-onnx`.
///
/// Produces per-token embeddings (N_tokens x 48d) for both queries and documents.
/// The encoder is thread-safe: the inner ONNX session is protected by a
/// `parking_lot::Mutex` (single-threaded inference).
pub struct ColbertEmbedder {
    encoder: Mutex<Colbert>,
}

/// Global singleton for `ColbertEmbedder`.
static GLOBAL_COLBERT: OnceLock<ColbertEmbedder> = OnceLock::new();
/// Serializes initialization of the global ColBERT embedder (double-checked locking).
static COLBERT_INIT_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

impl ColbertEmbedder {
    /// Initialize from a model directory containing `model_int8.onnx` and `tokenizer.json`.
    ///
    /// Thread count is read from `SEMANTEX_ORT_THREADS` (default 4).
    /// CoreML acceleration is gated behind `SEMANTEX_COREML=1`.
    pub fn new(model_dir: &Path) -> Result<Self> {
        let threads: usize = std::env::var("SEMANTEX_ORT_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);

        #[allow(unused_mut)]
        let mut builder = Colbert::builder(model_dir)
            .with_quantized(true)
            .with_threads(threads);

        #[cfg(target_os = "macos")]
        if std::env::var("SEMANTEX_COREML").is_ok_and(|v| v == "1") {
            builder = builder.with_execution_provider(next_plaid_onnx::ExecutionProvider::CoreML);
        }

        let encoder = builder.build()?;
        Ok(Self {
            encoder: Mutex::new(encoder),
        })
    }

    /// Get or initialize the global singleton (double-checked locking).
    ///
    /// The first call creates the encoder; subsequent calls return the existing
    /// instance. Returns an error only if initialization itself fails.
    pub fn global(model_dir: &Path) -> Result<&'static ColbertEmbedder> {
        if let Some(embedder) = GLOBAL_COLBERT.get() {
            return Ok(embedder);
        }
        let _guard = COLBERT_INIT_LOCK.lock();
        if let Some(embedder) = GLOBAL_COLBERT.get() {
            return Ok(embedder);
        }
        tracing::info!("Initializing global ColBERT encoder singleton");
        let embedder = Self::new(model_dir)?;
        let _ = GLOBAL_COLBERT.set(embedder);
        Ok(GLOBAL_COLBERT.get().expect("just set"))
    }

    /// Encode a search query into per-token embeddings `[N_query_tokens, 48]`.
    pub fn encode_query(&self, text: &str) -> Result<TokenEmbeddings> {
        let encoder = self.encoder.lock();
        let mut embeddings = encoder.encode_queries(&[text])?;
        embeddings
            .pop()
            .ok_or_else(|| anyhow::anyhow!("encode_queries returned empty result"))
    }

    /// Encode documents into per-token embeddings, one `Array2<f32>` per document.
    pub fn encode_documents(&self, texts: &[String]) -> Result<Vec<TokenEmbeddings>> {
        let encoder = self.encoder.lock();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let embeddings = encoder.encode_documents(&refs, None)?;
        Ok(embeddings)
    }
}
