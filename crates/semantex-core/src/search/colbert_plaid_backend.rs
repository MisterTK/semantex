//! `colbert-plaid` — the first `DenseBackend`/`DenseIndexBuilder` impl,
//! wrapping the ColBERT late-interaction + vendored next-plaid PLAID path.
//! Behavior is byte-identical to the pre-seam inline PLAID code.

use crate::embedding::colbert::ColbertEmbedder;
use crate::search::dense_backend::{DenseBackend, DenseHit};
use crate::search::plaid_search::PlaidSearcher;
use anyhow::Result;
use ndarray::Axis;
use std::path::Path;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_name_is_colbert_plaid() {
        assert_eq!(ColbertPlaidBackend::NAME, "colbert-plaid");
    }

    /// S1/S7 seam: `embed_text_vector` returns a `Some(Vec<f32>)` of the model
    /// dimension (48) for a non-empty query. Opening a real backend needs a
    /// PLAID index + the ColBERT model, so this is `#[ignore]`'d (run with
    /// `--ignored`); it builds a tiny synthetic repo (repo-agnostic tempdir,
    /// no hardcoded paths) and opens the colbert-plaid backend from the
    /// per-backend dense subdir.
    #[test]
    #[ignore] // builds a PLAID index + loads the ColBERT model; run with --ignored
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
        assert_eq!(v.len(), 48, "mean-pooled query vector must have the model dim");
        // L2-normalized → unit length (within float tolerance).
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "projection must be L2-normalized, got norm={norm}"
        );
    }
}
