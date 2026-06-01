//! The `DenseBackend` seam: a trait abstraction over the dense search/build
//! channel so multiple dense backends (today: `colbert-plaid`) can coexist and
//! be selected by config/env. See `docs/superpowers/specs/2026-05-31-semantex-sota-overhaul-design.md` §3/§4 S1.

use crate::types::ScoredChunkId;
use anyhow::Result;
use std::path::{Path, PathBuf};

/// Identity of a dense backend — drives config selection and on-disk paths.
///
/// Today only `colbert-plaid` exists (the ColBERT late-interaction + vendored
/// next-plaid path). S2 adds `coderank-hnsw`. The string form is what gets
/// written into `meta.json` and read from `SEMANTEX_DENSE_BACKEND`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DenseBackendKind {
    /// ColBERT late-interaction over a vendored next-plaid PLAID index.
    #[default]
    ColbertPlaid,
}

impl DenseBackendKind {
    /// Stable on-disk / config identity. MUST stay in sync with `parse`.
    pub fn name(self) -> &'static str {
        match self {
            DenseBackendKind::ColbertPlaid => "colbert-plaid",
        }
    }

    /// Parse a backend name (case-insensitive, whitespace-trimmed).
    /// Returns `None` for an unknown name so callers can fall back to the
    /// default and warn, rather than panicking on a typo'd env var.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "colbert-plaid" => Some(DenseBackendKind::ColbertPlaid),
            _ => None,
        }
    }
}

/// The per-backend on-disk directory: `<index_dir>/dense/<backend>/`.
///
/// Per-backend isolation lets `colbert-plaid` and a future `coderank-hnsw`
/// index coexist on disk during the S2 A/B without clobbering each other.
pub fn dense_subdir(index_dir: &Path, backend: DenseBackendKind) -> PathBuf {
    index_dir.join("dense").join(backend.name())
}

/// Verify that the persisted `dense_backend` in `<index_dir>/meta.json` matches
/// `expected` (mirrors `sparse_search::verify_persisted_stemmer_matches`).
///
/// Returns:
/// * `Ok(())` if the persisted backend agrees with `expected`, OR if meta.json
///   is missing / unparseable (production callers reach this only after
///   `state::detect` has vetted meta.json; in-crate tests open without one).
/// * `Err(anyhow!)` on a value mismatch, naming both backends and pointing the
///   user at `semantex index --rebuild`.
pub fn verify_persisted_backend_matches(index_dir: &Path, expected: &str) -> Result<()> {
    let meta_path = index_dir.join("meta.json");
    let Ok(meta_str) = std::fs::read_to_string(&meta_path) else {
        return Ok(());
    };
    let Ok(meta) = serde_json::from_str::<crate::types::IndexMeta>(&meta_str) else {
        // Unparseable meta.json — `state::detect` returns `Stale` for the same
        // condition, so production callers should never reach here.
        return Ok(());
    };
    if meta.dense_backend != expected {
        anyhow::bail!(
            "dense backend mismatch: index built with dense_backend={}, \
             config says dense_backend={}. Run `semantex index --rebuild` \
             to reconcile.",
            meta.dense_backend,
            expected,
        );
    }
    Ok(())
}

/// A scored chunk returned by the dense channel. Items are sorted by
/// descending `score`. This is the project-wide `ScoredChunkId` (5 fields);
/// dense backends populate only `chunk_id` + `score` (per-channel fields stay
/// zero and are filled by the fusion stage).
pub type DenseHit = ScoredChunkId;

/// Query-time dense backend. Implementations are `Send + Sync` because the
/// `HybridSearcher` shares one instance across the rayon-parallel search
/// channels (`dense_handle` + `exp_dense_handle`).
pub trait DenseBackend: Send + Sync {
    /// Backend identity for on-disk paths + config selection.
    fn name(&self) -> &'static str;

    /// Search the dense channel for a text query, returning the top `k`
    /// `(chunk_id, score)` hits sorted by descending score.
    fn search(&self, query: &str, k: usize) -> Result<Vec<DenseHit>>;

    /// Restrict scoring to a candidate `subset` of chunk IDs (used by the
    /// `file_filter` prefilter). An empty subset MUST yield an empty result.
    fn search_with_subset(&self, query: &str, k: usize, subset: &[u64]) -> Result<Vec<DenseHit>>;

    /// Positional doc→chunk mapping, if this backend keeps one (colbert-plaid
    /// does; HNSW will not). Used by `hybrid.rs` to build the `file_filter`
    /// candidate subset. Returns `None` for backends without positional docs.
    fn positional_chunk_ids(&self) -> Option<&[u64]> {
        None
    }

    // optional vector accessors for S7 (MMR / semantic cache); colbert-plaid provides a mean-pooled+L2-normalized projection, coderank-hnsw (S2) returns its exact int8-store vectors.
    fn embed_text_vector(&self, _query: &str) -> Option<Vec<f32>> {
        None
    }
    fn embed_doc_vectors(&self, _chunk_ids: &[u64]) -> Option<Vec<(u64, Vec<f32>)>> {
        None
    }
}

/// Build-time dense index builder. Mirrors the dense build/update lifecycle
/// the PLAID block in `index/builder.rs` performs today.
pub trait DenseIndexBuilder: Send + Sync {
    /// Backend identity (matches the query-side `DenseBackend::name`).
    fn name(&self) -> &'static str;

    /// Full (re)build from the complete `(chunk_id, content)` corpus.
    fn build(&mut self, chunks: &[(u64, &str)]) -> Result<()>;

    /// Incremental add of new `(chunk_id, content)` pairs.
    fn insert(&mut self, chunks: &[(u64, &str)]) -> Result<()>;

    /// Incremental delete of the given chunk IDs from the dense index.
    fn delete(&mut self, chunk_ids: &[u64]) -> Result<()>;

    /// Persist the dense index + any sidecar mapping into `dir`
    /// (a per-backend `dense/<backend>/` directory).
    fn persist(&self, dir: &Path) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_backend_kind_default_is_colbert_plaid() {
        assert_eq!(DenseBackendKind::default(), DenseBackendKind::ColbertPlaid);
        assert_eq!(DenseBackendKind::default().name(), "colbert-plaid");
    }

    #[test]
    fn parse_known_backend_names() {
        assert_eq!(
            DenseBackendKind::parse("colbert-plaid"),
            Some(DenseBackendKind::ColbertPlaid)
        );
        // Whitespace + case are normalized.
        assert_eq!(
            DenseBackendKind::parse("  Colbert-Plaid  "),
            Some(DenseBackendKind::ColbertPlaid)
        );
    }

    #[test]
    fn parse_unknown_backend_is_none() {
        assert_eq!(DenseBackendKind::parse("totally-made-up"), None);
        assert_eq!(DenseBackendKind::parse(""), None);
    }

    #[test]
    fn dense_subdir_is_per_backend() {
        let root = Path::new("/tmp/proj/.semantex");
        let p = dense_subdir(root, DenseBackendKind::ColbertPlaid);
        assert_eq!(p, Path::new("/tmp/proj/.semantex/dense/colbert-plaid"));
    }

    #[test]
    fn verify_backend_matches_on_agreement() {
        let tmp = tempfile::TempDir::new().unwrap();
        let index_dir = tmp.path();
        write_meta_with_backend(index_dir, "colbert-plaid");
        // Matching backend → Ok.
        verify_persisted_backend_matches(index_dir, "colbert-plaid").unwrap();
    }

    #[test]
    fn verify_backend_errors_on_mismatch() {
        let tmp = tempfile::TempDir::new().unwrap();
        let index_dir = tmp.path();
        write_meta_with_backend(index_dir, "colbert-plaid");
        let err = verify_persisted_backend_matches(index_dir, "coderank-hnsw")
            .expect_err("mismatched backend must error");
        let msg = err.to_string();
        assert!(msg.contains("dense backend mismatch"), "got: {msg}");
        assert!(
            msg.contains("colbert-plaid") && msg.contains("coderank-hnsw"),
            "got: {msg}"
        );
        assert!(msg.contains("semantex index --rebuild"), "got: {msg}");
    }

    #[test]
    fn verify_backend_skips_when_meta_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        // No meta.json written — skip the check (mirrors stemmer guard).
        verify_persisted_backend_matches(tmp.path(), "colbert-plaid").unwrap();
    }

    /// Helper: write a current-shape meta.json carrying `backend`.
    fn write_meta_with_backend(index_dir: &Path, backend: &str) {
        let meta = crate::types::IndexMeta {
            schema_version: crate::types::IndexMeta::CURRENT_SCHEMA_VERSION,
            project_path: index_dir.to_path_buf(),
            created_at: "0".to_string(),
            updated_at: "0".to_string(),
            file_count: 0,
            chunk_count: 0,
            embedding_model: "test".to_string(),
            embedding_dim: 48,
            use_bm25_stemmer: true,
            dense_backend: backend.to_string(),
        };
        std::fs::write(
            index_dir.join("meta.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();
    }
}
