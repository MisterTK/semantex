//! The `DenseBackend` seam: a trait abstraction over the dense search/build
//! channel so dense backends can be selected by config/env and future backends
//! slot in without touching call sites. Today the sole built-in backend is
//! `coderank-hnsw` (CodeRankEmbed single-vector + instant-distance HNSW); the
//! enum is kept so a new backend is one variant + one match arm away. See
//! `docs/superpowers/specs/2026-05-31-semantex-sota-overhaul-design.md` §3/§4 S1.

use crate::types::ScoredChunkId;
use anyhow::Result;
use std::path::{Path, PathBuf};

/// Identity of a dense backend — drives config selection and on-disk paths.
///
/// Single-variant today (`coderank-hnsw`), but kept an enum behind the seam so a
/// future backend slots in. The string form is what gets written into
/// `meta.json` and read from `SEMANTEX_DENSE_BACKEND`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DenseBackendKind {
    /// CodeRankEmbed single-vector embeddings over a pure-Rust HNSW index.
    #[default]
    CoderankHnsw,
}

impl DenseBackendKind {
    /// Stable on-disk / config identity. MUST stay in sync with `parse`.
    pub fn name(self) -> &'static str {
        match self {
            DenseBackendKind::CoderankHnsw => "coderank-hnsw",
        }
    }

    /// Parse a backend name (case-insensitive, whitespace-trimmed).
    /// Returns `None` for an unknown name (e.g. a stale `colbert-plaid` from an
    /// old config, or a typo) so callers fall back to the default rather than
    /// panicking — old indexes degrade to a clean rebuild, not a crash.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "coderank-hnsw" => Some(DenseBackendKind::CoderankHnsw),
            _ => None,
        }
    }
}

/// The per-backend on-disk directory: `<index_dir>/dense/<backend>/`.
///
/// Per-backend isolation lets multiple backends coexist on disk (e.g. a future
/// A/B) without clobbering each other.
pub fn dense_subdir(index_dir: &Path, backend: DenseBackendKind) -> PathBuf {
    index_dir.join("dense").join(backend.name())
}

/// The versioned dense index dir for a specific embedder fingerprint:
/// `<index_dir>/dense/<backend>/<fingerprint>/`. A new embedder builds here
/// alongside the old one, so the live index is never disturbed mid-rebuild (S8
/// zero-downtime switchover).
pub fn active_dense_dir(index_dir: &Path, backend: DenseBackendKind, fingerprint: &str) -> PathBuf {
    dense_subdir(index_dir, backend).join(fingerprint)
}

/// The per-backend "dense index present" sentinel file. coderank-hnsw writes
/// `vectors.bin`. Its presence in a dense dir marks the store as built (else:
/// rebuild). Owned by the seam so both the builder and the reader-side
/// [`resolve_active_dense_dir`] agree on what "present" means per backend.
pub fn dense_sentinel_file(backend: DenseBackendKind) -> &'static str {
    match backend {
        DenseBackendKind::CoderankHnsw => "vectors.bin",
    }
}

/// Resolve the directory the dense store actually lives in: the ACTIVE
/// versioned dir (`dense/<backend>/<fingerprint>/`) when an ACTIVE pointer
/// exists and its versioned dir holds the store sentinel, else the legacy
/// plain `dense_subdir` (pre-versioned indexes built before S8 hot-swap).
///
/// This is the single resolver both readers (`hybrid.rs`, `validate.rs`,
/// cache-warming in `storage.rs`) and the builder's presence check go through,
/// so a live versioned store is found while legacy plain-layout indexes still
/// open via the fallback (backward-compat, no schema bump).
pub fn resolve_active_dense_dir(index_dir: &Path, backend: DenseBackendKind) -> PathBuf {
    if let Some(fp) = read_active_pointer(index_dir, backend) {
        let versioned = active_dense_dir(index_dir, backend, &fp);
        if versioned.join(dense_sentinel_file(backend)).exists() {
            return versioned;
        }
    }
    // No pointer, or the pointed-at versioned dir is missing/empty → fall back
    // to the legacy plain layout (still valid for pre-S8 indexes).
    dense_subdir(index_dir, backend)
}

/// Path of the active-pointer file for a backend: `<index_dir>/dense/<backend>/ACTIVE`.
/// Its contents are the fingerprint of the currently-live versioned dir.
fn active_pointer_path(index_dir: &Path, backend: DenseBackendKind) -> PathBuf {
    dense_subdir(index_dir, backend).join("ACTIVE")
}

/// Read the currently-active fingerprint for `backend`, or `None` if no pointer
/// exists yet (fresh index).
pub fn read_active_pointer(index_dir: &Path, backend: DenseBackendKind) -> Option<String> {
    std::fs::read_to_string(active_pointer_path(index_dir, backend))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Flip the active pointer to `fingerprint` atomically (write a temp file in the
/// same dir, then rename — rename is atomic on the same filesystem). The new
/// versioned dir must already be fully built before this is called, so readers
/// either see the old fingerprint or the new one, never a partial index.
pub fn write_active_pointer(
    index_dir: &Path,
    backend: DenseBackendKind,
    fingerprint: &str,
) -> Result<()> {
    let dir = dense_subdir(index_dir, backend);
    std::fs::create_dir_all(&dir)?;
    let final_path = dir.join("ACTIVE");
    // PID-suffixed temp name so two concurrent rebuilds don't clobber each
    // other's staging file before the atomic rename.
    let tmp_path = dir.join(format!(".ACTIVE.{}.tmp", std::process::id()));
    std::fs::write(&tmp_path, fingerprint.as_bytes())?;
    std::fs::rename(&tmp_path, &final_path)?;
    Ok(())
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

/// Verify the persisted `embedder_fingerprint` in `<index_dir>/meta.json` matches
/// `expected`. Mirrors [`verify_persisted_backend_matches`]: `Ok(())` when meta
/// is missing/unparseable (production callers reach here only after
/// `state::detect` vetted meta), `Err` on a value mismatch — pointing the user at
/// a rebuild. This is the uniform "index stamped with its embedder; mismatch →
/// rebuild" check that generalizes the schema/stemmer/backend guards (S8).
pub fn verify_persisted_fingerprint_matches(index_dir: &Path, expected: &str) -> Result<()> {
    let meta_path = index_dir.join("meta.json");
    let Ok(meta_str) = std::fs::read_to_string(&meta_path) else {
        return Ok(());
    };
    let Ok(meta) = serde_json::from_str::<crate::types::IndexMeta>(&meta_str) else {
        return Ok(());
    };
    if meta.embedder_fingerprint != expected {
        anyhow::bail!(
            "embedder changed: index built with embedder_fingerprint={}, \
             config's active embedder has fingerprint={}. Run \
             `semantex index --rebuild` to re-embed under the new model.",
            meta.embedder_fingerprint,
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

    /// Positional doc→chunk mapping, if this backend keeps one. Used by
    /// `hybrid.rs` to build the `file_filter` candidate subset. Returns `None`
    /// for backends without positional docs (coderank-hnsw does not keep one).
    fn positional_chunk_ids(&self) -> Option<&[u64]> {
        None
    }

    // optional vector accessors for S7 (MMR / semantic cache); coderank-hnsw returns its exact int8-store vectors.
    fn embed_text_vector(&self, _query: &str) -> Option<Vec<f32>> {
        None
    }
    fn embed_doc_vectors(&self, _chunk_ids: &[u64]) -> Option<Vec<(u64, Vec<f32>)>> {
        None
    }
}

/// Build-time dense index builder. Mirrors the dense build/update lifecycle
/// the dense block in `index/builder.rs` performs today.
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

    /// S7 consumes (does NOT declare) the S1 seam: a backend that does not
    /// override `embed_doc_vectors` returns None, so the MMR pass safely no-ops.
    #[test]
    fn embed_doc_vectors_defaults_to_none() {
        struct StubBackend;
        impl DenseBackend for StubBackend {
            fn name(&self) -> &'static str {
                "stub"
            }
            fn search(&self, _q: &str, _k: usize) -> Result<Vec<DenseHit>> {
                Ok(vec![])
            }
            fn search_with_subset(&self, _q: &str, _k: usize, _s: &[u64]) -> Result<Vec<DenseHit>> {
                Ok(vec![])
            }
        }
        let b = StubBackend;
        // S1's signature: Option<Vec<(u64, Vec<f32>)>>, default None.
        assert!(b.embed_doc_vectors(&[1, 2, 3]).is_none());
        // And the text-vector seam (consumed by the cache) also defaults None.
        assert!(b.embed_text_vector("anything").is_none());
    }

    #[test]
    fn dense_backend_kind_default_is_coderank_hnsw() {
        assert_eq!(DenseBackendKind::default(), DenseBackendKind::CoderankHnsw);
        assert_eq!(DenseBackendKind::default().name(), "coderank-hnsw");
    }

    #[test]
    fn parse_unknown_backend_is_none() {
        // A removed/stale backend name no longer parses → falls back to default.
        assert_eq!(DenseBackendKind::parse("colbert-plaid"), None);
        assert_eq!(DenseBackendKind::parse("totally-made-up"), None);
        assert_eq!(DenseBackendKind::parse(""), None);
    }

    #[test]
    fn parse_coderank_hnsw_backend() {
        assert_eq!(
            DenseBackendKind::parse("coderank-hnsw"),
            Some(DenseBackendKind::CoderankHnsw)
        );
        assert_eq!(
            DenseBackendKind::parse("  Coderank-HNSW "),
            Some(DenseBackendKind::CoderankHnsw)
        );
        assert_eq!(DenseBackendKind::CoderankHnsw.name(), "coderank-hnsw");
    }

    #[test]
    fn coderank_dense_subdir() {
        let p = dense_subdir(Path::new("/x/.semantex"), DenseBackendKind::CoderankHnsw);
        assert_eq!(p, Path::new("/x/.semantex/dense/coderank-hnsw"));
    }

    #[test]
    fn verify_backend_matches_on_agreement() {
        let tmp = tempfile::TempDir::new().unwrap();
        let index_dir = tmp.path();
        write_meta_with_backend(index_dir, "coderank-hnsw");
        // Matching backend → Ok.
        verify_persisted_backend_matches(index_dir, "coderank-hnsw").unwrap();
    }

    #[test]
    fn verify_backend_errors_on_mismatch() {
        // An old index built with a now-removed backend (`colbert-plaid`) opened
        // under the current default (`coderank-hnsw`) must error CLEANLY with
        // rebuild guidance — NOT panic. This is the graceful-degradation guard
        // for stragglers that survived the schema bump.
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
        verify_persisted_backend_matches(tmp.path(), "coderank-hnsw").unwrap();
    }

    #[test]
    fn verify_fingerprint_errors_on_mismatch() {
        let tmp = tempfile::TempDir::new().unwrap();
        let index_dir = tmp.path();
        write_meta_with_fingerprint(index_dir, "coderank-hnsw", "OLDFP");
        let err = verify_persisted_fingerprint_matches(index_dir, "NEWFP")
            .expect_err("fingerprint mismatch must error");
        let msg = err.to_string();
        assert!(
            msg.contains("embedder changed") || msg.contains("fingerprint"),
            "got: {msg}"
        );
        assert!(msg.contains("OLDFP") && msg.contains("NEWFP"), "got: {msg}");
    }

    #[test]
    fn verify_fingerprint_ok_on_match() {
        let tmp = tempfile::TempDir::new().unwrap();
        write_meta_with_fingerprint(tmp.path(), "coderank-hnsw", "SAME");
        verify_persisted_fingerprint_matches(tmp.path(), "SAME").unwrap();
    }

    #[test]
    fn verify_fingerprint_skips_when_meta_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        verify_persisted_fingerprint_matches(tmp.path(), "anything").unwrap();
    }

    /// Helper: write a current-shape meta.json carrying `backend` + `fingerprint`.
    fn write_meta_with_fingerprint(index_dir: &Path, backend: &str, fingerprint: &str) {
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
            embedder_fingerprint: fingerprint.to_string(),
        };
        std::fs::write(
            index_dir.join("meta.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn versioned_dir_nests_fingerprint_under_backend() {
        let root = Path::new("/tmp/proj/.semantex");
        let p = active_dense_dir(root, DenseBackendKind::CoderankHnsw, "deadbeef");
        assert_eq!(
            p,
            Path::new("/tmp/proj/.semantex/dense/coderank-hnsw/deadbeef")
        );
    }

    #[test]
    fn active_pointer_round_trips() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        // No pointer yet → None.
        assert_eq!(
            read_active_pointer(root, DenseBackendKind::CoderankHnsw),
            None
        );
        // Write then read back.
        write_active_pointer(root, DenseBackendKind::CoderankHnsw, "abc123").unwrap();
        assert_eq!(
            read_active_pointer(root, DenseBackendKind::CoderankHnsw),
            Some("abc123".to_string())
        );
        // Overwrite flips atomically.
        write_active_pointer(root, DenseBackendKind::CoderankHnsw, "def456").unwrap();
        assert_eq!(
            read_active_pointer(root, DenseBackendKind::CoderankHnsw),
            Some("def456".to_string())
        );
    }

    #[test]
    fn resolve_active_dense_dir_no_pointer_returns_plain() {
        // (a) No ACTIVE pointer → the legacy plain layout.
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let got = resolve_active_dense_dir(root, DenseBackendKind::CoderankHnsw);
        assert_eq!(got, dense_subdir(root, DenseBackendKind::CoderankHnsw));
    }

    #[test]
    fn resolve_active_dense_dir_pointer_plus_populated_returns_versioned() {
        // (b) Pointer present AND the versioned dir holds the store sentinel.
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let fp = "deadbeef";
        let versioned = active_dense_dir(root, DenseBackendKind::CoderankHnsw, fp);
        std::fs::create_dir_all(&versioned).unwrap();
        std::fs::write(
            versioned.join(dense_sentinel_file(DenseBackendKind::CoderankHnsw)),
            b"x",
        )
        .unwrap();
        write_active_pointer(root, DenseBackendKind::CoderankHnsw, fp).unwrap();
        let got = resolve_active_dense_dir(root, DenseBackendKind::CoderankHnsw);
        assert_eq!(got, versioned);
    }

    #[test]
    fn resolve_active_dense_dir_pointer_but_missing_sentinel_falls_back() {
        // (c) Pointer present but the versioned dir lacks the sentinel (e.g. a
        // crashed/partial build never flipped a complete store) → plain fallback.
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let fp = "deadbeef";
        // Create the versioned dir but DO NOT write the sentinel.
        std::fs::create_dir_all(active_dense_dir(root, DenseBackendKind::CoderankHnsw, fp))
            .unwrap();
        write_active_pointer(root, DenseBackendKind::CoderankHnsw, fp).unwrap();
        let got = resolve_active_dense_dir(root, DenseBackendKind::CoderankHnsw);
        assert_eq!(got, dense_subdir(root, DenseBackendKind::CoderankHnsw));
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
            embedder_fingerprint: "test".to_string(),
        };
        std::fs::write(
            index_dir.join("meta.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();
    }
}
