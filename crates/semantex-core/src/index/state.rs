//! Index state detection — determines whether a project's semantex index is ready,
//! currently being built, stale (schema mismatch), or absent.

use crate::types::IndexMeta;
use std::path::Path;

/// The current state of a project's semantex index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexState {
    /// No index exists for this project.
    NotIndexed,
    /// An index build is currently in progress (lock held).
    Building,
    /// Index exists but has an outdated schema version — needs rebuild.
    Stale,
    /// Index exists and is ready for queries.
    Ready,
}

/// Detect the index state for a project at `project_path`.
///
/// Checks for `<project>/.semantex/meta.json` existence, validates schema version,
/// then probes the lock file with a non-blocking `flock` to distinguish
/// `Building` from `Ready`.
pub fn detect(project_path: &Path) -> IndexState {
    let semantex_dir = project_path.join(".semantex");
    let meta_path = semantex_dir.join("meta.json");

    if !meta_path.exists() {
        // No meta.json — check if a build is in progress via lock file
        let lock_path = semantex_dir.join(".semantex.lock");
        if is_locked(&lock_path) {
            return IndexState::Building;
        }
        return IndexState::NotIndexed;
    }

    // Meta exists — check schema version
    if is_stale(&meta_path) {
        return IndexState::Stale;
    }

    // Check if a rebuild is in progress
    let lock_path = semantex_dir.join(".semantex.lock");
    if is_locked(&lock_path) {
        return IndexState::Building;
    }

    IndexState::Ready
}

/// Returns the age of the index in seconds since its last update, or `None` if
/// no index exists or the timestamp cannot be parsed.
pub fn index_age_secs(project_path: &Path) -> Option<u64> {
    let meta_path = project_path.join(".semantex").join("meta.json");
    let content = std::fs::read_to_string(meta_path).ok()?;
    let meta: crate::types::IndexMeta = serde_json::from_str(&content).ok()?;
    // updated_at is stored as Unix epoch seconds (plain integer string)
    let updated_epoch: u64 = meta.updated_at.trim().parse().ok()?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    Some(now.saturating_sub(updated_epoch))
}

/// Check if the index has an outdated schema version.
///
/// Exposed so callers that already know `meta.json` is present can re-validate
/// without re-running the full `detect` pass (used by the MCP warm-state fast
/// path to cheaply enforce the staleness invariant).
pub fn is_stale(meta_path: &Path) -> bool {
    let Ok(content) = std::fs::read_to_string(meta_path) else {
        return true; // unreadable meta.json → treat as stale
    };
    let meta: IndexMeta = match serde_json::from_str(&content) {
        Ok(m) => m,
        Err(_) => return true, // unparseable meta → treat as stale
    };
    meta.schema_version != IndexMeta::CURRENT_SCHEMA_VERSION
}

/// Try to acquire a non-blocking exclusive lock on the file.
/// Returns `true` if the file is currently locked by another process.
/// Uses `File::try_lock()` (stabilized in Rust 1.84) for cross-platform support.
///
/// Exposed so warm-state fast paths can cheaply re-validate that no concurrent
/// rebuild is in progress without re-running the full `detect` pass. A single
/// `flock` syscall — sub-microsecond on warm cache.
pub fn is_locked(lock_path: &Path) -> bool {
    let Ok(file) = std::fs::File::open(lock_path) else {
        return false;
    };

    match file.try_lock() {
        Err(std::fs::TryLockError::WouldBlock) => {
            // Another process holds the lock.
            true
        }
        // Ok: we acquired the lock — nobody else holds it (released on drop).
        // Error: other error (NFS, unsupported FS) — assume not locked.
        Ok(()) | Err(std::fs::TryLockError::Error(_)) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_not_indexed() {
        let tmp = TempDir::new().unwrap();
        assert_eq!(detect(tmp.path()), IndexState::NotIndexed);
    }

    #[test]
    fn test_ready() {
        let tmp = TempDir::new().unwrap();
        let semantex_dir = tmp.path().join(".semantex");
        std::fs::create_dir_all(&semantex_dir).unwrap();
        let meta = serde_json::json!({
            "schema_version": IndexMeta::CURRENT_SCHEMA_VERSION,
            "project_path": tmp.path(),
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "file_count": 10,
            "chunk_count": 50,
            "embedding_model": "test",
            "embedding_dim": 48,
            "use_bm25_stemmer": true,
            "dense_backend": "coderank-hnsw",
            "embedder_fingerprint": "test",
        });
        std::fs::write(semantex_dir.join("meta.json"), meta.to_string()).unwrap();
        assert_eq!(detect(tmp.path()), IndexState::Ready);
    }

    #[test]
    fn test_stale_schema() {
        let tmp = TempDir::new().unwrap();
        let semantex_dir = tmp.path().join(".semantex");
        std::fs::create_dir_all(&semantex_dir).unwrap();
        let meta = serde_json::json!({
            "schema_version": 1,
            "project_path": tmp.path(),
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "file_count": 10,
            "chunk_count": 50,
            "embedding_model": "test",
            "embedding_dim": 48
        });
        std::fs::write(semantex_dir.join("meta.json"), meta.to_string()).unwrap();
        assert_eq!(detect(tmp.path()), IndexState::Stale);
    }

    #[test]
    fn test_unreadable_meta_treated_as_stale() {
        let tmp = TempDir::new().unwrap();
        let semantex_dir = tmp.path().join(".semantex");
        std::fs::create_dir_all(&semantex_dir).unwrap();
        // Write invalid JSON to meta.json
        std::fs::write(semantex_dir.join("meta.json"), "not valid json").unwrap();
        assert_eq!(detect(tmp.path()), IndexState::Stale);
    }

    #[test]
    fn test_building_with_lock() {
        let tmp = TempDir::new().unwrap();
        let semantex_dir = tmp.path().join(".semantex");
        std::fs::create_dir_all(&semantex_dir).unwrap();
        let lock_path = semantex_dir.join(".semantex.lock");
        let lock_file = std::fs::File::create(&lock_path).unwrap();

        // Hold an exclusive lock (cross-platform)
        lock_file.lock().expect("Failed to acquire test lock");

        assert_eq!(detect(tmp.path()), IndexState::Building);

        // Lock released when lock_file drops
    }

    /// Regression: a pre-v0.3 index (schema_version=7) must be detected as
    /// `Stale` after the v8 bump, so the MCP/CLI rebuild path runs and creates
    /// the v0.3 auxiliary tables (`chunk_annotations`, `pattern_matches`,
    /// `chunk_centrality`) before M5/M6 handlers query them.
    #[test]
    fn test_pre_v0_3_schema_v7_is_stale() {
        // Sanity: this test only makes sense as long as the current schema
        // is past v7. If we ever roll back, the test should be updated.
        assert!(
            IndexMeta::CURRENT_SCHEMA_VERSION > 7,
            "current schema version regressed; revisit pre-v0.3 stale test"
        );

        let tmp = TempDir::new().unwrap();
        let semantex_dir = tmp.path().join(".semantex");
        std::fs::create_dir_all(&semantex_dir).unwrap();

        let meta = IndexMeta {
            schema_version: 7,
            project_path: tmp.path().to_path_buf(),
            created_at: "0".to_string(),
            updated_at: "0".to_string(),
            file_count: 0,
            chunk_count: 0,
            embedding_model: "test".to_string(),
            embedding_dim: 48,
            use_bm25_stemmer: true,
            dense_backend: "coderank-hnsw".to_string(),
            embedder_fingerprint: "test".to_string(),
        };
        let meta_json = serde_json::to_string(&meta).unwrap();
        std::fs::write(semantex_dir.join("meta.json"), meta_json).unwrap();

        assert_eq!(detect(tmp.path()), IndexState::Stale);
    }
}
