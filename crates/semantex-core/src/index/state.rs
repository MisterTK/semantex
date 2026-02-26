//! Index state detection — determines whether a project's sage index is ready,
//! currently being built, stale (schema mismatch), or absent.

use crate::types::IndexMeta;
use std::path::Path;

/// The current state of a project's sage index.
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
    let sage_dir = project_path.join(".semantex");
    let meta_path = sage_dir.join("meta.json");

    if !meta_path.exists() {
        // No meta.json — check if a build is in progress via lock file
        let lock_path = sage_dir.join(".sage.lock");
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
    let lock_path = sage_dir.join(".sage.lock");
    if is_locked(&lock_path) {
        return IndexState::Building;
    }

    IndexState::Ready
}

/// Check if the index has an outdated schema version.
fn is_stale(meta_path: &Path) -> bool {
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
fn is_locked(lock_path: &Path) -> bool {
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
        let sage_dir = tmp.path().join(".semantex");
        std::fs::create_dir_all(&sage_dir).unwrap();
        let meta = serde_json::json!({
            "schema_version": IndexMeta::CURRENT_SCHEMA_VERSION,
            "project_path": tmp.path(),
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "file_count": 10,
            "chunk_count": 50,
            "embedding_model": "test",
            "embedding_dim": 48
        });
        std::fs::write(sage_dir.join("meta.json"), meta.to_string()).unwrap();
        assert_eq!(detect(tmp.path()), IndexState::Ready);
    }

    #[test]
    fn test_stale_schema() {
        let tmp = TempDir::new().unwrap();
        let sage_dir = tmp.path().join(".semantex");
        std::fs::create_dir_all(&sage_dir).unwrap();
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
        std::fs::write(sage_dir.join("meta.json"), meta.to_string()).unwrap();
        assert_eq!(detect(tmp.path()), IndexState::Stale);
    }

    #[test]
    fn test_unreadable_meta_treated_as_stale() {
        let tmp = TempDir::new().unwrap();
        let sage_dir = tmp.path().join(".semantex");
        std::fs::create_dir_all(&sage_dir).unwrap();
        // Write invalid JSON to meta.json
        std::fs::write(sage_dir.join("meta.json"), "not valid json").unwrap();
        assert_eq!(detect(tmp.path()), IndexState::Stale);
    }

    #[test]
    fn test_building_with_lock() {
        let tmp = TempDir::new().unwrap();
        let sage_dir = tmp.path().join(".semantex");
        std::fs::create_dir_all(&sage_dir).unwrap();
        let lock_path = sage_dir.join(".sage.lock");
        let lock_file = std::fs::File::create(&lock_path).unwrap();

        // Hold an exclusive lock (cross-platform)
        lock_file.lock().expect("Failed to acquire test lock");

        assert_eq!(detect(tmp.path()), IndexState::Building);

        // Lock released when lock_file drops
    }
}
