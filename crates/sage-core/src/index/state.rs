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
/// Checks for `<project>/.sage/meta.json` existence, validates schema version,
/// then probes the lock file with a non-blocking `flock` to distinguish
/// `Building` from `Ready`.
pub fn detect(project_path: &Path) -> IndexState {
    let sage_dir = project_path.join(".sage");
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
    let content = match std::fs::read_to_string(meta_path) {
        Ok(c) => c,
        Err(_) => return true, // unreadable meta.json → treat as stale
    };
    let meta: IndexMeta = match serde_json::from_str(&content) {
        Ok(m) => m,
        Err(_) => return true, // unparseable meta → treat as stale
    };
    meta.schema_version != IndexMeta::CURRENT_SCHEMA_VERSION
}

/// Try to acquire a non-blocking exclusive lock on the file.
/// Returns `true` if the file is currently locked by another process.
#[cfg(unix)]
fn is_locked(lock_path: &Path) -> bool {
    use std::os::unix::io::AsRawFd;

    let file = match std::fs::File::open(lock_path) {
        Ok(f) => f,
        Err(_) => return false,
    };

    let fd = file.as_raw_fd();

    // SAFETY: fd is a valid file descriptor from an open File.
    // flock with LOCK_EX | LOCK_NB is a safe, non-destructive probe.
    let ret = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };

    if ret == 0 {
        // We acquired the lock — nobody else holds it. File drop releases it.
        false
    } else {
        // Check errno: only EWOULDBLOCK means "locked by another process".
        // Other errors (ENOTSUP on NFS/FUSE, EINTR, etc.) → assume not locked.
        let err = std::io::Error::last_os_error();
        matches!(err.raw_os_error(), Some(libc::EWOULDBLOCK))
    }
}

/// Non-unix fallback — cannot probe locks, so assume not locked.
#[cfg(not(unix))]
fn is_locked(_lock_path: &Path) -> bool {
    false
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
        let sage_dir = tmp.path().join(".sage");
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
        let sage_dir = tmp.path().join(".sage");
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
        let sage_dir = tmp.path().join(".sage");
        std::fs::create_dir_all(&sage_dir).unwrap();
        // Write invalid JSON to meta.json
        std::fs::write(sage_dir.join("meta.json"), "not valid json").unwrap();
        assert_eq!(detect(tmp.path()), IndexState::Stale);
    }

    #[cfg(unix)]
    #[test]
    fn test_building_with_lock() {
        use std::os::unix::io::AsRawFd;

        let tmp = TempDir::new().unwrap();
        let sage_dir = tmp.path().join(".sage");
        std::fs::create_dir_all(&sage_dir).unwrap();
        let lock_path = sage_dir.join(".sage.lock");
        let lock_file = std::fs::File::create(&lock_path).unwrap();

        // Hold an exclusive lock
        let fd = lock_file.as_raw_fd();
        let ret = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
        assert_eq!(ret, 0, "Failed to acquire test lock");

        assert_eq!(detect(tmp.path()), IndexState::Building);

        // Release lock
        unsafe {
            libc::flock(fd, libc::LOCK_UN);
        }
    }
}
