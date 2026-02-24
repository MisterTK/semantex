use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use xxhash_rust::xxh64;

/// Hash file contents using xxhash64.
pub fn hash_file(path: &Path) -> Result<u64> {
    let data =
        fs::read(path).with_context(|| format!("Failed to read file: {}", path.display()))?;
    Ok(hash_bytes(&data))
}

/// Hash bytes directly using xxhash64.
pub fn hash_bytes(data: &[u8]) -> u64 {
    xxh64::xxh64(data, 0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_bytes_deterministic() {
        let data = b"hello world";
        let h1 = hash_bytes(data);
        let h2 = hash_bytes(data);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_bytes_different_inputs() {
        let h1 = hash_bytes(b"hello");
        let h2 = hash_bytes(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, b"test content").unwrap();
        let hash = hash_file(&file_path).unwrap();
        assert_eq!(hash, hash_bytes(b"test content"));
    }
}
