//! Index validation: consistency checks between meta.json, SQLite, dense index, and filesystem.

use crate::config::SemantexConfig;
use crate::index::storage::ChunkStore;
use crate::types::IndexMeta;
use anyhow::{Context, Result};
use std::path::Path;

/// Result of a single validation check.
#[derive(Debug, Clone)]
pub struct CheckResult {
    pub name: String,
    pub passed: bool,
    pub message: String,
}

/// Full validation report.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub checks: Vec<CheckResult>,
    pub passed: usize,
    pub failed: usize,
    pub warnings: usize,
}

impl ValidationReport {
    pub fn all_passed(&self) -> bool {
        self.failed == 0
    }

    pub fn summary(&self) -> String {
        format!("{}/{} checks passed", self.passed, self.checks.len())
    }
}

/// Run all validation checks for a project's index.
pub fn validate(project_path: &Path) -> Result<ValidationReport> {
    let index_dir = SemantexConfig::project_index_dir(project_path);
    let checks = vec![
        check_meta_db_consistency(&index_dir),
        check_stale_files(&index_dir, project_path),
        check_dense_index(&index_dir),
        check_sparse_index(&index_dir),
        check_graph_consistency(&index_dir),
    ];

    let passed = checks.iter().filter(|c| c.passed).count();
    let failed = checks.iter().filter(|c| !c.passed).count();

    Ok(ValidationReport {
        checks,
        passed,
        failed,
        warnings: 0,
    })
}

/// Check 1: Meta-DB consistency — compare meta.json counts with SQLite.
fn check_meta_db_consistency(index_dir: &Path) -> CheckResult {
    let name = "meta_db_consistency".to_string();

    let meta_path = index_dir.join("meta.json");
    let meta = match read_meta(&meta_path) {
        Ok(m) => m,
        Err(e) => {
            return CheckResult {
                name,
                passed: false,
                message: format!("Cannot read meta.json: {e}"),
            };
        }
    };

    // Check schema version
    if meta.schema_version != IndexMeta::CURRENT_SCHEMA_VERSION {
        return CheckResult {
            name,
            passed: false,
            message: format!(
                "Schema version mismatch: meta has {}, expected {}",
                meta.schema_version,
                IndexMeta::CURRENT_SCHEMA_VERSION
            ),
        };
    }

    let db_path = index_dir.join("chunks.db");
    let store = match ChunkStore::open_for_search(&db_path) {
        Ok(s) => s,
        Err(e) => {
            return CheckResult {
                name,
                passed: false,
                message: format!("Cannot open chunks.db: {e}"),
            };
        }
    };

    let db_file_count = store.file_count().unwrap_or(0);
    let db_chunk_count = store.chunk_count().unwrap_or(0);

    let file_match = db_file_count == meta.file_count;
    // Chunks within 5% tolerance (incremental indexing can cause minor drift)
    let chunk_tolerance = (meta.chunk_count as f64 * 0.05).max(1.0) as u64;
    let chunk_match = db_chunk_count.abs_diff(meta.chunk_count) <= chunk_tolerance;

    if file_match && chunk_match {
        CheckResult {
            name,
            passed: true,
            message: format!(
                "OK: files={db_file_count} (meta: {}), chunks={db_chunk_count} (meta: {}), schema=v{}",
                meta.file_count, meta.chunk_count, meta.schema_version
            ),
        }
    } else {
        let mut issues = Vec::new();
        if !file_match {
            issues.push(format!(
                "file count mismatch: DB={db_file_count}, meta={}",
                meta.file_count
            ));
        }
        if !chunk_match {
            issues.push(format!(
                "chunk count drift: DB={db_chunk_count}, meta={} (>{chunk_tolerance} tolerance)",
                meta.chunk_count
            ));
        }
        CheckResult {
            name,
            passed: false,
            message: issues.join("; "),
        }
    }
}

/// Check 2: Stale file detection — files deleted or modified since indexing.
fn check_stale_files(index_dir: &Path, project_path: &Path) -> CheckResult {
    let name = "stale_files".to_string();

    let db_path = index_dir.join("chunks.db");
    let store = match ChunkStore::open_for_search(&db_path) {
        Ok(s) => s,
        Err(e) => {
            return CheckResult {
                name,
                passed: false,
                message: format!("Cannot open chunks.db: {e}"),
            };
        }
    };

    let file_paths = match store.get_all_file_paths() {
        Ok(p) => p,
        Err(e) => {
            return CheckResult {
                name,
                passed: false,
                message: format!("Cannot read file paths from DB: {e}"),
            };
        }
    };

    let total_files = file_paths.len();
    let mut missing_count = 0;

    for rel_path in &file_paths {
        let abs_path = project_path.join(rel_path);
        if !abs_path.exists() {
            missing_count += 1;
        }
    }

    // Sample up to 100 files for mtime comparison
    let mtime_sample = store.get_file_mtimes_sample(100).unwrap_or_default();

    let mut stale_count = 0;
    for (rel_path, stored_mtime) in &mtime_sample {
        let abs_path = project_path.join(rel_path);
        if let Ok(metadata) = std::fs::metadata(&abs_path)
            && let Ok(fs_mtime) = metadata.modified()
        {
            let fs_mtime_secs = fs_mtime
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs() as i64);
            if fs_mtime_secs != *stored_mtime {
                stale_count += 1;
            }
        }
    }

    let passed = missing_count == 0 && stale_count == 0;
    let mut parts = vec![format!("{total_files} indexed files")];
    if missing_count > 0 {
        parts.push(format!("{missing_count} missing from disk"));
    }
    if stale_count > 0 {
        parts.push(format!(
            "{stale_count}/{} sampled files have changed mtime",
            mtime_sample.len()
        ));
    }
    if passed {
        parts.push("all present, mtime sample OK".to_string());
    }

    CheckResult {
        name,
        passed,
        message: parts.join(", "),
    }
}

/// Check 3: Dense index integrity — PLAID directory and mapping file.
fn check_dense_index(index_dir: &Path) -> CheckResult {
    let name = "dense_index".to_string();

    let plaid_dir = index_dir.join("plaid");
    let mapping_file = index_dir.join("plaid_mapping.bin");

    let plaid_exists = plaid_dir.exists() && plaid_dir.is_dir();
    let mapping_exists = mapping_file.exists();

    if !plaid_exists && !mapping_exists {
        return CheckResult {
            name,
            passed: false,
            message: "Dense index missing: no plaid/ directory or plaid_mapping.bin".to_string(),
        };
    }

    let mut issues = Vec::new();

    if !plaid_exists {
        issues.push("plaid/ directory missing");
    }
    if !mapping_exists {
        issues.push("plaid_mapping.bin missing");
    }

    if mapping_exists
        && let Ok(meta) = std::fs::metadata(&mapping_file)
        && meta.len() == 0
    {
        issues.push("plaid_mapping.bin is empty (0 bytes)");
    }

    if issues.is_empty() {
        let mapping_size = std::fs::metadata(&mapping_file).map_or(0, |m| m.len());
        CheckResult {
            name,
            passed: true,
            message: format!("OK: plaid/ directory present, mapping={mapping_size} bytes"),
        }
    } else {
        CheckResult {
            name,
            passed: false,
            message: issues.join("; "),
        }
    }
}

/// Check 4: Sparse index integrity — chunks.db existence and readability.
fn check_sparse_index(index_dir: &Path) -> CheckResult {
    let name = "sparse_index".to_string();

    let db_path = index_dir.join("chunks.db");

    if !db_path.exists() {
        return CheckResult {
            name,
            passed: false,
            message: "chunks.db not found".to_string(),
        };
    }

    match ChunkStore::open_for_search(&db_path) {
        Ok(store) => {
            let chunk_count = store.chunk_count().unwrap_or(0);
            if chunk_count == 0 {
                CheckResult {
                    name,
                    passed: false,
                    message: "chunks.db is readable but contains 0 chunks".to_string(),
                }
            } else {
                CheckResult {
                    name,
                    passed: true,
                    message: format!("OK: chunks.db readable, {chunk_count} chunks"),
                }
            }
        }
        Err(e) => CheckResult {
            name,
            passed: false,
            message: format!("chunks.db is corrupted or unreadable: {e}"),
        },
    }
}

/// Check 5: Graph consistency — symbol defs, call graph, type refs.
fn check_graph_consistency(index_dir: &Path) -> CheckResult {
    let name = "graph_consistency".to_string();

    let db_path = index_dir.join("chunks.db");
    let store = match ChunkStore::open_for_search(&db_path) {
        Ok(s) => s,
        Err(e) => {
            return CheckResult {
                name,
                passed: false,
                message: format!("Cannot open chunks.db: {e}"),
            };
        }
    };

    let chunk_count = store.chunk_count().unwrap_or(0);

    let stats = match store.graph_stats() {
        Ok(s) => s,
        Err(e) => {
            return CheckResult {
                name,
                passed: false,
                message: format!("Cannot read graph stats: {e}"),
            };
        }
    };

    // If there are chunks, we expect at least some symbol definitions
    let has_symbols = chunk_count == 0 || stats.symbol_defs_count > 0;

    CheckResult {
        name,
        passed: has_symbols,
        message: format!(
            "symbol_defs={}, calls_resolved={}, types_resolved={}, hierarchy={}, modules={}{}",
            stats.symbol_defs_count,
            stats.calls_resolved,
            stats.types_resolved,
            stats.hierarchy_resolved,
            stats.module_edges_count,
            if has_symbols {
                ""
            } else {
                " (WARNING: no symbol defs despite having chunks)"
            }
        ),
    }
}

/// Read and parse meta.json.
fn read_meta(meta_path: &Path) -> Result<IndexMeta> {
    let content = std::fs::read_to_string(meta_path)
        .with_context(|| format!("reading {}", meta_path.display()))?;
    let meta: IndexMeta = serde_json::from_str(&content)
        .with_context(|| format!("parsing {}", meta_path.display()))?;
    Ok(meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_meta(dir: &Path, file_count: u64, chunk_count: u64) {
        let meta = serde_json::json!({
            "schema_version": IndexMeta::CURRENT_SCHEMA_VERSION,
            "project_path": dir.parent().unwrap_or(dir),
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "file_count": file_count,
            "chunk_count": chunk_count,
            "embedding_model": "test",
            "embedding_dim": 48,
            // v0.4.1 W-Index #4: use_bm25_stemmer is now part of IndexMeta.
            "use_bm25_stemmer": true,
            // S1: dense_backend is now part of IndexMeta (schema v10).
            "dense_backend": "colbert-plaid",
        });
        std::fs::write(dir.join("meta.json"), meta.to_string()).unwrap();
    }

    #[test]
    fn test_validate_nonexistent_project() {
        let tmp = TempDir::new().unwrap();
        let fake_project = tmp.path().join("nonexistent");
        let report = validate(&fake_project).unwrap();
        // All checks should fail for a nonexistent project
        assert!(report.failed > 0);
        assert!(!report.all_passed());
    }

    #[test]
    fn test_check_meta_db_consistency_matching() {
        let tmp = TempDir::new().unwrap();
        let index_dir = tmp.path().join(".semantex");
        std::fs::create_dir_all(&index_dir).unwrap();

        // Create a DB with some data
        let db_path = index_dir.join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();

        // Insert a file entry
        store
            .set_file_entry(&crate::types::FileEntry {
                path: std::path::PathBuf::from("test.rs"),
                hash: 123,
                size: 100,
                mtime: 1000,
            })
            .unwrap();

        // Insert a chunk
        let chunk = crate::types::Chunk {
            id: 0,
            file_path: std::path::PathBuf::from("test.rs"),
            start_line: 1,
            end_line: 10,
            content: "fn main() {}".to_string(),
            chunk_type: crate::types::ChunkType::TextWindow { window_index: 0 },
        };
        store.insert_chunk(&chunk, 123, 1000).unwrap();

        // Write matching meta.json
        write_meta(&index_dir, 1, 1);

        let result = check_meta_db_consistency(&index_dir);
        assert!(result.passed, "Expected pass, got: {}", result.message);
    }

    #[test]
    fn test_check_meta_db_consistency_mismatch() {
        let tmp = TempDir::new().unwrap();
        let index_dir = tmp.path().join(".semantex");
        std::fs::create_dir_all(&index_dir).unwrap();

        // Create a DB with 1 file, 1 chunk
        let db_path = index_dir.join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();
        store
            .set_file_entry(&crate::types::FileEntry {
                path: std::path::PathBuf::from("test.rs"),
                hash: 123,
                size: 100,
                mtime: 1000,
            })
            .unwrap();
        let chunk = crate::types::Chunk {
            id: 0,
            file_path: std::path::PathBuf::from("test.rs"),
            start_line: 1,
            end_line: 10,
            content: "fn main() {}".to_string(),
            chunk_type: crate::types::ChunkType::TextWindow { window_index: 0 },
        };
        store.insert_chunk(&chunk, 123, 1000).unwrap();

        // Write meta.json with wrong counts
        write_meta(&index_dir, 99, 999);

        let result = check_meta_db_consistency(&index_dir);
        assert!(!result.passed, "Expected fail, got: {}", result.message);
    }

    #[test]
    fn test_check_stale_files_all_present() {
        let tmp = TempDir::new().unwrap();
        let project_dir = tmp.path();
        let index_dir = project_dir.join(".semantex");
        std::fs::create_dir_all(&index_dir).unwrap();

        // Create a real file
        let test_file = project_dir.join("test.rs");
        std::fs::write(&test_file, "fn main() {}").unwrap();
        let fs_mtime = std::fs::metadata(&test_file)
            .unwrap()
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        // Create DB with matching entry
        let db_path = index_dir.join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();
        store
            .set_file_entry(&crate::types::FileEntry {
                path: std::path::PathBuf::from("test.rs"),
                hash: 123,
                size: 100,
                mtime: fs_mtime,
            })
            .unwrap();

        let result = check_stale_files(&index_dir, project_dir);
        assert!(result.passed, "Expected pass, got: {}", result.message);
    }

    #[test]
    fn test_check_stale_files_missing() {
        let tmp = TempDir::new().unwrap();
        let project_dir = tmp.path();
        let index_dir = project_dir.join(".semantex");
        std::fs::create_dir_all(&index_dir).unwrap();

        // Create DB referencing a file that doesn't exist
        let db_path = index_dir.join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();
        store
            .set_file_entry(&crate::types::FileEntry {
                path: std::path::PathBuf::from("deleted.rs"),
                hash: 123,
                size: 100,
                mtime: 1000,
            })
            .unwrap();

        let result = check_stale_files(&index_dir, project_dir);
        assert!(!result.passed, "Expected fail, got: {}", result.message);
        assert!(result.message.contains("missing"));
    }
}
