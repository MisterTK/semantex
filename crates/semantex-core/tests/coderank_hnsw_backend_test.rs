//! S2 acceptance gate: the `coderank-hnsw` dense backend indexes a synthetic
//! repo, searches end-to-end via HybridSearcher, and returns deterministic,
//! relevant dense results. Repo-agnostic (tempdir + inline sources). `#[ignore]`
//! because it downloads + runs the CodeRankEmbed ONNX model.
//!
//! Run: cargo test -p semantex-core --test coderank_hnsw_backend_test -- --ignored --nocapture
#![allow(clippy::unwrap_used)]

use anyhow::Result;
use semantex_core::config::SemantexConfig;
use semantex_core::index::builder::IndexBuilder;
use semantex_core::search::SearchQuery;
use semantex_core::search::hybrid::HybridSearcher;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

fn write_repo(root: &Path) {
    fs::write(
        root.join("fib.rs"),
        "/// nth fibonacci number\npub fn fibonacci(n: u32) -> u64 {\n    if n < 2 { n as u64 } else { fibonacci(n-1) + fibonacci(n-2) }\n}\n",
    )
    .unwrap();
    fs::write(
        root.join("search.py"),
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
    )
    .unwrap();
    fs::write(
        root.join("db.go"),
        "package db\nimport \"database/sql\"\nfunc OpenConn(dsn string) (*sql.DB, error) {\n\treturn sql.Open(\"postgres\", dsn)\n}\n",
    )
    .unwrap();
}

/// Select the coderank-hnsw backend via the canonical embedder id, exactly like
/// the S0 harness's `SEMANTEX_EMBEDDER=coderank-137m` A/B.
fn cfg() -> SemantexConfig {
    SemantexConfig {
        embedder: "coderank-137m".to_string(),
        rerank: false,
        ..SemantexConfig::default()
    }
}

fn build_and_open(project: &Path) -> Result<HybridSearcher> {
    let stats = IndexBuilder::new(&cfg())?.build(project)?;
    assert!(stats.chunks_created > 0, "fixture must create chunks");
    HybridSearcher::open(&project.join(".semantex"), &cfg())
}

/// Dense-only search returns the topically-correct chunk first.
#[test]
#[ignore]
fn coderank_hnsw_dense_search_is_relevant() -> Result<()> {
    let tmp = TempDir::new()?;
    let project = tmp.path().join("repo");
    fs::create_dir_all(&project)?;
    write_repo(&project);
    let searcher = build_and_open(&project)?;

    let q = SearchQuery::new("recursive fibonacci function")
        .max_results(3)
        .dense_only();
    let res = searcher.search(&q)?;
    assert!(!res.results.is_empty(), "dense search returned nothing");
    let top = res.results[0]
        .chunk
        .file_path
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    assert_eq!(
        top, "fib.rs",
        "fibonacci query should rank fib.rs first, got {top}"
    );
    Ok(())
}

/// Determinism: two opens of the same index give identical dense rankings.
#[test]
#[ignore]
fn coderank_hnsw_dense_is_deterministic() -> Result<()> {
    let tmp = TempDir::new()?;
    let project = tmp.path().join("repo");
    fs::create_dir_all(&project)?;
    write_repo(&project);

    let s1 = build_and_open(&project)?;
    let s2 = HybridSearcher::open(&project.join(".semantex"), &cfg())?;

    let q = SearchQuery::new("open a database connection")
        .max_results(3)
        .dense_only();
    let sig = |s: &HybridSearcher| -> Vec<(String, u32)> {
        s.search(&q)
            .unwrap()
            .results
            .iter()
            .map(|r| {
                (
                    r.chunk
                        .file_path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                    r.chunk.start_line,
                )
            })
            .collect()
    };
    assert_eq!(
        sig(&s1),
        sig(&s2),
        "dense ranking must be identical across opens"
    );
    Ok(())
}

/// The deprecated `SEMANTEX_DENSE_BACKEND=coderank-hnsw` alias also selects the
/// backend (the S0 harness's kept-live alias knob).
#[test]
#[ignore]
fn coderank_hnsw_alias_selection_builds_and_searches() -> Result<()> {
    let tmp = TempDir::new()?;
    let project = tmp.path().join("repo");
    fs::create_dir_all(&project)?;
    write_repo(&project);

    let alias_cfg = SemantexConfig {
        dense_backend: "coderank-hnsw".to_string(),
        rerank: false,
        ..SemantexConfig::default()
    };
    let stats = IndexBuilder::new(&alias_cfg)?.build(&project)?;
    assert!(stats.chunks_created > 0);
    // The dense index must be written under the coderank-hnsw subdir.
    assert!(
        project
            .join(".semantex/dense/coderank-hnsw/vectors.bin")
            .exists(),
        "alias selection must build the coderank-hnsw dense index"
    );
    let searcher = HybridSearcher::open(&project.join(".semantex"), &alias_cfg)?;
    let q = SearchQuery::new("binary search algorithm")
        .max_results(3)
        .dense_only();
    let res = searcher.search(&q)?;
    assert!(!res.results.is_empty(), "alias dense search returned nothing");
    Ok(())
}
