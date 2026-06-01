//! S1 acceptance gate: the `colbert-plaid` dense channel must be byte-identical
//! before and after the DenseBackend seam refactor.
//!
//! Strategy (repo-agnostic — tempdir + inline sources, no hardcoded paths):
//!   1. Build a small synthetic multi-language repo and index it.
//!   2. For a fixed query set, run a DENSE-ONLY search and capture the ordered
//!      (chunk content-key, score) pairs from the dense channel.
//!   3. Assert determinism (two opens of the same index agree) and stability
//!      (the captured sequence equals a checked-in baseline string).
//!
//! `#[ignore]` because it builds the full PLAID index (requires the ColBERT
//! model + several seconds). Run explicitly:
//!   cargo test -p semantex-core --test dense_backend_golden_test -- --ignored --nocapture
#![allow(clippy::unwrap_used)]

use anyhow::Result;
use semantex_core::config::SemantexConfig;
use semantex_core::index::builder::IndexBuilder;
use semantex_core::search::SearchQuery;
use semantex_core::search::hybrid::HybridSearcher;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

const QUERIES: &[&str] = &[
    "recursive fibonacci function",
    "binary search over a sorted array",
    "http get request handler",
    "open a database connection",
];

fn write_repo(root: &Path) {
    fs::write(
        root.join("fib.rs"),
        "/// nth fibonacci number\npub fn fibonacci(n: u32) -> u64 {\n    if n < 2 { n as u64 } else { fibonacci(n-1) + fibonacci(n-2) }\n}\n",
    )
    .unwrap();
    fs::write(
        root.join("search.py"),
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
    )
    .unwrap();
    fs::write(
        root.join("server.js"),
        "function handleGet(req, res) {\n  if (req.method === 'GET') {\n    res.writeHead(200);\n    res.end('ok');\n  }\n}\nmodule.exports = { handleGet };\n",
    )
    .unwrap();
    fs::write(
        root.join("db.go"),
        "package db\nimport \"database/sql\"\nfunc OpenConn(dsn string) (*sql.DB, error) {\n\treturn sql.Open(\"postgres\", dsn)\n}\n",
    )
    .unwrap();
}

/// Build the index once and return an opened searcher + the index dir.
fn build_and_open(project: &Path) -> Result<HybridSearcher> {
    let config = SemantexConfig {
        rerank: false, // dense-channel test — keep rerank out of the way
        ..SemantexConfig::default()
    };
    let stats = IndexBuilder::new(&config)?.build(project)?;
    assert!(stats.chunks_created > 0, "fixture must create chunks");
    HybridSearcher::open(&project.join(".semantex"), &config)
}

/// Run a DENSE-ONLY search per query and serialize the ordered results as a
/// stable, path-relative key string: "query|relpath:start-end:score3dp; ...".
/// We key on (relative file stem, line range) — stable across machines — and
/// round scores to 3 dp so float noise below the ranking threshold doesn't
/// flap the golden string.
fn capture_dense_signature(searcher: &HybridSearcher) -> String {
    let mut out = String::new();
    for q in QUERIES {
        let query = SearchQuery::new(*q).max_results(5).dense_only();
        let result = searcher.search(&query).unwrap();
        out.push_str(q);
        out.push('|');
        for r in &result.results {
            let stem = r
                .chunk
                .file_path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            out.push_str(&format!(
                "{}:{}-{}:{:.3};",
                stem, r.chunk.start_line, r.chunk.end_line, r.score
            ));
        }
        out.push('\n');
    }
    out
}

/// Determinism: opening the SAME index twice yields identical dense rankings.
#[test]
#[ignore] // builds full PLAID index; run with --ignored
fn colbert_plaid_dense_is_deterministic() -> Result<()> {
    let tmp = TempDir::new()?;
    let project = tmp.path().join("repo");
    fs::create_dir_all(&project)?;
    write_repo(&project);

    let s1 = build_and_open(&project)?;
    let sig1 = capture_dense_signature(&s1);

    // Re-open the already-built index (no rebuild) and re-capture.
    let config = SemantexConfig {
        rerank: false,
        ..SemantexConfig::default()
    };
    let s2 = HybridSearcher::open(&project.join(".semantex"), &config)?;
    let sig2 = capture_dense_signature(&s2);

    assert_eq!(
        sig1, sig2,
        "dense rankings must be identical across two opens of the same index"
    );
    Ok(())
}

/// Stability: the dense signature equals the checked-in baseline.
///
/// To (re)generate the baseline after an INTENTIONAL change, run this test with
/// `--ignored --nocapture`, copy the printed `ACTUAL:` block into
/// `tests/fixtures/dense_backend_golden.txt`, and re-run. The baseline is
/// committed so CI catches any unintended dense-ranking drift from the seam.
#[test]
#[ignore] // builds full PLAID index; run with --ignored
fn colbert_plaid_dense_matches_baseline() -> Result<()> {
    let tmp = TempDir::new()?;
    let project = tmp.path().join("repo");
    fs::create_dir_all(&project)?;
    write_repo(&project);

    let searcher = build_and_open(&project)?;
    let actual = capture_dense_signature(&searcher);

    let baseline_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dense_backend_golden.txt");
    let baseline = fs::read_to_string(&baseline_path).unwrap_or_default();

    if baseline.trim().is_empty() {
        // First run: emit the signature so the author can seed the baseline.
        println!(
            "ACTUAL:\n{actual}\n--- (seed tests/fixtures/dense_backend_golden.txt with the above)"
        );
        panic!(
            "baseline missing/empty at {} — seed it from the ACTUAL block above",
            baseline_path.display()
        );
    }

    assert_eq!(
        actual, baseline,
        "dense ranking drifted from the committed baseline — the seam must be behavior-identical.\nACTUAL:\n{actual}"
    );
    Ok(())
}
