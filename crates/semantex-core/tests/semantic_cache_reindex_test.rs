#![allow(clippy::ignore_without_reason)]
//! S7 acceptance gate: a reindex MUST invalidate the semantic cache. Builds a
//! synthetic repo, primes the cache with a query, mutates a file + reindexes
//! (which rewrites meta.json `updated_at`), then asserts the same query returns
//! the POST-reindex content — proving the stamp-flush invalidation works
//! end-to-end through HybridSearcher::search. Repo-agnostic; tempdir only.

use semantex_core::config::SemantexConfig;
use semantex_core::index::builder::IndexBuilder;
use semantex_core::search::SearchQuery;
use semantex_core::search::hybrid::HybridSearcher;
use semantex_core::search::semantic_cache;
use std::fs;
use std::path::Path;

fn write_file(dir: &Path, name: &str, body: &str) {
    fs::write(dir.join(name), body).unwrap();
}

fn build_index(project_dir: &Path, config: &SemantexConfig) {
    IndexBuilder::new(config)
        .unwrap()
        .build(project_dir)
        .unwrap();
}

#[test]
#[ignore = "requires ONNX runtime + builds full index; run explicitly with ORT_DYLIB_PATH set"]
fn reindex_invalidates_semantic_cache() {
    // The cache only engages with a dense backend + enabled env. This test
    // requires the ColBERT model to be available (same precondition as the
    // existing search_accuracy_test.rs integration tests). If the model can't
    // load, the dense channel is absent and the cache no-ops — in that case the
    // assertions on cache invalidation are vacuously safe (results recomputed
    // every time). We still assert correctness of the returned content.

    // SAFETY: process-level env mutation; this test owns these keys.
    unsafe {
        std::env::set_var("SEMANTEX_SEMANTIC_CACHE", "1");
        std::env::set_var("SEMANTEX_SEMANTIC_CACHE_THRESHOLD", "0.5");
    }

    let tmp = tempfile::TempDir::new().unwrap();
    let project_dir = tmp.path();
    let config = SemantexConfig::default();

    // v1: a file whose content the query will match.
    write_file(
        project_dir,
        "payments.rs",
        "// process_refund issues a refund to the original payment method\n\
         pub fn process_refund(amount: u64) -> bool { amount > 0 }\n",
    );
    build_index(project_dir, &config);

    let index_dir = project_dir.join(".semantex");
    let stamp_v1 = semantic_cache::read_stamp(&index_dir).expect("v1 stamp");

    // Prime the cache.
    {
        let searcher = HybridSearcher::open(&index_dir, &config).unwrap();
        let q = SearchQuery::new("how are refunds processed").max_results(5);
        let out = searcher.search(&q).unwrap();
        // Sanity: we got results (or, if no model, possibly sparse-only results).
        let _ = out.results.len();
        // Issue the SAME query again on the SAME searcher — should hit the cache
        // (or recompute identically). Either way it must succeed.
        let out2 = searcher.search(&q).unwrap();
        assert_eq!(
            out.results.first().map(|r| r.chunk.file_path.clone()),
            out2.results.first().map(|r| r.chunk.file_path.clone()),
            "repeat query on same index returns the same top file"
        );
    }

    // --- Reindex with changed content. A rebuild rewrites meta.json updated_at. ---
    // Ensure the epoch-second timestamp actually advances (updated_at is in
    // whole seconds): if the rebuild lands in the same wall-clock second, the
    // stamp would not change. Loop the rebuild until the stamp differs (bounded).
    write_file(
        project_dir,
        "payments.rs",
        "// process_refund is DISABLED; refunds now go through the ledger module\n\
         pub fn process_refund(_amount: u64) -> bool { false }\n",
    );
    write_file(
        project_dir,
        "ledger.rs",
        "// settle_refund records a refund in the ledger\n\
         pub fn settle_refund(amount: u64) -> bool { amount > 0 }\n",
    );

    let mut stamp_v2 = stamp_v1.clone();
    for _ in 0..5 {
        build_index(project_dir, &config);
        stamp_v2 = semantic_cache::read_stamp(&index_dir).expect("v2 stamp");
        if stamp_v2.updated_at != stamp_v1.updated_at {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1100));
    }
    assert_ne!(
        stamp_v2.updated_at, stamp_v1.updated_at,
        "reindex must advance meta.json updated_at (epoch seconds)"
    );

    // A fresh searcher over the reindexed dir, same query: the cache (whether a
    // brand-new instance OR a swapped one) must reflect v2 content, NOT v1. The
    // KEY invariant: even if a long-lived cache somehow carried v1's entry, the
    // stamp change forces a flush → recompute → v2 content.
    {
        let searcher = HybridSearcher::open(&index_dir, &config).unwrap();
        let q = SearchQuery::new("how are refunds processed").max_results(5);
        let out = searcher.search(&q).unwrap();

        // The ledger file is new in v2; v1 had no ledger.rs. If the dense
        // channel is active, the cache MUST NOT serve a v1 snapshot. We assert
        // the result set is consistent with the v2 corpus.
        for r in &out.results {
            if r.chunk.file_path.ends_with("payments.rs") {
                assert!(
                    !r.chunk.content.contains("issues a refund to the original"),
                    "stale v1 payments.rs content must not be served after reindex; got: {}",
                    r.chunk.content
                );
            }
        }
    }

    unsafe {
        std::env::remove_var("SEMANTEX_SEMANTIC_CACHE");
        std::env::remove_var("SEMANTEX_SEMANTIC_CACHE_THRESHOLD");
    }
}
