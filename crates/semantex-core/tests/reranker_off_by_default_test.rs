//! Regression guard for the S3 off-by-default safety contract: with
//! `SEMANTEX_RERANKER` unset, building + searching an index must NOT construct a
//! reranker, NOT download weights, and return results in the same order the
//! fusion produced (rerank stage is a no-op). Repo-agnostic: synthetic tempdir.
//!
//! `#[ignore]` because it builds a full index (downloads the dense embedder),
//! matching the other index-building integration tests in this crate. The
//! engine-level off-by-default contract is covered by the always-run unit tests
//! `search::onnx_reranker::tests::rerank_is_identity_when_disabled` and
//! `search::reranker_engine::tests::{new_default,from_config}_refuses_when_disabled`.

use semantex_core::config::SemantexConfig;
use semantex_core::index::builder::IndexBuilder;
use semantex_core::search::SearchQuery;
use semantex_core::search::hybrid::HybridSearcher;
use semantex_core::types::SearchSource;

#[test]
#[ignore = "slow: builds a full index (downloads the dense embedder weights)"]
fn rerank_off_by_default_produces_no_reranked_source() {
    // Ensure the master switch is unset for this process.
    // SAFETY: single-threaded test; no other thread reads this var here.
    unsafe {
        std::env::remove_var("SEMANTEX_RERANKER");
    }

    let tmp = tempfile::TempDir::new().expect("tempdir");
    let project = tmp.path();
    std::fs::write(
        project.join("a.rs"),
        "fn binary_search(arr: &[i32], target: i32) -> Option<usize> { todo!() }\n",
    )
    .unwrap();
    std::fs::write(
        project.join("b.rs"),
        "fn quicksort(arr: &mut [i32]) { /* partition and recurse */ }\n",
    )
    .unwrap();

    // Even with config.rerank on, the env gate keeps it off.
    let config = SemantexConfig {
        rerank: true,
        ..SemantexConfig::default()
    };

    IndexBuilder::new(&config)
        .expect("builder")
        .build(project)
        .expect("index build");

    let searcher = HybridSearcher::open(&project.join(".semantex"), &config).expect("open");
    let query = SearchQuery::new("binary search");
    let out = searcher.search(&query).expect("search");

    // No result may carry the Reranked source — the stage was a no-op.
    assert!(
        out.results
            .iter()
            .all(|r| r.source != SearchSource::Reranked),
        "rerank must be a no-op when SEMANTEX_RERANKER is unset"
    );
}
