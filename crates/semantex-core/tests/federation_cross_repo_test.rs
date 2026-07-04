//! Cross-repo federation integration test (v13 Wave 2 §B/§F).
//!
//! Builds two REAL, independently-indexed tempdir projects, registers three
//! projects (the two indexed ones plus a third that's registered but never
//! indexed) in a tempdir registry (`SEMANTEX_HOME` points at a fresh tempdir
//! for the duration of this test), then exercises
//! `search::federation::run_federated_search` end to end:
//!   - `SearchScope::All` returns RRF-fused hits carrying provenance from
//!     BOTH indexed projects, sorted by fused score, while the un-indexed
//!     third project is skipped (not errored) with a "not indexed" reason.
//!   - `SearchScope::Named([..])` restricts the fan-out to just the named
//!     project.
//!
//! `#[ignore]` like this crate's other full-index-build integration tests
//! (e.g. `reranker_off_by_default_test.rs`) — building a real index downloads
//! the dense embedder, so this is not part of the default `cargo test --all`
//! gate. Run explicitly with `cargo test --test federation_cross_repo_test -- --ignored`.

use semantex_core::config::SemantexConfig;
use semantex_core::index::builder::IndexBuilder;
use semantex_core::index::registry;
use semantex_core::search::SearchQuery;
use semantex_core::search::federation::{self, SearchScope, SequentialIndexSearcher};
use std::collections::HashSet;
use std::path::PathBuf;

fn build_indexed_project(files: &[(&str, &str)], config: &SemantexConfig) -> tempfile::TempDir {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    for (name, content) in files {
        std::fs::write(tmp.path().join(name), content).expect("write fixture file");
    }
    IndexBuilder::new(config)
        .expect("builder")
        .build(tmp.path())
        .expect("index build");
    tmp
}

#[test]
#[ignore = "slow: builds two full indexes (downloads the dense embedder)"]
#[allow(clippy::too_many_lines)] // one linear scenario walkthrough reads better unsplit
fn cross_repo_federation_scope_all_named_and_missing_index_skip() {
    // SAFETY: this integration test file compiles to its own `cargo test`
    // binary/process (one per `tests/*.rs` file) and contains a single
    // `#[test]` fn, so mutating SEMANTEX_HOME here cannot race any other
    // test — nothing else runs concurrently in this process.
    let home = tempfile::TempDir::new().expect("home tempdir");
    unsafe {
        std::env::set_var("SEMANTEX_HOME", home.path());
    }

    let config = SemantexConfig::default();

    // Both projects define a `handle_request_*` function, so a query on
    // "handle request" gets a strong BM25 match in EACH repo independently
    // of dense/embedding behavior — the test doesn't need to assume
    // anything about embedding-model semantics, just that both are indexed
    // and searchable.
    let proj_a = build_indexed_project(
        &[(
            "auth.rs",
            "fn handle_request_auth(req: &str) -> bool { validate_token(req) }\n\
             fn validate_token(req: &str) -> bool { !req.is_empty() }\n",
        )],
        &config,
    );
    let proj_b = build_indexed_project(
        &[(
            "payment.rs",
            "fn handle_request_payment(req: &str) -> bool { validate_amount(req) }\n\
             fn validate_amount(req: &str) -> bool { !req.is_empty() }\n",
        )],
        &config,
    );
    let proj_c_unindexed = tempfile::TempDir::new().expect("tempdir c");
    std::fs::write(proj_c_unindexed.path().join("x.rs"), "fn x() {}\n").expect("write x.rs");

    let proj_a_path = proj_a.path().canonicalize().expect("canonicalize a");
    let proj_b_path = proj_b.path().canonicalize().expect("canonicalize b");
    let proj_c_path = proj_c_unindexed
        .path()
        .canonicalize()
        .expect("canonicalize c");

    // Register all three in the tempdir registry that SEMANTEX_HOME now points at.
    registry::register(&proj_a_path);
    registry::register(&proj_b_path);
    registry::register(&proj_c_path);

    let registry_v2 = registry::read_all_v2();
    assert_eq!(
        registry_v2.projects.len(),
        3,
        "all three projects registered"
    );

    let searcher = SequentialIndexSearcher {
        config: &config,
        build_query: Box::new(|q: &str, limit: usize| {
            SearchQuery::new(q).max_results(limit).no_rerank()
        }),
    };

    // ── SearchScope::All ────────────────────────────────────────────────
    let outcome = federation::run_federated_search(
        &SearchScope::All,
        &registry_v2,
        &proj_a_path,
        "handle request",
        10,
        &searcher,
    );

    assert!(
        !outcome.hits.is_empty(),
        "expected hits from the federated query"
    );

    let projects_hit: HashSet<PathBuf> = outcome
        .hits
        .iter()
        .map(|h| h.target.project_root.clone())
        .collect();
    assert!(
        projects_hit.contains(&proj_a_path),
        "expected at least one hit from project A; got targets: {projects_hit:?}"
    );
    assert!(
        projects_hit.contains(&proj_b_path),
        "expected at least one hit from project B; got targets: {projects_hit:?}"
    );

    // Fused ordering: RRF scores must be sorted descending.
    for w in outcome.hits.windows(2) {
        assert!(
            w[0].result.score >= w[1].result.score,
            "fused hits must be sorted by descending score"
        );
    }

    // The un-indexed third project is SKIPPED, not an error, and reported
    // in the response metadata with an actionable reason.
    assert_eq!(
        outcome.skipped.len(),
        1,
        "the un-indexed project must be skipped exactly once"
    );
    assert_eq!(outcome.skipped[0].target.project_root, proj_c_path);
    assert_eq!(outcome.skipped[0].reason, "not indexed");

    // ── SearchScope::Named([a]) ─────────────────────────────────────────
    let display_a = registry_v2
        .projects
        .iter()
        .find(|p| p.path == proj_a_path)
        .expect("project a registered")
        .display_name
        .clone();

    let outcome_named = federation::run_federated_search(
        &SearchScope::Named(vec![display_a]),
        &registry_v2,
        &proj_a_path,
        "handle request",
        10,
        &searcher,
    );

    assert!(
        !outcome_named.hits.is_empty(),
        "named scope must still return hits from the matched project"
    );
    assert!(
        outcome_named
            .hits
            .iter()
            .all(|h| h.target.project_root == proj_a_path),
        "named scope must restrict hits to ONLY the named project"
    );
    assert!(
        outcome_named.skipped.is_empty(),
        "named scope resolves to a single target it doesn't touch project C or B at all"
    );
}
