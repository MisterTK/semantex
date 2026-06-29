//! Phase-0 formatter regression net.
//!
//! These snapshot tests capture the **formatted output** of every major
//! formatter entry point in `agent_formatter.rs` using fully synthetic,
//! deterministic inputs (no live index, no daemon, no file I/O).
//!
//! ## Purpose
//!
//! The upcoming route-simplification work (Phase 1+) will rename and merge
//! `AgentRoute` variants. The `[route: …]` prefix that `agent.rs:212`
//! prepends to `AgentResponse.formatted` changes whenever a route is
//! renamed. This net snapshots the formatters **below** that prefix — i.e.
//! the pure formatter output without the `[route: …]` line — so that:
//!
//! - Behavior-preserving merges (same formatter, different route name) do
//!   **not** produce a snapshot delta.
//! - Behavior-changing merges (different formatter output) produce a
//!   **reviewable delta** that reviewers must explicitly accept.
//!
//! ## Normalisation decision
//!
//! The `[route: …]` line is generated in `agent.rs`, not in the formatter
//! functions. Because these tests call the formatter functions directly,
//! the prefix is **absent from all snapshots by construction**. No
//! stripping is required.
//!
//! ## How to review an intended delta
//!
//! After a deliberate format change, run:
//!
//! ```text
//! cargo insta test -p semantex-core --test formatter_snapshot_test
//! cargo insta review
//! ```
//!
//! Accept the new snapshots with `a`, skip with `s`, or reject with `r`.
//! Committed `.snap` files in `tests/snapshots/` are the ground truth.

use semantex_core::search::agent_formatter::{
    DEFAULT_BUDGET, FormatStyle, append_disambiguation_block, format_code_blocks,
    format_deep_results, format_graph_results, format_search_results,
};
use semantex_core::server::protocol::{
    DeepResponseMetrics, DeepSearchResponse, DeepSearchSource, DisambigSuggestion,
    GraphWalkResponse, SearchResponse, SearchResultItem,
};

// ── helpers ──────────────────────────────────────────────────────────────────

// Test fixture builder: a flat positional arg list keeps call sites compact.
#[allow(clippy::too_many_arguments)]
fn make_item(
    file: &str,
    start: u32,
    end: u32,
    name: Option<&str>,
    kind: Option<&str>,
    score: f32,
    content: Option<&str>,
    language: Option<&str>,
) -> SearchResultItem {
    SearchResultItem {
        file: file.into(),
        start_line: start,
        end_line: end,
        score,
        source: "Dense".into(),
        chunk_type: "AstNode".into(),
        name: name.map(Into::into),
        language: language.map(Into::into),
        content: content.map(Into::into),
        kind: kind.map(Into::into),
        summary: None,
    }
}

fn search_response(results: Vec<SearchResultItem>, ms: u64, confidence: &str) -> SearchResponse {
    SearchResponse {
        results,
        duration_ms: ms,
        dense_count: 3,
        sparse_count: 2,
        fused_count: 4,
        metrics: None,
        confidence: Some(confidence.into()),
        disambiguation: None,
    }
}

// ── format_search_results — Default style ────────────────────────────────────

#[test]
fn snap_search_default_single_result() {
    let resp = search_response(
        vec![make_item(
            "src/auth.rs",
            10,
            25,
            Some("AuthMiddleware"),
            Some("struct"),
            0.92,
            Some("pub struct AuthMiddleware { token: String }"),
            Some("rust"),
        )],
        18,
        "high",
    );
    let out = format_search_results(&resp, FormatStyle::Default, DEFAULT_BUDGET);
    insta::assert_snapshot!(out);
}

#[test]
fn snap_search_default_multiple_results() {
    let resp = search_response(
        vec![
            make_item(
                "src/auth.rs",
                10,
                25,
                Some("AuthMiddleware"),
                Some("struct"),
                0.92,
                Some("pub struct AuthMiddleware { token: String }"),
                Some("rust"),
            ),
            make_item(
                "src/handler.rs",
                42,
                67,
                Some("handle_request"),
                Some("fn"),
                0.77,
                Some("pub fn handle_request(req: Request) -> Response { todo!() }"),
                Some("rust"),
            ),
        ],
        22,
        "medium",
    );
    let out = format_search_results(&resp, FormatStyle::Default, DEFAULT_BUDGET);
    insta::assert_snapshot!(out);
}

#[test]
fn snap_search_default_empty() {
    let resp = search_response(vec![], 1, "none");
    let out = format_search_results(&resp, FormatStyle::Default, DEFAULT_BUDGET);
    insta::assert_snapshot!(out);
}

#[test]
fn snap_search_default_budget_truncation() {
    // Use a tiny budget so only the first result fits and the "… more" line appears.
    // Casts are on small, known-non-negative loop indices (0..5), so sign/precision
    // loss cannot occur for these fixture values.
    #[allow(clippy::cast_sign_loss, clippy::cast_precision_loss)]
    let results: Vec<SearchResultItem> = (0..5)
        .map(|i| {
            make_item(
                &format!("src/module{i}.rs"),
                (i * 10 + 1) as u32,
                (i * 10 + 20) as u32,
                Some(&format!("Symbol{i}")),
                Some("fn"),
                0.9 - (i as f32 * 0.05),
                Some(&format!("fn symbol_{i}() {{ /* impl */ }}")),
                Some("rust"),
            )
        })
        .collect();
    let resp = search_response(results, 30, "high");
    let out = format_search_results(&resp, FormatStyle::Default, 300);
    insta::assert_snapshot!(out);
}

// ── format_search_results — Grep style ───────────────────────────────────────

#[test]
fn snap_search_grep_style() {
    let resp = search_response(
        vec![
            make_item(
                "src/auth.rs",
                42,
                42,
                None,
                None,
                0.5,
                Some("fn authenticate(token: &str) -> bool {"),
                None,
            ),
            make_item(
                "src/session.rs",
                17,
                17,
                None,
                None,
                0.4,
                Some("pub fn session_start() {"),
                None,
            ),
        ],
        5,
        "medium",
    );
    let out = format_search_results(&resp, FormatStyle::Grep, DEFAULT_BUDGET);
    insta::assert_snapshot!(out);
}

// ── format_deep_results ───────────────────────────────────────────────────────

#[test]
fn snap_deep_with_answer_and_sources() {
    let resp = DeepSearchResponse {
        answer: "Authentication uses JWT tokens validated via the `AuthMiddleware` struct \
                 in `src/auth.rs`. The middleware chain calls `handle_request` which \
                 delegates to the token validator."
            .into(),
        sources: vec![
            DeepSearchSource {
                file: "src/auth.rs".into(),
                start_line: 10,
                end_line: 50,
                name: Some("AuthMiddleware".into()),
                kind: Some("struct".into()),
            },
            DeepSearchSource {
                file: "src/handler.rs".into(),
                start_line: 42,
                end_line: 80,
                name: Some("handle_request".into()),
                kind: Some("fn".into()),
            },
        ],
        metrics: DeepResponseMetrics {
            search_ms: 15,
            triage_ms: 3,
            graph_ms: 4,
            read_ms: 8,
            summarize_ms: 12,
            total_ms: 42,
            chunks_searched: 25,
            chunks_read: 6,
            confidence_zone: "high".into(),
        },
        confidence: 0.88,
    };
    let out = format_deep_results(&resp, DEFAULT_BUDGET);
    insta::assert_snapshot!(out);
}

#[test]
fn snap_deep_empty() {
    let resp = DeepSearchResponse {
        answer: String::new(),
        sources: vec![],
        metrics: DeepResponseMetrics::default(),
        confidence: 0.0,
    };
    let out = format_deep_results(&resp, DEFAULT_BUDGET);
    insta::assert_snapshot!(out);
}

#[test]
fn snap_deep_budget_truncation() {
    let long_answer = "The system uses ".to_string() + &"a complex algorithm ".repeat(500);
    let resp = DeepSearchResponse {
        answer: long_answer,
        sources: vec![],
        metrics: DeepResponseMetrics {
            total_ms: 100,
            chunks_searched: 50,
            chunks_read: 10,
            ..Default::default()
        },
        confidence: 0.5,
    };
    let out = format_deep_results(&resp, 200);
    insta::assert_snapshot!(out);
}

// ── format_graph_results ──────────────────────────────────────────────────────

#[test]
fn snap_graph_with_callers_and_callees() {
    let resp = GraphWalkResponse {
        target: vec![make_item(
            "src/auth.rs",
            10,
            50,
            Some("AuthMiddleware"),
            Some("struct"),
            1.0,
            None,
            Some("rust"),
        )],
        callers: vec![
            make_item(
                "src/middleware.rs",
                5,
                20,
                Some("apply_middleware"),
                Some("fn"),
                0.9,
                None,
                Some("rust"),
            ),
            make_item(
                "src/server.rs",
                100,
                115,
                Some("start_server"),
                Some("fn"),
                0.8,
                None,
                Some("rust"),
            ),
        ],
        callees: vec![make_item(
            "src/token.rs",
            1,
            30,
            Some("validate_token"),
            Some("fn"),
            0.95,
            None,
            Some("rust"),
        )],
        type_refs: vec![],
        hierarchy: vec![],
    };
    let out = format_graph_results(&resp);
    insta::assert_snapshot!(out);
}

#[test]
fn snap_graph_all_empty() {
    let resp = GraphWalkResponse {
        target: vec![],
        callers: vec![],
        callees: vec![],
        type_refs: vec![],
        hierarchy: vec![],
    };
    let out = format_graph_results(&resp);
    insta::assert_snapshot!(out);
}

// ── format_code_blocks ────────────────────────────────────────────────────────

#[test]
fn snap_code_blocks_single() {
    let results = vec![make_item(
        "src/auth.rs",
        10,
        14,
        Some("AuthMiddleware"),
        Some("struct"),
        0.9,
        None,
        Some("rust"),
    )];
    let code = vec!["pub struct AuthMiddleware {\n    token: String,\n    ttl: u64,\n}\n".into()];
    let out = format_code_blocks(&results, &code, DEFAULT_BUDGET);
    insta::assert_snapshot!(out);
}

#[test]
fn snap_code_blocks_empty_content() {
    let results = vec![make_item(
        "src/auth.rs",
        10,
        14,
        None,
        None,
        0.9,
        None,
        Some("rust"),
    )];
    let code = vec![String::new()];
    let out = format_code_blocks(&results, &code, DEFAULT_BUDGET);
    insta::assert_snapshot!(out);
}

// ── append_disambiguation_block ───────────────────────────────────────────────

#[test]
fn snap_disambiguation_three_suggestions() {
    let mut base = "search results here".to_string();
    let suggestions = vec![
        DisambigSuggestion {
            name: "userAuthHandler".into(),
            path: "auth/users.rs".into(),
            line: 42,
        },
        DisambigSuggestion {
            name: "tokenAuthHandler".into(),
            path: "auth/tokens.rs".into(),
            line: 18,
        },
        DisambigSuggestion {
            name: "sessionAuth".into(),
            path: "sessions/handler.rs".into(),
            line: 107,
        },
    ];
    append_disambiguation_block(&mut base, &suggestions);
    insta::assert_snapshot!(base);
}

#[test]
fn snap_disambiguation_empty() {
    let mut base = "existing output".to_string();
    append_disambiguation_block(&mut base, &[]);
    insta::assert_snapshot!(base);
}
