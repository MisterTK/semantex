//! Cross-repo `--scope` support shared by the plain CLI search and `semantex
//! agent`. (v13 Wave 2 §B/§F: "cross-repo federation in semantex_agent
//! (`scope` param)".)
//!
//! `--scope repo` (the default) never reaches this module — `main.rs` keeps
//! calling `commands::search::run` / the daemon-backed `Agent` path exactly
//! as before, so single-repo behavior is completely unaffected. Any other
//! scope runs entirely IN-PROCESS (no daemon): it resolves targets via the
//! registry (`semantex_core::search::federation::resolve_targets`), opens
//! each target's index SEQUENTIALLY with
//! [`semantex_core::search::federation::SequentialIndexSearcher`] — open,
//! search, drop, next — so a CLI process (which has no long-lived cache to
//! amortize the open cost across calls, unlike the MCP server's LRU searcher
//! cache) never holds more than one dense index resident at a time, and
//! fuses the per-target hits with rank-based RRF
//! (`federation::run_federated_search` → `fuse_targets`, k=60).
//!
//! `semantex agent --scope <non-repo>` deliberately skips the single-repo
//! route classifier (structural walks / deep synthesis / architecture are
//! inherently single-repo operations — see `run_agent`'s doc comment) and
//! always performs a hybrid dense+sparse search, matching
//! `semantex-mcp`'s `tool_agent_federated`.

use anyhow::Result;
use colored::Colorize;
use semantex_core::config::SemantexConfig;
use semantex_core::index::registry::{self, RegistryV2};
use semantex_core::search::SearchQuery;
use semantex_core::search::agent_formatter::{DEFAULT_BUDGET, FormatStyle, format_search_results};
use semantex_core::search::federation::{
    self, FederatedHit, SearchScope, SequentialIndexSearcher, SkippedTarget, project_display_name,
};
use semantex_core::server::protocol::{SearchResponse, SearchResultItem};
use semantex_core::types::ChunkType;
use std::fmt::Write as _;
use std::path::Path;

/// Parse a `--scope` CLI value into a [`SearchScope`]. `"repo"` (the
/// default) and any unrecognized value map to `CurrentRepo`; `"all"` maps to
/// `All`; anything else is split on commas into project display
/// names/paths (`Named`).
pub fn parse_scope(raw: &str) -> SearchScope {
    match raw {
        "repo" | "" => SearchScope::CurrentRepo,
        "all" => SearchScope::All,
        other => {
            let names: Vec<String> = other
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
            if names.is_empty() {
                SearchScope::CurrentRepo
            } else {
                SearchScope::Named(names)
            }
        }
    }
}

/// Convert a federated hit into a plain [`SearchResultItem`] with `file` left
/// bare (repo-relative, un-prefixed) — callers add the `[project] ` prefix
/// themselves for human-formatted text, and carry `project` as a separate
/// field for JSON.
fn item_from_hit(hit: &FederatedHit) -> SearchResultItem {
    let (name, language) = match &hit.result.chunk.chunk_type {
        ChunkType::AstNode { name, language, .. } => (Some(name.clone()), Some(language.clone())),
        _ => (None, None),
    };
    SearchResultItem {
        file: hit.result.chunk.file_path.display().to_string(),
        start_line: hit.result.chunk.start_line,
        end_line: hit.result.chunk.end_line,
        score: hit.result.score,
        source: format!("{:?}", hit.result.source),
        chunk_type: format!("{:?}", hit.result.chunk.chunk_type),
        name,
        language,
        content: Some(hit.result.chunk.content.clone()),
        kind: None,
        summary: None,
    }
}

fn print_skipped(registry: &RegistryV2, skipped: &[SkippedTarget]) {
    if skipped.is_empty() {
        return;
    }
    let mut names: Vec<String> = skipped
        .iter()
        .map(|s| {
            format!(
                "{} ({})",
                project_display_name(registry, &s.target.project_root),
                s.reason
            )
        })
        .collect();
    names.sort();
    names.dedup();
    eprintln!("[skipped: {}]", names.join(", "));
}

/// `--scope` handling for the plain CLI search command (the top-level `Cli`
/// default/no-subcommand path in `main.rs`). Prints results directly, like
/// `commands::search::run` does for the single-repo case.
#[allow(clippy::too_many_arguments)]
pub fn run_search(
    scope: &SearchScope,
    project_path: &Path,
    query: &str,
    max_results: usize,
    rerank: bool,
    grep_mode: bool,
    json: bool,
    config: &SemantexConfig,
) -> Result<()> {
    let registry_v2 = registry::read_all_v2();

    let build_query = move |q: &str, limit: usize| -> SearchQuery {
        if grep_mode {
            SearchQuery::new(q).grep_mode().max_results(limit)
        } else if rerank {
            SearchQuery::new(q).max_results(limit)
        } else {
            SearchQuery::new(q).max_results(limit).no_rerank()
        }
    };
    let searcher = SequentialIndexSearcher {
        config,
        build_query: Box::new(build_query),
    };

    let outcome = federation::run_federated_search(
        scope,
        &registry_v2,
        project_path,
        query,
        max_results,
        &searcher,
    );

    if outcome.hits.is_empty() {
        if json {
            println!("[]");
        } else {
            eprintln!("No results found.");
        }
        print_skipped(&registry_v2, &outcome.skipped);
        return Ok(());
    }

    if json {
        let arr: Vec<serde_json::Value> = outcome
            .hits
            .iter()
            .map(|h| {
                let project = project_display_name(&registry_v2, &h.target.project_root);
                serde_json::json!({
                    "file": h.result.chunk.file_path.display().to_string(),
                    "lines": format!("{}-{}", h.result.chunk.start_line, h.result.chunk.end_line),
                    "score": h.result.score,
                    "project": project,
                    "content": h.result.chunk.content,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr)?);
    } else {
        for h in &outcome.hits {
            let project = project_display_name(&registry_v2, &h.target.project_root);
            println!(
                "[{}] {}:{}-{} {}",
                project.cyan(),
                h.result.chunk.file_path.display().to_string().green(),
                h.result.chunk.start_line,
                h.result.chunk.end_line,
                format!("{:.4}", h.result.score).yellow(),
            );
            let preview: String = h
                .result
                .chunk
                .content
                .lines()
                .take(2)
                .collect::<Vec<_>>()
                .join(" / ");
            if !preview.trim().is_empty() {
                println!("  {}", preview.trim());
            }
        }
        eprintln!(
            "\n{} results [scope: cross-repo, RRF-fused]",
            outcome.hits.len()
        );
    }
    print_skipped(&registry_v2, &outcome.skipped);
    Ok(())
}

/// `--scope` handling for `semantex agent`. Runs a hybrid dense+sparse
/// search across every resolved target (bypassing the single-repo route
/// classifier — structural/deep/architecture routes assume one repo's call
/// graph, which doesn't generalize across repos), fuses with RRF, and
/// formats the fused hits with the SAME [`format_search_results`] helper the
/// single-repo routes use — federated answers read like any other
/// `semantex agent` answer, just with `[project]`-prefixed paths.
///
/// Returns `(formatted_text, hits_json)` — `hits_json` carries a clean
/// (un-prefixed) `file` plus a sibling `project` field, mirroring
/// `semantex-mcp`'s `tool_agent_federated` structuredContent shape.
pub fn run_agent(
    scope: &SearchScope,
    project_path: &Path,
    query: &str,
    budget: usize,
    config: &SemantexConfig,
) -> (String, Vec<serde_json::Value>) {
    let registry_v2 = registry::read_all_v2();
    let searcher = SequentialIndexSearcher {
        config,
        build_query: Box::new(|q: &str, limit: usize| SearchQuery::new(q).max_results(limit)),
    };

    let outcome =
        federation::run_federated_search(scope, &registry_v2, project_path, query, 20, &searcher);

    let items: Vec<SearchResultItem> = outcome
        .hits
        .iter()
        .map(|h| {
            let project = project_display_name(&registry_v2, &h.target.project_root);
            let mut item = item_from_hit(h);
            item.file = format!("[{project}] {}", item.file);
            item
        })
        .collect();

    let hits_json: Vec<serde_json::Value> = outcome
        .hits
        .iter()
        .map(|h| {
            let project = project_display_name(&registry_v2, &h.target.project_root);
            let item = item_from_hit(h);
            serde_json::json!({
                "file": item.file,
                "lines": format!("{}-{}", item.start_line, item.end_line),
                "score": item.score,
                "name": item.name,
                "lang": item.language,
                "project": project,
            })
        })
        .collect();

    let resp = SearchResponse {
        results: items,
        duration_ms: 0,
        dense_count: 0,
        sparse_count: 0,
        fused_count: outcome.hits.len(),
        metrics: None,
        confidence: None,
        disambiguation: None,
    };
    let effective_budget = if budget == 0 { DEFAULT_BUDGET } else { budget };
    let mut formatted = format_search_results(&resp, FormatStyle::Default, effective_budget);

    if !outcome.skipped.is_empty() {
        let mut names: Vec<String> = outcome
            .skipped
            .iter()
            .map(|s| project_display_name(&registry_v2, &s.target.project_root))
            .collect();
        names.sort();
        names.dedup();
        let _ = write!(
            formatted,
            "\n\n[federation: skipped {} project(s) with no ready index: {}]",
            names.len(),
            names.join(", ")
        );
    }

    (formatted, hits_json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_scope_default_and_all() {
        assert_eq!(parse_scope("repo"), SearchScope::CurrentRepo);
        assert_eq!(parse_scope(""), SearchScope::CurrentRepo);
        assert_eq!(parse_scope("all"), SearchScope::All);
    }

    #[test]
    fn parse_scope_comma_separated_names() {
        assert_eq!(
            parse_scope("frontend,backend"),
            SearchScope::Named(vec!["frontend".to_string(), "backend".to_string()])
        );
        assert_eq!(
            parse_scope(" frontend , backend "),
            SearchScope::Named(vec!["frontend".to_string(), "backend".to_string()])
        );
    }

    #[test]
    fn parse_scope_single_name() {
        assert_eq!(
            parse_scope("my-project"),
            SearchScope::Named(vec!["my-project".to_string()])
        );
    }
}
