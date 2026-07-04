//! `semantex_docs_context` — the deterministic half of the docs workflow.
//!
//! Wave 0 contract §F / maintainer decision: semantex ships **zero LLM
//! wiring**. This tool never writes prose. It calls into
//! `semantex_core::index::docs_scaffold` to build a structurally-complete
//! JSON scaffold (symbol inventory, call-graph edges, import edges, existing
//! doc-comment text, file:line provenance) and renders a compact
//! human-readable text alongside it. The *user's own* agent (Claude Code,
//! etc.) — guided by the `semantex-docs` Skill
//! (`plugin/skills/semantex-docs/SKILL.md`) — turns that scaffold into
//! maintained markdown under `semantex_docs/` in the user's repo.
//!
//! Kept in its own module (rather than inline in `server.rs`) so the tool's
//! param parsing / rendering logic doesn't grow the already-large
//! `server.rs`. `McpServer::tool_docs_context` in `server.rs` does the
//! index-readiness gating and `ChunkStore` open (same as every other tool
//! handler) and then delegates here.

use anyhow::{Context, Result};
use semantex_core::index::docs_scaffold::{
    self, ModuleScaffold, OverviewScaffold, apply_module_budget, apply_overview_budget,
};
use semantex_core::index::storage::ChunkStore;
use std::path::Path;

/// Default scaffold size budget in (approximate) tokens when the caller
/// doesn't supply one. Mirrors the order of magnitude of `semantex_agent`'s
/// default byte budget (12000 bytes ≈ 3K tokens), sized up a bit since a
/// scaffold is meant to be exhaustive within one file/overview rather than a
/// ranked top-K search result.
pub const DEFAULT_BUDGET_TOKENS: u64 = 6_000;
const MIN_BUDGET_TOKENS: u64 = 500;
const MAX_BUDGET_TOKENS: u64 = 100_000;

/// Parsed `scope` argument.
#[derive(Debug)]
enum Scope {
    Overview,
    Module(String),
}

fn parse_scope(args: &serde_json::Value) -> Result<Scope> {
    let raw = args
        .get("scope")
        .ok_or_else(|| anyhow::anyhow!("Missing required parameter: scope"))?;

    if let Some(s) = raw.as_str() {
        if s == "overview" {
            return Ok(Scope::Overview);
        }
        anyhow::bail!(
            "Unknown scope `{s}` — expected the string \"overview\" or an object \
             {{\"module\": \"<path>\"}}"
        );
    }
    if let Some(module) = raw.get("module").and_then(|v| v.as_str()) {
        if module.trim().is_empty() {
            anyhow::bail!("`scope.module` must be a non-empty path");
        }
        return Ok(Scope::Module(module.to_string()));
    }
    anyhow::bail!(
        "Invalid `scope` — expected the string \"overview\" or an object \
         {{\"module\": \"<path>\"}}, got: {raw}"
    );
}

fn parse_budget(args: &serde_json::Value) -> u64 {
    args.get("budget")
        .and_then(serde_json::Value::as_u64)
        .map_or(DEFAULT_BUDGET_TOKENS, |b| {
            b.clamp(MIN_BUDGET_TOKENS, MAX_BUDGET_TOKENS)
        })
}

/// Result of a `semantex_docs_context` call: the structured scaffold plus a
/// compact human-readable rendering of the same data.
pub struct DocsContextResult {
    pub text: String,
    pub structured: serde_json::Value,
}

/// Run the tool: parse args, build the requested scaffold, apply the token
/// budget, and render both the structured JSON and a compact text summary.
pub fn run(
    store: &ChunkStore,
    db_path: &Path,
    args: &serde_json::Value,
) -> Result<DocsContextResult> {
    let scope = parse_scope(args)?;
    let budget_tokens = parse_budget(args) as usize;

    match scope {
        Scope::Overview => {
            let mut scaffold = docs_scaffold::build_overview_scaffold(store, db_path)
                .context("Failed to build overview docs scaffold")?;
            apply_overview_budget(&mut scaffold, budget_tokens);
            let text = render_overview_text(&scaffold);
            let structured = serde_json::to_value(&scaffold)
                .context("Failed to serialize overview docs scaffold")?;
            Ok(DocsContextResult { text, structured })
        }
        Scope::Module(module_path) => {
            let mut scaffold = docs_scaffold::build_module_scaffold(store, &module_path)
                .with_context(|| {
                    format!("Failed to build module docs scaffold for `{module_path}`")
                })?;
            apply_module_budget(&mut scaffold, budget_tokens);
            let text = render_module_text(&scaffold);
            let structured = serde_json::to_value(&scaffold)
                .context("Failed to serialize module docs scaffold")?;
            Ok(DocsContextResult { text, structured })
        }
    }
}

/// Truncate a docstring for the compact text rendering (the full text is
/// always present in `structured`).
fn short(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max {
        return s.replace('\n', " ");
    }
    let truncated: String = s.chars().take(max).collect();
    format!("{}…", truncated.replace('\n', " "))
}

fn render_overview_text(s: &OverviewScaffold) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "# Overview scaffold\n\n{} files, {} chunks indexed.\n",
        s.total_files, s.total_chunks
    );

    let _ = writeln!(out, "## God nodes ({})", s.god_nodes.len());
    for g in &s.god_nodes {
        let _ = writeln!(
            out,
            "- {} (centrality {:.3}){} — {}:{}-{}",
            g.symbol,
            g.centrality,
            g.semantic_role
                .as_ref()
                .map_or_else(String::new, |r| format!(" [{r}]")),
            g.provenance.file,
            g.provenance.start_line,
            g.provenance.end_line
        );
    }

    let _ = writeln!(out, "\n## Communities ({})", s.communities.len());
    for c in &s.communities {
        let entries: Vec<String> = c
            .entry_points
            .iter()
            .map(|ep| {
                format!(
                    "{} ({}:{})",
                    ep.symbol, ep.provenance.file, ep.provenance.start_line
                )
            })
            .collect();
        let _ = writeln!(
            out,
            "- {} — {} files. Entry points: {}",
            c.label,
            c.size,
            if entries.is_empty() {
                "none".to_string()
            } else {
                entries.join(", ")
            }
        );
    }

    let _ = writeln!(out, "\n## Boundaries ({})", s.boundaries.len());
    for b in &s.boundaries {
        let _ = writeln!(out, "- {} -> {} ({} edges)", b.from, b.to, b.edge_count);
    }

    let _ = writeln!(out, "\n## Language stats ({})", s.language_stats.len());
    for l in &s.language_stats {
        let _ = writeln!(
            out,
            "- {}: {} files, {} chunks",
            l.language, l.file_count, l.chunk_count
        );
    }

    let _ = writeln!(
        out,
        "\n## Module inventory ({}, sorted by symbol count)",
        s.module_inventory.len()
    );
    for m in &s.module_inventory {
        let _ = writeln!(
            out,
            "- {} [{}] role={} symbols={}",
            m.path,
            m.language.as_deref().unwrap_or("?"),
            m.file_role.as_deref().unwrap_or("unknown"),
            m.symbol_count
        );
    }

    out
}

fn render_module_text(s: &ModuleScaffold) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "# Module scaffold: {}\nlanguage={} role={}\n",
        s.path,
        s.language.as_deref().unwrap_or("?"),
        s.file_role.as_deref().unwrap_or("unknown")
    );

    let _ = writeln!(out, "## Symbols ({})", s.symbols.len());
    for sym in &s.symbols {
        let _ = writeln!(
            out,
            "- [{}] {}{} — {}:{}-{}",
            sym.kind,
            sym.name,
            sym.signature
                .as_ref()
                .map_or_else(String::new, |sig| format!(" `{}`", short(sig, 80))),
            sym.provenance.file,
            sym.provenance.start_line,
            sym.provenance.end_line
        );
        if let Some(doc) = &sym.docstring
            && !doc.trim().is_empty()
        {
            let _ = writeln!(out, "    doc: {}", short(doc, 160));
        }
    }

    let _ = writeln!(out, "\n## Imports ({})", s.imports.len());
    for i in &s.imports {
        let _ = writeln!(out, "- {i}");
    }

    let _ = writeln!(out, "\n## Imported by ({})", s.imported_by.len());
    for i in &s.imported_by {
        let _ = writeln!(out, "- {i}");
    }

    let _ = writeln!(out, "\n## Calls out ({})", s.calls_out.len());
    for c in &s.calls_out {
        match &c.resolved {
            Some(p) => {
                let _ = writeln!(
                    out,
                    "- {} — {}:{}-{}",
                    c.callee_name, p.file, p.start_line, p.end_line
                );
            }
            None => {
                let _ = writeln!(out, "- {} (unresolved)", c.callee_name);
            }
        }
    }

    let _ = writeln!(out, "\n## Calls in ({})", s.calls_in.len());
    for c in &s.calls_in {
        let _ = writeln!(
            out,
            "- {} — {}:{}-{}",
            c.caller_symbol.as_deref().unwrap_or("(unknown caller)"),
            c.provenance.file,
            c.provenance.start_line,
            c.provenance.end_line
        );
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_scope_overview_string() {
        let args = serde_json::json!({ "scope": "overview" });
        assert!(matches!(parse_scope(&args).unwrap(), Scope::Overview));
    }

    #[test]
    fn parse_scope_module_object() {
        let args = serde_json::json!({ "scope": { "module": "src/lib.rs" } });
        match parse_scope(&args).unwrap() {
            Scope::Module(m) => assert_eq!(m, "src/lib.rs"),
            Scope::Overview => panic!("expected Module"),
        }
    }

    #[test]
    fn parse_scope_missing_errors() {
        let args = serde_json::json!({});
        let err = parse_scope(&args).unwrap_err();
        assert!(format!("{err}").contains("Missing required parameter"));
    }

    #[test]
    fn parse_scope_unknown_string_errors() {
        let args = serde_json::json!({ "scope": "bogus" });
        let err = parse_scope(&args).unwrap_err();
        assert!(format!("{err}").contains("Unknown scope"));
    }

    #[test]
    fn parse_scope_empty_module_errors() {
        let args = serde_json::json!({ "scope": { "module": "  " } });
        let err = parse_scope(&args).unwrap_err();
        assert!(format!("{err}").contains("non-empty path"));
    }

    #[test]
    fn parse_budget_defaults_and_clamps() {
        assert_eq!(parse_budget(&serde_json::json!({})), DEFAULT_BUDGET_TOKENS);
        assert_eq!(
            parse_budget(&serde_json::json!({ "budget": 1 })),
            MIN_BUDGET_TOKENS
        );
        assert_eq!(
            parse_budget(&serde_json::json!({ "budget": 999_999_999u64 })),
            MAX_BUDGET_TOKENS
        );
        assert_eq!(parse_budget(&serde_json::json!({ "budget": 8000 })), 8000);
    }

    #[test]
    fn short_truncates_and_flattens_newlines() {
        assert_eq!(short("hello\nworld", 20), "hello world");
        let long = "a".repeat(50);
        let truncated = short(&long, 10);
        assert_eq!(truncated.chars().count(), 11); // 10 chars + ellipsis
    }
}
