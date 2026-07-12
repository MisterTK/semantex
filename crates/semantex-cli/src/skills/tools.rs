//! Canonical metadata for semantex MCP tools.
//!
//! **NOTE — duplicated source of truth.**
//!
//! The MCP server in `crates/semantex-mcp/src/server.rs::handle_tools_list`
//! declares the same tool surface for JSON-RPC clients. For v0.3 we keep a
//! parallel registry here in the CLI so that `semantex skills-generate` can
//! emit platform skill files without depending on a public function from
//! `semantex-mcp`. After v0.3, factor a shared `pub fn tool_metadata()` in
//! `semantex-mcp` and import it from here — see follow-up TODO.
//!
//! Until then, any change to the MCP tool surface must be mirrored in both
//! places. The build will not catch the drift on its own; reviewers must
//! enforce.

use serde::Serialize;

/// One MCP tool described in a platform-neutral shape.
#[derive(Debug, Clone, Serialize)]
pub struct ToolMetadata {
    /// Canonical machine name, e.g. `semantex_agent`.
    pub name: &'static str,
    /// Human-readable title for UI listings.
    pub title: &'static str,
    /// One-paragraph natural-language description for the agent.
    pub description: &'static str,
    /// Bullet-point guidance: when an agent should prefer this tool.
    pub when_to_use: &'static [&'static str],
    /// Argument shape for the tool. Stays small on purpose — these are skill
    /// docs, not full JSON Schema.
    pub args: &'static [ToolArg],
    /// Short usage examples for the skill body.
    pub examples: &'static [ToolExample],
    /// Mutates local state? Used to set platform-specific safety hints.
    pub mutates: bool,
    /// Is this tool currently live in the MCP server? `false` means it ships
    /// in a future v0.3.x point release. We still document it so skill files
    /// stay forward-compatible.
    pub live: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolArg {
    pub name: &'static str,
    pub ty: &'static str,
    pub required: bool,
    pub description: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolExample {
    /// Short label, e.g. `Find a definition`.
    pub label: &'static str,
    /// Example argument blob as a compact JSON-ish string.
    pub args_json: &'static str,
}

/// Returns the canonical set of semantex MCP tools.
///
/// Mirrors the 13 tools currently registered in
/// `semantex-mcp/src/server.rs::handle_tools_list` after W5 merge (the seven
/// original tools plus the six structural M1–M6 tools shipped in v0.3 per
/// `docs/superpowers/specs/2026-05-24-semantex-v0.3-sota-design.md`). Future
/// tools that ship later can be marked `live: false` so generated skill files
/// can be filtered per release.
pub fn all_tools() -> Vec<ToolMetadata> {
    vec![
        // ---- Original 7 tools ----
        ToolMetadata {
            name: "semantex_agent",
            title: "Intelligent Code Search",
            description: concat!(
                "Intelligent code search with automatic query classification. ",
                "Analyzes the query, picks the best strategy (semantic, exact symbol, ",
                "graph walk, deep search, regex, file pattern), executes with fallbacks, ",
                "and returns a pre-formatted answer. Default for all code search queries."
            ),
            when_to_use: &[
                "Default entry point for any code search question.",
                "Natural-language questions like \"how does X work?\".",
                "When you would otherwise chain grep + read.",
            ],
            args: &[
                ToolArg {
                    name: "query",
                    ty: "string",
                    required: false,
                    description: "Natural language question, code symbol, regex pattern, or glob pattern. Use this for a single query.",
                },
                ToolArg {
                    name: "queries",
                    ty: "array<string>",
                    required: false,
                    description: "Multiple queries merged into one result. Use instead of `query` for 2-3 related concepts.",
                },
                ToolArg {
                    name: "path",
                    ty: "string",
                    required: false,
                    description: "Project path (default: current working directory).",
                },
                ToolArg {
                    name: "depth",
                    ty: "string",
                    required: false,
                    description: "One of `quick`, `search`, `deep`. Omit to auto-detect.",
                },
                ToolArg {
                    name: "focus",
                    ty: "string",
                    required: false,
                    description: "One of `implementation`, `callers`, `signatures`, `patterns`.",
                },
                ToolArg {
                    name: "full_code",
                    ty: "boolean",
                    required: false,
                    description: "Include full source code blocks (default: false).",
                },
                ToolArg {
                    name: "budget",
                    ty: "integer",
                    required: false,
                    description: "Response size budget in bytes (default: 12000).",
                },
            ],
            examples: &[
                ToolExample {
                    label: "How does something work",
                    args_json: r#"{"query": "how does auth work", "depth": "deep"}"#,
                },
                ToolExample {
                    label: "Two related questions in one call",
                    args_json: r#"{"queries": ["rate limiting logic", "retry backoff"], "depth": "search"}"#,
                },
            ],
            mutates: false,
            live: true,
        },
        ToolMetadata {
            name: "semantex_search",
            title: "Semantic Code Search",
            description: concat!(
                "Find code by meaning or exact match (`grep_mode=true`). Returns ranked file ",
                "chunks with paths, lines and scores. 25+ languages supported. Prefer ",
                "`semantex_agent` for most queries; use this only when you need structured JSON."
            ),
            when_to_use: &[
                "You need structured JSON results for programmatic processing.",
                "You need explicit control over `max_results`, `rerank` or `grep_mode`.",
            ],
            args: &[
                ToolArg {
                    name: "query",
                    ty: "string",
                    required: true,
                    description: "Natural language search query.",
                },
                ToolArg {
                    name: "path",
                    ty: "string",
                    required: false,
                    description: "Project path to search (defaults to current working directory).",
                },
                ToolArg {
                    name: "max_results",
                    ty: "integer",
                    required: false,
                    description: "Maximum results to return (default: 10).",
                },
                ToolArg {
                    name: "rerank",
                    ty: "boolean",
                    required: false,
                    description: "Enable cross-encoder reranking (slower but may improve ranking).",
                },
                ToolArg {
                    name: "grep_mode",
                    ty: "boolean",
                    required: false,
                    description: "Fast grep-like search using only sparse + exact matching.",
                },
            ],
            examples: &[ToolExample {
                label: "Structured search with reranking",
                args_json: r#"{"query": "database connection pool", "rerank": true, "max_results": 20}"#,
            }],
            mutates: false,
            live: true,
        },
        ToolMetadata {
            name: "semantex_deep",
            title: "Deep Code Search",
            description: concat!(
                "One call replaces 5-10 grep+read iterations. Searches, reads, graph-expands ",
                "and summarises into a prose answer with sources. Prefer `semantex_agent` ",
                "for most queries; use this only when you need structured JSON."
            ),
            when_to_use: &[
                "You need structured JSON deep-search results.",
                "You explicitly want the deep pipeline (search → graph → read → summarise).",
            ],
            args: &[
                ToolArg {
                    name: "query",
                    ty: "string",
                    required: true,
                    description: "Natural language question about the code.",
                },
                ToolArg {
                    name: "path",
                    ty: "string",
                    required: false,
                    description: "Project path (defaults to current working directory).",
                },
            ],
            examples: &[ToolExample {
                label: "Architecture question",
                args_json: r#"{"query": "how does the indexing pipeline work end to end"}"#,
            }],
            mutates: false,
            live: true,
        },
        ToolMetadata {
            name: "semantex_index",
            title: "Build Search Index",
            description: concat!(
                "Build or update the semantex search index. Usually NOT needed — semantex ",
                "auto-indexes on first search. Call only to force a rebuild after major changes."
            ),
            when_to_use: &[
                "Force a rebuild after a large refactor.",
                "Prime an index in CI before the first search.",
            ],
            args: &[ToolArg {
                name: "path",
                ty: "string",
                required: true,
                description: "Project path to index.",
            }],
            examples: &[ToolExample {
                label: "Reindex current directory",
                args_json: r#"{"path": "."}"#,
            }],
            mutates: true,
            live: true,
        },
        ToolMetadata {
            name: "semantex_status",
            title: "Index Status",
            description: concat!(
                "Check semantex index status: whether it exists, file count, chunk count, ",
                "freshness. Use to verify indexing is complete."
            ),
            when_to_use: &[
                "Verify the index has finished building.",
                "Inspect all known projects and their freshness.",
            ],
            args: &[ToolArg {
                name: "path",
                ty: "string",
                required: false,
                description: "Project path to check (defaults to all registered projects).",
            }],
            examples: &[ToolExample {
                label: "Check all indexed projects",
                args_json: "{}",
            }],
            mutates: false,
            live: true,
        },
        ToolMetadata {
            name: "semantex_health",
            title: "Health Check",
            description: "Health check for the semantex system, including model availability and configuration.",
            when_to_use: &["Diagnose why semantex isn't returning results."],
            args: &[],
            examples: &[ToolExample {
                label: "Run a health check",
                args_json: "{}",
            }],
            mutates: false,
            live: true,
        },
        ToolMetadata {
            name: "semantex_validate",
            title: "Validate Index",
            description: concat!(
                "Run consistency checks on a semantex index: meta-DB sync, stale files, ",
                "dense/sparse index integrity, graph consistency. Returns per-check pass/fail ",
                "with details."
            ),
            when_to_use: &[
                "After a crash or partial rebuild.",
                "When search results look stale or inconsistent.",
            ],
            args: &[ToolArg {
                name: "path",
                ty: "string",
                required: false,
                description: "Project path (defaults to current working directory).",
            }],
            examples: &[ToolExample {
                label: "Validate current directory's index",
                args_json: "{}",
            }],
            mutates: false,
            live: true,
        },
        // ---- v0.3 structural tools (M1–M6 from the SOTA spec) ----
        ToolMetadata {
            name: "semantex_symbol",
            title: "Exact Symbol Lookup",
            description: concat!(
                "Exact symbol lookup. Returns location, signature, docstring, semantic role, ",
                "callers and callees count. Backed by the global graph symbol index. ",
                "Replaces 3–5 grep+read iterations."
            ),
            when_to_use: &[
                "You know the exact name of a function, type or constant.",
                "You want a one-shot answer to \"where is `foo` defined?\".",
            ],
            args: &[
                ToolArg {
                    name: "name",
                    ty: "string",
                    required: true,
                    description: "Exact symbol name.",
                },
                ToolArg {
                    name: "kind",
                    ty: "string",
                    required: false,
                    description: "Optional kind filter (function, class, type, constant, …).",
                },
            ],
            examples: &[ToolExample {
                label: "Find a function definition",
                args_json: r#"{"name": "NewSessionManager"}"#,
            }],
            mutates: false,
            live: false,
        },
        ToolMetadata {
            name: "semantex_callers",
            title: "Reverse Call Graph",
            description: concat!(
                "Reverse call-graph walk, depth 1 or 2. Returns an array of ",
                "`{caller_location, caller_signature, edge_kind}`. Replaces 5–15 grep iterations."
            ),
            when_to_use: &[
                "Find every caller of a function.",
                "Impact analysis before changing a signature.",
            ],
            args: &[
                ToolArg {
                    name: "symbol",
                    ty: "string",
                    required: true,
                    description: "Exact symbol name.",
                },
                ToolArg {
                    name: "depth",
                    ty: "integer",
                    required: false,
                    description: "Walk depth (1 or 2, default 1).",
                },
            ],
            examples: &[ToolExample {
                label: "Find direct callers",
                args_json: r#"{"symbol": "ConnectionPool::get"}"#,
            }],
            mutates: false,
            live: false,
        },
        ToolMetadata {
            name: "semantex_callees",
            title: "Forward Call Graph",
            description: "Forward call-graph walk. Same shape as `semantex_callers` but outbound edges.",
            when_to_use: &[
                "Understand what a function calls.",
                "Trace data flow downstream from a public API.",
            ],
            args: &[
                ToolArg {
                    name: "symbol",
                    ty: "string",
                    required: true,
                    description: "Exact symbol name.",
                },
                ToolArg {
                    name: "depth",
                    ty: "integer",
                    required: false,
                    description: "Walk depth (1 or 2, default 1).",
                },
            ],
            examples: &[ToolExample {
                label: "List downstream calls",
                args_json: r#"{"symbol": "handle_request", "depth": 2}"#,
            }],
            mutates: false,
            live: false,
        },
        ToolMetadata {
            name: "semantex_implementations",
            title: "Trait / Interface Implementations",
            description: concat!(
                "Find all implementations of a trait, interface or protocol. Returns ",
                "`{impl_location, type_name, method_overrides}`. Backed by hierarchy edges in ",
                "the global graph."
            ),
            when_to_use: &[
                "Enumerate concrete types behind an interface.",
                "Audit overrides of a virtual method.",
            ],
            args: &[ToolArg {
                name: "trait_or_interface",
                ty: "string",
                required: true,
                description: "Trait, interface, abstract class or protocol name.",
            }],
            examples: &[ToolExample {
                label: "List implementations",
                args_json: r#"{"trait_or_interface": "Handler"}"#,
            }],
            mutates: false,
            live: false,
        },
        ToolMetadata {
            name: "semantex_examples",
            title: "Pattern Exemplars",
            description: concat!(
                "Pattern-catalog-backed exemplar finder. `pattern` must be an **exact ",
                "catalog name** (a language-prefixed enum value), not a free-form label. ",
                "Valid Rust pattern names include: `rust.drop_impl`, `rust.tokio_spawn`, ",
                "`rust.serde_derive`, `rust.result_question_mark_chain`, ",
                "`rust.error_with_thiserror`, `rust.anyhow_context`, `rust.builder_pattern`, ",
                "`rust.async_trait_method`, `rust.test_fn`, `rust.unsafe_block`. ",
                "Valid TypeScript pattern names include: `ts.promise_all_settled`, ",
                "`ts.promise_all`, `ts.async_await`, `ts.try_catch_async`, ",
                "`ts.optional_chaining`, `ts.discriminated_union`, `ts.fetch_api_call`, ",
                "`ts.error_throw`. See `crates/semantex-core/src/index/pattern_catalog.rs` ",
                "for the full set (32 Rust + 30 TypeScript patterns). ",
                "TODO: expose `semantex examples --list-patterns` to enumerate at runtime."
            ),
            when_to_use: &[
                "Find idiomatic in-repo examples of a known pattern.",
                "Avoid copying from broad grep hits when curated exemplars exist.",
            ],
            args: &[
                ToolArg {
                    name: "pattern",
                    ty: "string",
                    required: true,
                    description: "Exact catalog name like `rust.tokio_spawn` or `ts.try_catch_async` — NOT a free-form label.",
                },
                ToolArg {
                    name: "language",
                    ty: "string",
                    required: false,
                    description: "Restrict to a specific language (`rust`, `typescript`).",
                },
                ToolArg {
                    name: "max",
                    ty: "integer",
                    required: false,
                    description: "Maximum exemplars to return.",
                },
            ],
            examples: &[
                ToolExample {
                    label: "Find tokio::spawn usages in Rust",
                    args_json: r#"{"pattern": "rust.tokio_spawn"}"#,
                },
                ToolExample {
                    label: "Find try/catch around await in TypeScript",
                    args_json: r#"{"pattern": "ts.try_catch_async", "max": 5}"#,
                },
            ],
            mutates: false,
            live: false,
        },
        ToolMetadata {
            name: "semantex_architecture",
            title: "Architectural Primer",
            description: concat!(
                "Session-start architectural primer. Returns compact JSON: god_nodes ",
                "(high-centrality symbols), communities (subsystems with entry points), and ",
                "boundaries (cross-community edges). Primes agents at turn 1 instead of letting ",
                "them explore for 10 turns. `focus` is an **enum** selecting which section to ",
                "return — one of `god_nodes`, `communities`, or `boundaries`. Omit `focus` to ",
                "return all three sections. Free-form values are ignored and yield an empty ",
                "response."
            ),
            when_to_use: &[
                "First call when starting work on an unfamiliar repository.",
                "Whenever you need a bird's-eye structural overview.",
            ],
            args: &[ToolArg {
                name: "focus",
                ty: "string",
                required: false,
                description: "Section enum: `god_nodes` | `communities` | `boundaries`. Omit to return all three.",
            }],
            examples: &[
                ToolExample {
                    label: "Whole-repo overview (all three sections)",
                    args_json: "{}",
                },
                ToolExample {
                    label: "Only the most-central symbols",
                    args_json: r#"{"focus": "god_nodes"}"#,
                },
            ],
            mutates: false,
            live: false,
        },
        // ---- v13 Wave 2 — deterministic docs scaffold (zero LLM wiring) ----
        ToolMetadata {
            name: "semantex_docs_context",
            title: "Documentation Context Scaffold",
            description: concat!(
                "Deterministic documentation scaffold — NOT an LLM call, does not write prose. ",
                "Returns structurally-complete data (symbol inventory, call-graph edges, import ",
                "edges, existing doc-comment text, file:line provenance) for the calling agent to ",
                "turn into maintained markdown docs. Pair with the `semantex-docs` skill for the ",
                "full write/refresh workflow."
            ),
            when_to_use: &[
                "Asked to document a codebase, write architecture docs, or generate a README.",
                "Keeping existing docs under `semantex_docs/` in sync with code that changed.",
            ],
            args: &[
                ToolArg {
                    name: "scope",
                    ty: "string | object",
                    required: true,
                    description: "\"overview\" for the repo-wide architecture + module inventory scaffold, or {\"module\": \"<path>\"} for one file's symbol/call/import scaffold.",
                },
                ToolArg {
                    name: "path",
                    ty: "string",
                    required: false,
                    description: "Project path (defaults to current working directory).",
                },
                ToolArg {
                    name: "budget",
                    ty: "integer",
                    required: false,
                    description: "Approximate token budget for the returned scaffold (default 6000). Oversized scaffolds are trimmed deterministically.",
                },
            ],
            examples: &[
                ToolExample {
                    label: "Repo-wide architecture + module inventory",
                    args_json: r#"{"scope": "overview"}"#,
                },
                ToolExample {
                    label: "Symbol/call/import scaffold for one file",
                    args_json: r#"{"scope": {"module": "src/index/storage.rs"}}"#,
                },
            ],
            mutates: false,
            live: true,
        },
        // ---- v13 Wave 2 — project memory ----
        ToolMetadata {
            name: "semantex_memory_save",
            title: "Save Project Memory Note",
            description: concat!(
                "Save a short note to durable project memory — persists across sessions in ",
                "`.semantex/memory.db`, independent of the code index. Use for decisions, gotchas, or ",
                "conventions discovered that aren't recoverable by searching the code. Do NOT use for ",
                "anything derivable by reading or searching the code."
            ),
            when_to_use: &[
                "You just figured out a non-obvious design decision, gotcha, or convention worth remembering.",
                "A task-specific follow-up needs to survive to a later session.",
            ],
            args: &[
                ToolArg {
                    name: "content",
                    ty: "string",
                    required: true,
                    description: "The note text. Keep it short and self-contained.",
                },
                ToolArg {
                    name: "scope",
                    ty: "string",
                    required: false,
                    description: "Freeform scope key (default \"global\"). Suggested: \"global\", \"file:<rel_path>\", \"module:<dir>\", \"task:<slug>\".",
                },
                ToolArg {
                    name: "tags",
                    ty: "array<string>",
                    required: false,
                    description: "Optional freeform tags for this note.",
                },
                ToolArg {
                    name: "path",
                    ty: "string",
                    required: false,
                    description: "Project path (defaults to current working directory).",
                },
            ],
            examples: &[ToolExample {
                label: "Record a gotcha discovered while working on a module",
                args_json: r#"{"content": "memory.db writes must go through MemoryStore, not raw SQL", "scope": "module:index", "tags": ["gotcha"]}"#,
            }],
            mutates: true,
            live: true,
        },
        ToolMetadata {
            name: "semantex_memory_recall",
            title: "Recall Project Memory Notes",
            description: concat!(
                "Recall notes previously saved with semantex_memory_save, ranked best-match-first. Use ",
                "at the start of a task to check whether relevant context was already recorded. Falls ",
                "back to listing the most recent notes when `query` is omitted."
            ),
            when_to_use: &[
                "Starting a task and checking for prior decisions/gotchas before rediscovering them.",
                "Before making a decision that might already have recorded context.",
            ],
            args: &[
                ToolArg {
                    name: "query",
                    ty: "string",
                    required: false,
                    description: "Free text to rank notes by relevance against. Omit to list the most recent notes.",
                },
                ToolArg {
                    name: "scope",
                    ty: "string",
                    required: false,
                    description: "Restrict to notes saved under this exact scope key.",
                },
                ToolArg {
                    name: "limit",
                    ty: "integer",
                    required: false,
                    description: "Max notes to return (default 5, clamped to [1, 50]).",
                },
                ToolArg {
                    name: "path",
                    ty: "string",
                    required: false,
                    description: "Project path (defaults to current working directory).",
                },
            ],
            examples: &[ToolExample {
                label: "Check for prior notes about a module before editing it",
                args_json: r#"{"query": "caching invalidation", "scope": "module:index"}"#,
            }],
            mutates: false,
            live: true,
        },
        ToolMetadata {
            name: "semantex_history",
            title: "Git History",
            description: concat!(
                "Query indexed git history: recent commits, commits since a tag/sha/date, ",
                "commits touching a file, full-text search over commit messages, and ",
                "per-commit detail with a budget-bounded diff. Refreshes incrementally from ",
                "git on every call — results always reflect THIS LOCAL CLONE's current HEAD, ",
                "not upstream; an un-pulled clone looks falsely idle, so `git pull` first when ",
                "'latest' must mean truly current. Cross-repo via 'scope' for dependency change ",
                "tracking. NOT for code content — use semantex_agent for that."
            ),
            when_to_use: &[
                "Drafting release notes or a changelog since the last tag.",
                "Checking what changed recently, in this repo or across all indexed dependency repos (scope='all').",
                "Onboarding to an unfamiliar repo: recent change activity alongside code search.",
                "Before answering 'what's new/latest' for someone else's repo: consider `git pull` first — this tool reads the local clone, never the network.",
            ],
            args: &[
                ToolArg {
                    name: "since",
                    ty: "string",
                    required: false,
                    description: "Only commits after this point: tag, sha, git rev, or YYYY-MM-DD.",
                },
                ToolArg {
                    name: "query",
                    ty: "string",
                    required: false,
                    description: "Full-text match over commit messages.",
                },
                ToolArg {
                    name: "file",
                    ty: "string",
                    required: false,
                    description: "Repo-relative path — only commits touching it.",
                },
                ToolArg {
                    name: "author",
                    ty: "string",
                    required: false,
                    description: "Author-name substring filter.",
                },
                ToolArg {
                    name: "limit",
                    ty: "integer",
                    required: false,
                    description: "Max commits per project (default 20).",
                },
                ToolArg {
                    name: "commits",
                    ty: "array<string>",
                    required: false,
                    description: "Detail mode: shas to expand with --stat and a bounded patch (max 10/call).",
                },
                ToolArg {
                    name: "scope",
                    ty: "string|array<string>",
                    required: false,
                    description: "'repo' (default), 'all', or registered project names for cross-repo history.",
                },
                ToolArg {
                    name: "path",
                    ty: "string",
                    required: false,
                    description: "Project path (defaults to current working directory).",
                },
            ],
            examples: &[
                ToolExample {
                    label: "Release notes since the last tag",
                    args_json: r#"{"since": "v1.0.0", "limit": 50}"#,
                },
                ToolExample {
                    label: "What changed across all dependency repos this week",
                    args_json: r#"{"since": "2026-07-03", "scope": "all"}"#,
                },
                ToolExample {
                    label: "Drill into two commits with diffs",
                    args_json: r#"{"commits": ["abc1234f", "def5678a"]}"#,
                },
            ],
            mutates: false,
            live: true,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Finding 14 regression guard: `semantex_examples` must surface real
    /// catalog pattern names in its example invocations (language-prefixed
    /// enum values like `rust.*` or `ts.*`), NOT free-form labels such as
    /// "error handling pattern" which the MCP handler rejects with an
    /// empty result set.
    #[test]
    fn semantex_examples_uses_real_catalog_pattern_names() {
        let tool = all_tools()
            .into_iter()
            .find(|t| t.name == "semantex_examples")
            .expect("semantex_examples must be registered");
        assert!(
            !tool.examples.is_empty(),
            "semantex_examples must have at least one example"
        );
        for ex in tool.examples {
            assert!(
                ex.args_json.contains("rust.") || ex.args_json.contains("ts."),
                "example args_json `{}` must reference a language-prefixed pattern \
                 (e.g. `rust.tokio_spawn` or `ts.try_catch_async`) — free-form labels \
                 like \"error handling pattern\" don't match the catalog and return \
                 an empty response",
                ex.args_json
            );
        }
    }

    /// Finding 14 regression guard: `semantex_architecture`'s `focus` field
    /// is an enum (god_nodes | communities | boundaries), not a free-form
    /// subsystem label. Free-form values yield an empty `{}` response.
    #[test]
    fn semantex_architecture_focus_examples_use_enum_values() {
        let tool = all_tools()
            .into_iter()
            .find(|t| t.name == "semantex_architecture")
            .expect("semantex_architecture must be registered");
        assert!(
            !tool.examples.is_empty(),
            "semantex_architecture must have at least one example"
        );
        for ex in tool.examples {
            // Examples that pass `focus` must use one of the enum values.
            if ex.args_json.contains("\"focus\"") {
                let uses_enum_value = ex.args_json.contains("god_nodes")
                    || ex.args_json.contains("communities")
                    || ex.args_json.contains("boundaries");
                assert!(
                    uses_enum_value,
                    "example args_json `{}` references `focus` but doesn't use the \
                     enum values (god_nodes | communities | boundaries) — free-form \
                     focus values yield an empty MCP response",
                    ex.args_json
                );
            }
        }
    }
}
