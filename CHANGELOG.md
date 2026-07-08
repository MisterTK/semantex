# Changelog

## v1.0.0 — 2026-07-08

First stable release. Everything below shipped since the v0.8 tag, developed and
adversarially reviewed as a single program.

### Highlights

- **Multi-branch indexing** — branch switches are first-class: per-branch index
  snapshots with restore-and-incremental-update, so switching back and forth
  re-embeds only files that actually changed. Automatic detection at every entry
  point (CLI, watch, serve daemon, MCP), git-worktree-aware HEAD watching,
  snapshot retention caps (`SEMANTEX_MAX_BRANCH_INDEXES`, default 5).
- **Cross-repo federated search** — `scope: "repo" | "all" | [names]` on the
  `semantex_agent`/`semantex_search` MCP tools and `--scope` on the CLI. One
  query fans out across every indexed repo in the registry with provenance-tagged,
  RRF-fused results. Skipped/stale targets are reported, never silently dropped.
- **Git-history indexing** — commit history is a searchable dimension of the
  index: bounded incremental `git log` population (500 commits default,
  `SEMANTEX_HISTORY_COMMITS`), opt-in blame (`SEMANTEX_HISTORY_BLAME=1`),
  FTS5 commit-message search, `semantex history <file>`, and an
  `include_history` option on agent queries. Injection-proof parsing (NUL
  separators; hostile commit messages cannot forge records).
- **Multi-client serve daemon** — thread-per-connection with graceful drain,
  searcher LRU keyed by (project, branch), per-request branch-switch
  reconciliation, and staleness reload when the index is rebuilt externally.
- **Project memory** — `semantex_memory_save` / `semantex_memory_recall` MCP
  tools over a branch-independent FTS5 notes store; agents persist decisions and
  gotchas across sessions. Zero LLM wiring.
- **Docs scaffolding** — `semantex_docs_context` emits deterministic,
  provenance-carrying documentation scaffolds (architecture overview, per-module
  symbol/call-graph inventories); the `semantex-docs` skill guides the host
  agent to write and maintain `semantex_docs/`. Structure from semantex, prose
  from your agent.
- **MCP HTTP authentication** — bearer-token auth on the HTTP transport
  (`SEMANTEX_HTTP_TOKEN` or auto-generated persisted token), constant-time
  comparison, unauthenticated remote exposure is no longer possible.
- **SQL AST chunking** — real tree-sitter SQL parsing (25 AST languages total).

### Fixed

- **First-run memory abort on 16 GB machines**: the kernel memory cap limited
  virtual address space but was reasoned about as RSS; PLAID's k-means
  transiently needs ~20 GB AS at ~9.6 GB real RSS. The AS cap is now a true last
  resort; the RSS soft cap remains the OOM guard with an actionable message.
  Default indexing also uses all cores when it holds the only build slot
  (~1.5× faster).
- **Silently dark BM25 channel**: real-world queries containing markdown
  comments, dangling booleans, or `IN [` fragments failed tantivy's strict
  parser and were swallowed into empty sparse results. Lenient-parse fallback
  restores the hybrid contract; measured MRR on SWE-bench localization rose
  0.400 → 0.522 (hybrid) and 0.000 → 0.600 (agent-routed).
- Toolset surfaces: `--toolset structural` is a real 5-tool opt-in bundle again;
  help text counts match reality; `semantex status` names the active dense
  backend unambiguously.

### Measured (committed, reproducible — `benchmarks/RESULTS-v1.md`)

- Agent efficiency vs built-in Grep/Glob/Read (flask, blind-judged): −65% cost,
  −39% cumulative context burden, −51% tool calls, −54% wall time, quality
  4.80 vs 3.80.
- SWE-bench-Verified file localization: agent-routed MRR@10 0.600 vs ripgrep
  0.095.
- Cross-encoder reranking stays **off by default** by measurement: no quality
  win that survives noise at ~90× the latency.

### Upgrade notes

- **Index schema v13**: existing indexes rebuild automatically on first use
  after upgrading (one-time cost; see First Index times in the README).
- The MCP HTTP transport now requires a bearer token for non-localhost use.
- `benchmarks/` claims are now backed by committed runs; older README numbers
  were replaced with traceable ones.
