# Changelog

## v1.1.0 — 2026-07-19

Cinder — compiled encoder-free indexing — becomes the default dense-indexing
path. This is a real behavior change, not a drop-in upgrade: dense builds are
dramatically faster and more reliable, at the cost of a small, disclosed
retrieval-quality regression on two of three tested languages (below). Opt out
with `SEMANTEX_CINDER=0` or `SEMANTEX_CINDER=false` to restore the previous
contextual-encoder build path.

### Changed

- **Cinder is now the default dense-indexing path** (`SEMANTEX_CINDER` is
  default-on; it was opt-in before). Cinder embeds each token from a static
  table plus a distilled micro-mixer instead of running a full contextual
  encoder at build time, so a dense index builds in seconds to tens of seconds
  with no model inference. The previous default — the full contextual encoder —
  **cannot complete an index at all on very large repos** (it OOMs); Cinder
  finishes the same build. That changed the real-world baseline and is what
  motivated flipping the default.
- **Opt-out values.** Set `SEMANTEX_CINDER=0` or `SEMANTEX_CINDER=false`
  (case-insensitive) to disable Cinder and fall back to the previous build path.
  Only those two tokens are recognized — `off`/`no`/`disabled` do **not** turn
  it off (absent, or any other value, means on).

### Known tradeoffs

- **Retrieval quality regresses slightly on two of three tested languages,
  measured against the previous default (full contextual encoder), not
  against an internal target.** CodeSearchNet hybrid nDCG@10: Python improves
  (+1.1% relative — Cinder is actually better here), JavaScript regresses
  ~3.8% relative, Go regresses ~2.8% relative (retains 97.2% of the previous
  default's quality). This is an inherent property of encoder-free indexing —
  approximating what a real neural encoder would produce via a distilled
  micro-mixer over static lookups, instead of running it — not a bug. A
  quality-improvement fast-follow is planned (see
  `results/cinder-gate/report.md`).
- **Aggressive build targets not fully met at extreme scale.** At ~150k+ chunks
  Cinder's own internal build-speed / peak-memory targets aren't fully reached,
  though the build still completes successfully — and still far faster than the
  previous default, which cannot complete builds at that scale at all.

### Added

- **Ember/Cinder artifact distribution via GitHub release assets** — the static
  table, micro-mixer, and shortlist artifacts Cinder needs at build time are now
  published as release assets and fetched on demand, so a fresh install gets a
  working default indexing path instead of Cinder only working on a machine that
  trained the artifacts locally.

## v1.0.3 — 2026-07-12

`semantex_history` gains upstream-staleness and cross-branch visibility.
No retrieval-quality or index-schema changes — safe to upgrade in place.

### Added

- **Upstream staleness detection** — list-mode `semantex_history` calls now
  always report the local clone's ahead/behind count against its `@{u}`
  tracking branch (`upstream` in structured output), with a `[local clone
  is N commits behind ... — git pull for upstream-current state]` note in
  the text when behind > 0. `None` for a detached HEAD or a branch with no
  upstream configured.
- **`other_branches` param (opt-in)** — list mode only; surfaces other
  local and remote-tracking branches (excluding the current branch and its
  own upstream), most recently active first, capped at 10. Reports how
  many were skipped beyond the cap, and says so explicitly when none are
  found.
- **Worktree + nested-path dedup in `scope=all`** — cross-repo federation
  now excludes registered worktree checkouts and dedups nested registry
  paths (e.g. a subdirectory of an already-selected repo), so `scope=all`
  no longer renders the same commit history twice under two names. Naming
  a worktree or nested path explicitly via a named scope is unaffected.

### Fixed

- `scope=all`'s nested-path dedup silently reordered output sections by
  path length instead of preserving registry order.
- `other_branches` silently dropped branches beyond its 10-branch cap with
  no indication; now reports a skipped count in both text and structured
  output.
- `other_branches: true` with zero results rendered no text at all,
  indistinguishable from not having been requested; now says "No other
  active branches."
- Dropped dead `Serialize` derives on two internal history types whose
  JSON was always hand-built.
- `upsert_branch_at` computed a repo's worktree status twice on first
  registration.

## v1.0.2 — 2026-07-11

Git history becomes a first-class MCP surface. No retrieval-quality or
index-schema changes — safe to upgrade in place.

### Added

- **`semantex_history` MCP tool** — first-class git-history queries, in the
  default toolset (now 11 tools):
  - **List mode** — filter commits by `since` (a tag, sha, or `YYYY-MM-DD`
    UTC date; any rev `git rev-parse` accepts), `author` (substring),
    `file` (exact path), message `query` (FTS5 full-text), and `limit`
    (default 20). Filters compose. When `since` predates the captured
    history window, the response says so and names the
    `SEMANTEX_HISTORY_COMMITS` knob (default 500).
  - **Detail mode** — pass `commits: [shas]` (max 10 per call; excess
    reported as skipped) to expand each sha with full message, changed
    files, `--stat`, and a budget-bounded patch fetched live via
    `git show`. The total `budget` (default `SEMANTEX_MCP_BUDGET`) is
    split across the requested shas with a ~1000-byte floor per commit;
    truncation is UTF-8-boundary-safe and flagged.
  - **Cross-repo** — `scope: "all"` or a list of project names renders one
    `## [project]` section per registered repo, commits time-ordered
    within each section (no cross-repo interleaving). Unmatched names and
    per-project errors land in `skipped`, never silently dropped; a sha
    requested in detail mode degrades to a per-commit error line in repos
    that don't contain it.
  - History is refreshed incrementally from git on every call, so results
    are always current. Works with or without a search index, and never
    takes the index build lock.
  - Input hardening: `since` and sha arguments are validated before ever
    reaching git (leading-dash rejection, 7-40-hex-char shas), and all
    `git log`/`git show` output is parsed via NUL separators.

### Changed

- `--toolset all` now exposes **11 tools** (was 10); the `core` and
  `structural` bundles are unchanged. MCP server instructions, the plugin
  skill, and `semantex skills` tool metadata all document the new tool.
- `server.json` (MCP registry manifest) now advertises the complete tool
  list — added the previously missing `semantex_docs_context`,
  `semantex_memory_save`, `semantex_memory_recall`, and the new
  `semantex_history`.

## v1.0.1 — 2026-07-10

Incremental hardening + broader file-type coverage. No retrieval-quality or
index-schema changes — safe to upgrade in place; the v1.0.0 benchmark numbers
still hold.

### Added

- **Config-as-code AST chunking** — 9 new AST-chunked formats: Terraform/HCL,
  Bash, PowerShell, Protobuf, GraphQL, Starlark/Bazel, CMake, Groovy, and INI.
  Grammar choices and name-extraction logic ported from `lightonai/colgrep`
  (Apache-2.0) where no generic identifier field existed for a grammar.
  `CMakeLists.txt` and bare `BUILD`/`WORKSPACE`/`MODULE.bazel` filenames are
  now detected without needing an extension.

### Changed

- **next-plaid 1.3.1 → 1.6.2** — inherits upstream's incremental-delete
  performance work (IVF-patch delete, batched deletes, crash-safe atomic
  writes) for the `colbert-plaid` dense backend. Verified the exact
  `Colbert`/`MmapIndex` API surface this repo calls is unchanged across the
  range; re-applied the vendored k-means memory patch (26 GB → 9 GB peak RSS)
  on top of the new source.
- **`coderank-hnsw` is now genuinely incremental on disk** — replaced the
  single `vectors.bin` postcard blob with a split format: `index.bin` (ids,
  scales, tombstone flags; atomic temp+rename) plus an append-only
  `store.vecs` payload file. Insert appends only the new rows; delete only
  flips a tombstone flag — neither reloads nor rewrites the existing vector
  payload (previously O(index size) per mutation regardless of change size).
  `Int8VectorStore` gained tombstone-aware deletion so brute-force search and
  the HNSW-vs-brute-force threshold both respect deleted rows.
- **`colbert-plaid` mapping writes are now atomic** — `plaid_mapping.bin` is
  written via temp-file+rename on build/insert/delete, matching the existing
  `dense_backend::write_active_pointer` pattern; a crash mid-write no longer
  leaves a torn file the next open fails to deserialize.

### Fixed

- **PDF chunks were orphaned on incremental reindex.** The PDF chunker needs
  the absolute file path to open the file itself (unlike AST/text chunkers,
  which receive pre-read content + the repo-relative path), but that same
  absolute path was being stamped onto every resulting `Chunk::file_path`.
  Chunk-store lookups, deletion, and BM25 doc IDs all key on the relative
  path, so a changed or deleted PDF left its old chunks behind forever
  instead of being cleaned up on the next index.
- **Homebrew tap updates were non-idempotent.** The release workflow replaced
  literal placeholder tokens with real SHA-256 hashes via `sed`, which only
  matches once — every release after the first silently no-op'd, leaving the
  tap's pinned checksums stale while the actual release assets kept changing
  underneath them (`brew install` would then fail checksum verification).
  Replaced with a structural (position-based) match that's idempotent on
  every run.
- A latent off-by-one in the shared AST span→`end_line` conversion, exposed
  by INI's EOF-terminated `section` nodes (a definition ending exactly at a
  row boundary with no trailing content was counted as one line too long).
  No-op for every previously-supported grammar, whose definition nodes all
  end mid-line.
- GraphQL `operation_definition` name extraction was hardcoding the literal
  string `"operation"` instead of reading the actual `query`/`mutation`/
  `subscription` keyword and optional name — a named query like
  `query GetUser {...}` extracted as `"operation GetUser"` instead of
  `"query GetUser"`.

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
