# semantex — Semantic Code Search for AI Agents

**Cut your agent's cost by 65%, context burden by 39%, and tool calls by 51% — while *improving* answer quality.**

semantex is a fully local semantic code search MCP server that replaces the Grep→Read→Grep→Read loop AI agents fall into when exploring a codebase — the industry-default retrieval pattern, whose cognitive cost compounds quadratically with every extra tool call (see below). It combines ColBERT dense embeddings with BM25 sparse search to find code by meaning, then delivers pre-digested answers so your agent can act on the first call instead of the tenth.

As of the **v1.0.0** release, four things separate semantex from the rest of the semantic code search field:

- **The only active tool doing local ColBERT late-interaction retrieval.** Most semantic code search tools use single-vector embeddings; semantex uses per-token MaxSim scoring (ColBERT/PLAID) — the retrieval method the field has converged on for precision — and runs it entirely on-device.
- **The most self-contained implementation of that convergent architecture.** Single binary, no vector database, no embedding API, no API keys to provision or rotate. Your code and your queries never leave the machine.
- **Adoption machinery, not just an engine.** In-place, idempotent installers — MCP registration plus hooks/rules/skill files, written directly to each platform's real config path, no copy-paste — for **9 agent platforms** (Claude Code, Cursor, Codex, Aider, Gemini CLI, GitHub Copilot, OpenCode, Devin Desktop/Windsurf, Trae) route agents to semantex automatically instead of relying on the agent to remember it exists.
- **A codebase-understanding platform, not just a search box.** Multi-branch indexing, cross-repo federated search, searchable git history, durable project memory, and deterministic doc scaffolding all ship in v1.0 — see [What's New](#whats-new-in-v10).

See [How semantex compares](#how-semantex-compares) for an honest look at where it's ahead and where it isn't.

### Quick Start

Two commands, whichever coding tool you use:

```bash
curl -fsSL https://raw.githubusercontent.com/MisterTK/semantex/main/install.sh | sh
semantex install-claude-code   # or install-cursor / install-trae / install-copilot / install-gemini / install-devin-desktop / install-aider / install-codex / install-open-code
```

Each `install-*` command registers the MCP server **and** writes the platform's rules/instructions/skill file directly to its real, in-place config location — nothing to copy-paste by hand. All idempotent (safe to re-run) and reversible (`semantex uninstall-*`).

| Platform | Command | What it registers |
|---|---|---|
| **Claude Code** | `semantex install-claude-code` | MCP server + hooks (Grep/Read/Bash nudges) + skill |
| **Cursor** | `semantex install-cursor` | `.cursor/mcp.json` + `.cursor/rules/semantex.mdc` |
| **Trae** | `semantex install-trae` | `.trae/mcp.json` + `.trae/skills/semantex/SKILL.md` |
| **GitHub Copilot** (VS Code + CLI) | `semantex install-copilot` | `.vscode/mcp.json` + `~/.copilot/mcp-config.json` + `.github/copilot-instructions.md` |
| **Gemini CLI** | `semantex install-gemini` | `.gemini/settings.json` + `GEMINI.md` |
| **Devin Desktop** (formerly Windsurf) | `semantex install-devin-desktop` | `~/.codeium/windsurf/mcp_config.json` + `.devin/rules/semantex.md` |
| **Aider** | `semantex install-aider` | `.aider.conf.yml` + `.aider/semantex.md` (Aider has no native MCP support) |
| **Codex** | `semantex install-codex` | `~/.codex/config.toml` + `AGENTS.md` |
| **OpenCode** | `semantex install-open-code` | `opencode.json` + `.opencode/semantex.md` |

For any other MCP-capable editor without a dedicated `install-*` command, point it at:

```json
{
  "mcpServers": {
    "semantex": {
      "command": "semantex",
      "args": ["mcp"]
    }
  }
}
```

| Editor | Where to add it |
|--------|----------------|
| **Cline** | VS Code settings → Cline MCP Servers |
| **Continue** | `~/.continue/config.json` under `mcpServers` |

No manual indexing needed — semantex auto-indexes on first search.

---

## What's New in v1.0

v1.0 turns semantex from a search engine into a codebase-understanding platform. Everything below is real, shipped, and covered by tests — not a roadmap.

- **Multi-branch indexing.** Branch switches are first-class: each branch gets its own index snapshot, restored and incrementally updated on checkout, so flipping back and forth re-embeds only what actually changed. Detected automatically at every entry point (CLI, `watch`, `serve`, MCP), git-worktree-aware. Snapshot retention is capped (`SEMANTEX_MAX_BRANCH_INDEXES`, default 5).
- **Cross-repo federated search.** Pass `scope: "all"` (every project in your local registry) or a list of project names to `semantex_agent`/`semantex_search`, or `--scope` on the CLI, and one query fans out across every indexed repo with provenance-tagged, RRF-fused results. Repos that are stale or not ready are reported as skipped, never silently dropped.
- **Git-history indexing.** Commit history is now a searchable dimension of the index — bounded incremental `git log` population (500 commits by default, `SEMANTEX_HISTORY_COMMITS`), opt-in blame (`SEMANTEX_HISTORY_BLAME=1`), full-text commit-message search, `semantex history <file>` on the CLI, a first-class `semantex_history` MCP tool (v1.0.2: list/filter commits, expand shas with budget-bounded diffs, cross-repo `scope`), and an `include_history` flag on agent queries that appends a "recent changes" section to the answer.
- **Durable project memory.** `semantex_memory_save` / `semantex_memory_recall` let an agent persist decisions, gotchas, and conventions across sessions in `.semantex/memory.db` — independent of the code index, so it survives reindexing and branch switches.
- **Deterministic docs scaffolding.** `semantex_docs_context` returns a structurally-complete scaffold (symbol inventory, call-graph edges, import edges, existing doc comments, file:line provenance) for a module or the whole repo. It does **not** call an LLM — it hands your agent the facts, and the `semantex-docs` skill guides it to write maintained prose into `semantex_docs/`.
- **Multi-client, branch-aware daemon.** `semantex serve` is now thread-per-connection with graceful drain, a searcher cache keyed by `(project, branch)`, and staleness reload when the index is rebuilt out from under it — safe for several editors/agents to share one daemon on one machine.
- **MCP HTTP transport with bearer-token auth.** `semantex mcp --http` can now be exposed beyond localhost (`--allow-remote`) only with a required bearer token (`--auth-token` / `SEMANTEX_HTTP_TOKEN`, or an auto-generated one persisted at `~/.semantex/http_token`), compared in constant time.
- **Real installers for 9 agent platforms**, not just Claude Code — Cursor, Trae, GitHub Copilot, Gemini CLI, Devin Desktop/Windsurf, Aider, Codex, and OpenCode all get in-place, idempotent, reversible `install-*`/`uninstall-*` commands (see the Quick Start table).
- **SQL AST chunking.** SQL is now parsed with real tree-sitter grammar rather than falling back to text chunking, bringing AST-aware language support to 25 languages.
- **`semantex agent` CLI subcommand.** The same auto-routing intelligence behind the `semantex_agent` MCP tool, available as a standalone command for scripts, CI, and non-MCP agents — see [Standalone CLI](#standalone-cli).

## The Problem: Agents Waste Most of Their Tokens Searching

When an AI coding agent needs to understand a codebase, it falls into a predictable pattern:

```
Grep "authentication" → 15 files match
Read file1.rs (lines 1-200) → not quite right
Grep "verify.*token" → 8 files match
Read file2.rs (lines 50-150) → getting warmer
Read file3.rs (lines 1-300) → found it, but need callers
Grep "calls verify_token" → ...
```

Each tool call adds content to the context window that **never goes away**. The agent re-reads this growing history on every subsequent turn. The cost isn't linear — it's **quadratic**:

```
Context at turn k:  Cₖ = B + δ₁ + δ₂ + ... + δₖ₋₁     (monotonically growing)

Total cognitive load: CCB = Σ Cₖ ≈ N·B + δ·N·(N-1)/2   ← O(N²) in tool calls
```

An agent making 20 tool calls doesn't process 20× the content of one call — it processes closer to **200×**, because each call resends the entire accumulated history. And as context grows, attention degrades: information from early tool returns gets buried in the middle of a 70K+ token context, exactly where LLMs attend worst.

**The result**: agents spend 60-80% of their tokens on search overhead, not on reasoning about your actual question.

## The Solution: One Call, Complete Answer

semantex collapses the search loop into a single tool call:

```
Agent → semantex_agent "how does authentication work"
     ← Prose answer + source references           (1 call, done)
```

`semantex_agent` classifies the query — semantic, exact symbol, graph walk, deep synthesis, regex, or file glob — dispatches to the right internal strategy with fallbacks, and returns a ready-to-use answer. Instead of the agent iteratively searching and reading, semantex does it internally — hybrid search, graph expansion, content reading, and extractive summarization — and returns a pre-digested answer the agent can immediately act on.

### Measured Impact (v1.0.0 engine — committed, reproducible runs)

**Agent efficiency** — Claude Code agents answering 5 real-world questions on `pallets/flask`, built-in Grep/Glob/Read vs semantex, blind-judged:

| Metric | With semantex | Without (Grep/Glob/Read) | Improvement |
|---|---|---|---|
| **Cost (USD)** | $0.37 | $1.05 | **−65%** |
| **Cumulative context burden** | 937K | 1.53M | **−39%** |
| **Tool calls** | 15.0 | 30.4 | **−51%** |
| **Peak context size** | 53K | 84K | **−36%** |
| **Wall-clock time** | 116s | 254s | **−54%** |
| **Answer quality** (blind judge, 1–5) | **4.80** | 3.80 | **higher, not tied** |

**Retrieval quality** — SWE-bench-Verified file localization (find the file a real issue's fix touched), k=10:

| | semantex (agent-routed) | semantex (hybrid) | ripgrep |
|---|---|---|---|
| **Acc@1** | **3/5** | 2/5 | 0/5 |
| **MRR@10** | **0.600** | 0.522 | 0.095 |

The cost saving is the billing metric. The **cumulative context burden** is the cognitive one — how much the model actually had to attend to across all turns. Fewer turns × smaller context = quadratically less waste.

> Self-reported benchmarks, run by the semantex team on CPU-only hardware with small question sets — read them as directional, not definitive. Full methodology, per-question data, raw counts, and honest caveats: [`benchmarks/RESULTS-v1.md`](benchmarks/RESULTS-v1.md). Both suites are committed and reproducible from this repo.

## How semantex compares

No retrieval tool is right for every codebase or team. Here's how semantex compares to the tools agent users are most likely to already have installed, as of mid-2026:

| | semantex | Serena | claude-context (Zilliz) | Grep-only (the default) | Cloud context platforms (e.g. Augment) |
|---|---|---|---|---|---|
| **Retrieval type** | ColBERT late-interaction + BM25 hybrid, fully local | LSP symbol lookup — no conceptual/semantic search | Dense vector + BM25 hybrid | Regex / keyword only | Proprietary semantic indexing, server-side |
| **Infra required** | None — single binary | None — manages per-project language servers | Milvus (self-hosted) or Zilliz Cloud | None | None (fully hosted by vendor) |
| **API keys** | None | None | Embedding API key (OpenAI, Voyage, Gemini) unless using local Ollama | None | Vendor account |
| **Privacy** | Code never leaves the machine | Code never leaves the machine | Code chunks sent to the embedding API (unless using local Ollama); vectors stored in Milvus/Zilliz Cloud | Code never leaves the machine | Code indexed on vendor infrastructure |
| **Price** | Free, open source (Apache-2.0) | Free, open source (MIT) | Free, open source; embedding API usage + optional Zilliz Cloud costs extra | Free (built into the agent) | $20–$200+/seat/month; enterprise custom |

Where semantex is behind, honestly:

- **Symbol precision.** Serena's LSP backend gives exact go-to-definition/rename/refactor semantics across dozens of languages via real language servers. semantex's chunk-level retrieval finds the right region of code but doesn't do LSP-grade symbol operations — the two are complementary rather than competing, and some users run both.
- **Mindshare and ecosystem breadth.** claude-context/Zilliz is better known and already integrates with more agent clients. semantex's own platform coverage (9 platforms, above) is comparable but newer and less established.
- **Below roughly 1,000 files, plain grep is often enough.** semantex's advantage grows with codebase size and query ambiguity; on small, well-organized repos the overhead of maintaining an index may not pay for itself.
- **Team-shared indexing.** semantex indexes per-machine (with first-class multi-branch switching and cross-repo federated search); cloud platforms offer team-shared indexes maintained server-side. A shared daemon covers the single-machine multi-client case, but there's no cross-machine index sync.

## How It Works

### For the Agent: One Tool, Auto-Routed

| Question type | Tool | What happens |
|---|---|---|
| *Anything* | `semantex_agent` | Classifies the query and dispatches: hybrid search, exact symbol lookup, callers/callees/imports graph walk, deep multi-source synthesis, regex, or file glob. One call, complete answer. |

`semantex_agent` is the recommended entry point for essentially every code-search need — don't chain multiple `semantex_*` calls to assemble an answer; if a result is incomplete, refine the question and call it again. Power users who want structured JSON or explicit control over one stage of the pipeline can still call `semantex_search` (ranked hits) or `semantex_deep` (search→triage→graph-expand→read→summarize) directly.

### Under the Hood: Hybrid Search Pipeline

```
Query → Classification → Expansion (stemmed synonyms)
                              |
Query → ColBERT (48d) → PLAID MaxSim ──────┐
                                            ├→ Triple CC Fusion → Results
Query → Stemmed BM25 (Tantivy) ────────────┤
                                            │
Query → Exact string match ─────────────────┘
```

1. **Dense search** — ColBERT per-token embeddings with late interaction (MaxSim scoring via PLAID index)
2. **Sparse search** — BM25 via Tantivy with Snowball stemming and code-aware tokenization
3. **Fusion** — query-adaptive convex combination of dense, sparse, and exact match scores
4. **Deep pipeline** — search → triage → call graph expansion → content read → extractive summary

### Fully Local & Private

- All processing runs locally — your code never leaves your machine
- No cloud APIs, no internet required after initial model download (~17MB)
- Single ~17MB int8 ONNX model (ColBERT LateOn-Code-edge), cached and shared across projects
- Anonymous usage telemetry (command name, OS, arch) is opt-out: `export SEMANTEX_NO_TELEMETRY=1`

## Installation

**macOS / Linux** — one command:

```bash
curl -fsSL https://raw.githubusercontent.com/MisterTK/semantex/main/install.sh | sh
```

**Windows** — download from [GitHub Releases](https://github.com/MisterTK/semantex/releases/latest).

**Build from source** (Rust 1.91+):

```bash
git clone https://github.com/MisterTK/semantex.git
cd semantex && cargo install --path crates/semantex-cli
```

## Setup

semantex is an [MCP server](https://modelcontextprotocol.io/) (MCP 2025-03-26). Use the `install-*` command for your platform (registers the MCP server + writes its rules/instructions file, all in place — see the [Quick Start](#quick-start) table), or add it to your editor's MCP configuration by hand:

```json
{
  "mcpServers": {
    "semantex": {
      "command": "semantex",
      "args": ["mcp"]
    }
  }
}
```

| Editor | Config file |
|--------|-------------|
| **Claude Code** | `.mcp.json` in project root, or `~/.claude/settings.json` for global (`semantex install-claude-code`) |
| **Cursor** | `.cursor/mcp.json` in project root (`semantex install-cursor`) |
| **Devin Desktop** (formerly Windsurf) | `~/.codeium/windsurf/mcp_config.json` (`semantex install-devin-desktop`) |
| **Trae** | `.trae/mcp.json` (`semantex install-trae`) |
| **GitHub Copilot** | `.vscode/mcp.json` (VS Code) / `~/.copilot/mcp-config.json` (CLI) (`semantex install-copilot`) |
| **Gemini CLI** | `.gemini/settings.json` (`semantex install-gemini`) |
| **Aider** | `.aider.conf.yml` (`semantex install-aider`; no native MCP support, config-based) |
| **Codex** | `~/.codex/config.toml` (`semantex install-codex`) |
| **OpenCode** | `opencode.json` (`semantex install-open-code`) |
| **Cline** | VS Code settings → Cline MCP Servers |
| **Continue** | `~/.continue/config.json` under `mcpServers` |

That's it. semantex auto-indexes your project on first search — no manual indexing step required.

### Tools

By default, semantex exposes **11 MCP tools**, discovered by your editor at startup:

| Tool | Purpose |
|------|---------|
| `semantex_agent` | **The recommended tool for almost everything.** Auto-classifies the query and routes to the right strategy — semantic, exact symbol, graph walk, deep synthesis, regex, or file glob. Supports `mode` overrides, multi-query batching, response-size `budget`, cross-repo `scope`, and `include_history`. |
| `semantex_search` | Simple lookups: find definitions, list references, locate files. Ranked hits with file, lines, score, snippet. |
| `semantex_deep` | Complex questions: search → triage → graph-expand → read → summarize into one prose answer with sources. |
| `semantex_index` | Force a rebuild of the index (rarely needed — semantex auto-indexes on first search). |
| `semantex_status` | Check index state: file count, chunk count, freshness. |
| `semantex_health` | System health check (model availability, configuration). |
| `semantex_validate` | Consistency checks: meta-DB sync, stale files, dense/sparse index integrity, graph consistency. |
| `semantex_docs_context` | Deterministic documentation scaffold (symbol inventory, call graph, imports, provenance) for your agent to turn into `semantex_docs/*.md` prose. Zero LLM calls inside semantex itself. |
| `semantex_memory_save` | Persist a durable note (decision, gotcha, convention) to `.semantex/memory.db`, independent of the code index. |
| `semantex_memory_recall` | Recall previously saved notes, ranked by relevance or most-recent. |
| `semantex_history` | Git history, first-class: list/filter commits (`since` a tag, sha, or YYYY-MM-DD date; `author`; `file`; message `query`) or expand specific shas with full message, `--stat`, and a budget-bounded patch. Works cross-repo via `scope`, with or without a search index — refreshed incrementally from git on every call. |

A **5-tool `structural` bundle** — `semantex_symbol`, `semantex_callers`, `semantex_callees`, `semantex_implementations`, `semantex_architecture` — is available opt-in via `semantex mcp --toolset structural` (or `--toolset all` isn't needed; `semantex_agent`'s internal structural route already covers callers/callees/imports in one call). These are deliberately **not** in the default bundle: an earlier version exposed them by default and agents used them additively (calling 4-6 tools to reassemble what `semantex_deep` already returns in one call), measurably regressing efficiency. Use `--toolset core` for a minimal 3-tool surface (`semantex_search`, `semantex_deep`, `semantex_agent`) when a client's tool budget is tight.

### MCP Features

- **Server instructions** — at initialization, semantex sends routing guidance to the LLM: prefer `semantex_agent` for almost everything
- **Structured output** — tools return machine-readable JSON in `structuredContent` alongside text, so editors can render results natively
- **Progress notifications** — `semantex_deep` sends phase-by-phase progress (searching, triaging, graph-expanding, reading, summarizing)
- **Auto-indexing** — first search triggers background indexing; returns keyword (ripgrep) results immediately while the index builds
- **Logging** — tool events emitted as MCP log notifications; clients can set level via `logging/setLevel`
- **Toolset bundles** — `core` (3 tools), `structural` (5 tools, opt-in), `all` (11 tools, default) via `--toolset`
- **HTTP transport with auth** — `semantex mcp --http` for network access; `--allow-remote` requires a bearer token (`--auth-token` or `SEMANTEX_HTTP_TOKEN`, else auto-generated and persisted at `~/.semantex/http_token`), checked in constant time
- **Multi-client daemon** — one `semantex serve` process safely answers concurrent editors/agents, with a per-`(project, branch)` searcher cache and automatic reload if the index changes underneath it

## Standalone CLI

semantex also works as a standalone CLI tool for terminal use, scripts, and CI.

### Intelligent routing (recommended)

```bash
# Auto-classified query — same routing logic as the MCP semantex_agent tool
semantex agent "how does authentication work"

# Force a retrieval route instead of auto-classifying
semantex agent "ConnectionFactory" --route exact_symbol

# Full source code blocks, larger response budget
semantex agent "how does the daemon reconcile branches" --full --budget 20000

# Fan out across every registered project
semantex agent "rate limiting" --scope all
```

Routes: `semantic`, `deep`, `exact_symbol`, `structural`, `regex`, `analytical`, `exhaustive`, `file_pattern`, `architecture`, `feature_planning`.

### Direct search flags

```bash
# Search semantically (auto-indexes on first run)
semantex "authentication logic" /path/to/project

# Deep search — complete answer in one call
semantex --deep "how does the auth flow work"

# Compact references with signatures and metadata
semantex --refs "database connection pool"

# Call graph for a symbol (callers, callees, type refs)
semantex --around handle_request

# Code preview — refs + first 5 lines per result
semantex --peek "error handling middleware"

# Batch multiple queries in one call
semantex --refs "auth flow" "token validation" "session management"

# Fan out across every registered project (or --scope name1,name2)
semantex "rate limiting" --scope all
```

### More CLI Options

```bash
# Retrieval strategies
semantex --dense-only "user verification"     # ColBERT only (semantic)
semantex --sparse-only "authenticate"          # BM25 only (keyword)
semantex -G "ConnectionFactory"               # Grep parity (exact + BM25)
semantex --rerank "complex query"             # Cross-encoder reranking
semantex -e "async fn" "database pool"        # Regex + semantic hybrid

# Output formats
semantex -v "query"                 # Verbose: full content, ANSI colors
semantex -g "query"                 # Grep-like: one line per result
semantex --json "query"             # JSON array
semantex --json --no-content "q"    # JSON, metadata only

# Filtering
semantex -t rs "error handling"     # Only Rust files
semantex -t ts -t js "API endpoint" # Multiple types
semantex --code-only "factory"      # Exclude docs/config
semantex -m 20 "database"           # Max 20 results

# Daemon
semantex serve /path/to/project     # Start daemon (10ms warm search, multi-client)
semantex stop /path/to/project      # Stop daemon
semantex connect /path/to/project   # Persistent client (binary protocol, lower latency)
semantex disconnect                 # Stop persistent client
semantex watch /path/to/project     # Auto-reindex on file changes
```

### Repo intelligence

```bash
semantex history <file>             # Git history for a file (author, age, subject; --limit N)
semantex validate                   # Index consistency check (meta-DB, stale files, dense/sparse, graph)
semantex status                     # Index state: file/chunk counts, freshness
semantex download-models            # Pre-fetch the ONNX models (otherwise fetched on first use)
semantex skills-generate            # Emit per-platform skill files from the canonical tool registry
```

### MCP server

```bash
semantex mcp                                  # stdio transport (default)
semantex mcp --http --port 5050               # HTTP transport, loopback only
semantex mcp --http --toolset core            # Minimal 3-tool bundle
semantex mcp --http --allow-remote --auth-token <TOKEN>  # network-exposed, auth required
```

## Supported Languages

29 file types, including 25 with tree-sitter AST-aware chunking:

- **AST-parsed (25):** Rust, Python, JavaScript, TypeScript, Go, Java, C, C++, Ruby, PHP, C#, Dart, Scala, Kotlin, Swift, Elixir, Lua, Haskell, OCaml, Zig, R, HTML, Svelte, Vue, SQL
- **Text-chunked (4):** Markdown, JSON, TOML, YAML
- Fallback text chunking for any other file type

## Architecture

### Deep Search Pipeline

```
Agent → semantex_agent "query"
          │
          ├─ Classify (semantic / exact symbol / structural / deep / regex / glob)
          ├─ Phase 1: Hybrid search (top-20 candidates)
          ├─ Phase 2: Triage (dedup, per-file cap, kind preference → top-8)
          ├─ Phase 3: Graph expansion (callers/callees/types, 1 hop → up to 12)
          ├─ Phase 4: Read full content (32KB budget)
          └─ Phase 5: Extractive summarize → prose answer + source refs
```

### Project Structure

```
semantex/
├── crates/
│   ├── semantex-core/      # Search engine (indexing, search, embeddings, daemon, history, memory)
│   ├── semantex-cli/       # Command-line interface
│   └── semantex-mcp/       # MCP server (stdio + HTTP transports)
├── benchmarks/              # Benchmark scripts and ground truth
└── docs/                    # Architecture docs and benchmark data
```

### System Requirements

- **Platform**: macOS (ARM/Intel), Linux, Windows
- **Memory**: 2GB+ RAM recommended
- **Disk**: ~100MB for models + index storage
- **Build**: Rust 1.91+ (if building from source)

### First Index

The first `semantex index` (or first search / MCP call, which auto-indexes) on
a new project embeds every chunk of the codebase — CPU-bound, one-time work.
Measured on a 16 GB / 4-core box: **314 files (~5 MB source, 4669 chunks)
took ~11.4 minutes (684.8s)** end to end. Indexing scales with source size
and available cores, not repo file count alone; expect low-single-digit
minutes for small projects and longer for very large monorepos. Subsequent
runs are incremental (only changed files re-embed) and are much faster.
Peak memory during that same full build was ~9.6 GB RSS — comfortably
inside a 16 GB machine's budget; semantex enforces a hard cap (see
`SEMANTEX_MAX_RSS_MB` below) so a build can never take down the host, even
on smaller machines. Switching branches doesn't repeat this cost: each
branch keeps its own snapshot, restored and incrementally updated on
checkout.

## Development

```bash
cargo build --release          # Build
cargo test --all               # Test
cargo clippy --all             # Lint
```

### Environment Variables

The most commonly-needed knobs:

```bash
# Resources
SEMANTEX_ORT_THREADS=4           # ONNX Runtime threads for QUERIES (default: 4)
SEMANTEX_INDEX_ORT_THREADS=8     # ONNX Runtime threads for a full INDEX BUILD
                                  # (default: half the cores on a box that can run
                                  # several full builds at once, ALL cores when it
                                  # can only run one — clamped to [2, 8] either way)
SEMANTEX_COREML=1                # CoreML acceleration on macOS
SEMANTEX_MAX_RSS_MB=8192          # Soft RSS budget in MB — polled at build/search
                                  # checkpoints (default: 50% of system RAM,
                                  # clamped to [1024, 12288]; `0` disables it)
SEMANTEX_NO_RLIMIT=1              # Skip the kernel setrlimit(RLIMIT_AS) failsafe
                                  # (Linux only; useful in containers with their
                                  # own cgroup memory limits)
SEMANTEX_MAX_CONCURRENT_BUILDS=2  # Max full index builds allowed to run at once
                                  # across all projects (default: system RAM / 16 GB, min 1)

# MCP response shaping (semantex_agent)
SEMANTEX_MCP_BUDGET=12000         # Default response size budget in bytes (~3K tokens)
SEMANTEX_MCP_FULL_CODE=0          # Include full source blocks by default (0/1)
SEMANTEX_MCP_DEPTH=search         # Default depth when a query doesn't specify one

# Git history (v1.0)
SEMANTEX_HISTORY_COMMITS=500      # Commits to index per repo (bounded, incremental)
SEMANTEX_HISTORY_BLAME=1          # Opt-in blame population (off by default — slower)

# Multi-branch indexing (v1.0)
SEMANTEX_MAX_BRANCH_INDEXES=5     # Per-project branch snapshot retention cap

# MCP HTTP transport (v1.0)
SEMANTEX_HTTP_TOKEN=<token>       # Bearer token for --http --allow-remote (else auto-generated)

# Telemetry
SEMANTEX_NO_TELEMETRY=1           # Opt out of anonymous usage telemetry
```

Several other knobs exist for retrieval-tuning experiments (reranking, weighted RRF fusion, MMR diversity, graph centrality levers, adaptive result sizing, semantic query cache) — they ship **off by default** because the A/Bs that would justify flipping them on haven't shown a net win yet. See `crates/semantex-core/src/config.rs` and `docs/` for the full list if you want to experiment.

## License

Apache-2.0

## Credits

Built with [next-plaid](https://github.com/lightonai/next-plaid) (ColBERT/PLAID), [Tantivy](https://github.com/quickwit-oss/tantivy) (BM25), [fastembed-rs](https://github.com/Anush008/fastembed-rs) (reranking), [tree-sitter](https://tree-sitter.github.io/) (AST parsing), [ONNX Runtime](https://onnxruntime.ai/) (inference).
