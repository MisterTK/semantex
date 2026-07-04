# semantex — Semantic Code Search for AI Agents

**Cut your agent's token usage by 40%, context burden by 67%, and tool calls by 55% — with zero quality loss.**

semantex is a fully local semantic code search MCP server that replaces the Grep→Read→Grep→Read loop AI agents fall into when exploring a codebase — the industry-default retrieval pattern, whose cognitive cost compounds quadratically with every extra tool call (see below). It combines ColBERT dense embeddings with BM25 sparse search to find code by meaning, then delivers pre-digested answers so your agent can act on the first call instead of the tenth.

As of mid-2026, three things separate semantex from the rest of the semantic code search field:

- **The only active tool doing local ColBERT late-interaction retrieval.** Most semantic code search tools use single-vector embeddings; semantex uses per-token MaxSim scoring (ColBERT/PLAID) — the retrieval method the field has converged on for precision — and runs it entirely on-device.
- **The most self-contained implementation of that convergent architecture.** Single binary, no vector database, no embedding API, no API keys to provision or rotate. Your code and your queries never leave the machine.
- **Adoption machinery, not just an engine.** Hooks and generated skill files for 9 agent platforms (Claude Code, Cursor, Codex, Aider, Gemini CLI, Copilot CLI, OpenCode, Windsurf, Trae) route agents to semantex automatically instead of relying on the agent to remember it exists.

See [How semantex compares](#how-semantex-compares) for an honest look at where it's ahead and where it isn't.

### Quick Start

Install and add to your editor in under a minute:

```bash
# Install
curl -fsSL https://raw.githubusercontent.com/MisterTK/semantex/main/install.sh | sh

# Add to your editor's MCP config (Claude Code, Cursor, Windsurf, etc.)
```

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
| **Claude Code** | `.mcp.json` in project root, or `~/.claude/settings.json` globally |
| **Cursor** | `.cursor/mcp.json` in project root |
| **Windsurf** | `~/.codeium/windsurf/mcp_config.json` |
| **Cline** | VS Code settings → Cline MCP Servers |
| **Continue** | `~/.continue/config.json` under `mcpServers` |

No manual indexing needed — semantex auto-indexes on first search.

---

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
Agent → semantex_deep "how does authentication work"
     ← Prose answer + source references           (1 call, done)
```

Instead of the agent iteratively searching and reading, semantex does it internally — semantic search, graph expansion, content reading, and extractive summarization — and returns a pre-digested answer the agent can immediately act on.

### Measured Impact (Benchmark: 10 agents, 5 real-world tasks, Sonnet 4.6)

| Metric | With semantex | Without (Grep/Glob/Read) | Improvement |
|---|---|---|---|
| **Total tokens** | 212K | 355K | **-40%** |
| **Cumulative context burden** | 2.2M | 6.8M | **-67%** |
| **Tool calls** | 39 | 86 | **-55%** |
| **Peak context size** | 42K avg | 71K avg | **-40%** |
| **Wall-clock time** | 513s | 609s | **-16%** |
| **Answer quality** | Comprehensive | Comprehensive | **Tie** |

The -40% token saving is the billing metric. The **-67% cumulative context burden** is the cognitive metric — how much the model actually had to attend to across all turns. Fewer turns × smaller context = quadratically less waste.

> Self-reported benchmark, run by the semantex team. Full methodology and per-question data available in the benchmark suite (`benchmarks/`).

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
- **Mindshare and ecosystem breadth.** claude-context/Zilliz is better known and already integrates with more agent clients (Claude Code, Cursor, Codex CLI, Gemini CLI, Cline, Windsurf, Augment, and others). semantex's own platform coverage (9 platforms, above) is comparable but newer and less established.
- **Below roughly 1,000 files, plain grep is often enough.** semantex's advantage grows with codebase size and query ambiguity; on small, well-organized repos the overhead of maintaining an index may not pay for itself.
- **Multi-branch and team indexing.** semantex indexes one branch at a time per project today; cloud platforms and some hybrid setups already offer cross-branch and team-shared indexing.

## How It Works

### For the Agent: Two Tools, Simple Routing

| Question type | Tool | What happens |
|---|---|---|
| *"How does auth work?"* | `semantex_deep` | Search→triage→graph expand→read→summarize. One call, complete answer. |
| *"Find the Config struct"* | `semantex_search` | Ranked results with file, lines, score, snippet. |

`semantex_deep` replaces 5-10 Grep→Read iterations. `semantex_search` replaces Grep→Glob for lookups.

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

semantex is an [MCP server](https://modelcontextprotocol.io/) (MCP 2025-03-26). Add it to your editor's MCP configuration:

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
| **Claude Code** | `.mcp.json` in project root, or `~/.claude/settings.json` for global |
| **Cursor** | `.cursor/mcp.json` in project root |
| **Windsurf** | `~/.codeium/windsurf/mcp_config.json` |
| **Cline** | VS Code settings → Cline MCP Servers |
| **Continue** | `~/.continue/config.json` under `mcpServers` |
| **Codex** | `semantex install-codex` |
| **OpenCode** | `semantex install-open-code` |

That's it. semantex auto-indexes your project on first search — no manual indexing step required.

### Tools

5 tools, automatically discovered by your editor at startup:

| Tool | Purpose |
|------|---------|
| `semantex_deep` | **Complex questions.** One call replaces 5-10 grep+read iterations. Returns prose answer with sources. |
| `semantex_search` | Simple lookups: find definitions, list references, locate files. |
| `semantex_index` | Trigger or rebuild the index for a project. |
| `semantex_status` | Check index state (file count, chunk count, freshness). |
| `semantex_health` | System health check (model status, cache stats). |

All read-only tools are annotated with `readOnlyHint: true` so editors that support [MCP tool annotations](https://modelcontextprotocol.io/specification/2025-03-26/server/tools#annotations) can auto-approve them.

### MCP Features

- **Server instructions** — at initialization, semantex sends routing guidance to the LLM: use `deep` for complex questions, `search` for lookups
- **Structured output** — `semantex_search` and `semantex_deep` return machine-readable JSON in `structuredContent` alongside text, so editors can render results natively
- **Progress notifications** — `semantex_deep` sends phase-by-phase progress (searching, triaging, graph-expanding, reading, summarizing)
- **Auto-indexing** — first search triggers background indexing; returns keyword (ripgrep) results immediately while the index builds
- **Logging** — tool events emitted as MCP log notifications; clients can set level via `logging/setLevel`

## Standalone CLI

semantex also works as a standalone CLI tool for terminal use, scripts, and CI:

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
semantex serve /path/to/project     # Start daemon (10ms warm search)
semantex stop /path/to/project      # Stop daemon
semantex watch /path/to/project     # Auto-reindex on file changes
```

## Supported Languages

29 file types, including 25 with tree-sitter AST-aware chunking:

- **AST-parsed (25):** Rust, Python, JavaScript, TypeScript, Go, Java, C, C++, Ruby, PHP, C#, Dart, Scala, Kotlin, Swift, Elixir, Lua, Haskell, OCaml, Zig, R, HTML, Svelte, Vue, SQL
- **Text-chunked (4):** Markdown, JSON, TOML, YAML
- Fallback text chunking for any other file type

## Architecture

### Deep Search Pipeline

```
Agent → semantex_deep "query"
          │
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
│   ├── semantex-core/      # Search engine (indexing, search, embeddings, daemon)
│   ├── semantex-cli/       # Command-line interface
│   └── semantex-mcp/       # MCP server (5 tools over stdio)
├── benchmarks/              # Benchmark scripts and ground truth
└── docs/                    # Architecture docs and benchmark data
```

### System Requirements

- **Platform**: macOS (ARM/Intel), Linux, Windows
- **Memory**: 2GB+ RAM recommended
- **Disk**: ~100MB for models + index storage
- **Build**: Rust 1.91+ (if building from source)

## Development

```bash
cargo build --release          # Build
cargo test --all               # Test
cargo clippy --all             # Lint
```

### Environment Variables

```bash
SEMANTEX_ORT_THREADS=4         # ONNX Runtime threads (default: 4)
SEMANTEX_COREML=1              # CoreML acceleration on macOS
SEMANTEX_MAX_RSS_MB=2048       # Daemon RSS limit in MB (default: 2048)
```

## License

Apache-2.0

## Credits

Built with [next-plaid](https://github.com/lightonai/next-plaid) (ColBERT/PLAID), [Tantivy](https://github.com/quickwit-oss/tantivy) (BM25), [fastembed-rs](https://github.com/Anush008/fastembed-rs) (reranking), [tree-sitter](https://tree-sitter.github.io/) (AST parsing), [ONNX Runtime](https://onnxruntime.ai/) (inference).
