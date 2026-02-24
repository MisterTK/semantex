# sage — Semantic Code Search

**Hybrid dense+sparse code search with ColBERT late interaction and BM25**

sage is a fully local, production-grade semantic code search engine. It combines per-token neural embeddings (ColBERT/PLAID) with stemmed BM25 keyword search and intelligent fusion to find code by meaning, not just pattern matching.

## Why sage?

Traditional grep tools match exact patterns. sage understands **meaning**:

```bash
# Find authentication code, even if it doesn't contain "auth"
sage "verify user credentials" ./src

# Understand intent, not just keywords
sage "database connection pool initialization" .

# Search in natural language
sage "function that handles file uploads with progress tracking" .
```

## Key Features

### Hybrid Retrieval Architecture

sage uses a multi-stage search pipeline:

1. **Dense search** (ColBERT/PLAID) — per-token late interaction with MaxSim scoring (48d embeddings)
2. **Sparse search** (BM25 via Tantivy) — stemmed keyword matching with code-aware tokenization
3. **Triple CC fusion** — query-adaptive convex combination of dense, sparse, and exact scores
4. **File role boosting** — 11-class path heuristics (Service x1.15 to Test x0.70)

### Search Quality

- **ColBERT LateOn-Code-edge**: Per-token embeddings purpose-built for code, 48-dimensional via PLAID index
- **Snowball English stemmer**: BM25 pipeline stems both indexed content and queries for better NL recall
- **Stemmed synonym expansion**: Bridges natural language to code identifiers (e.g., "encrypting" matches "encrypt", "cipher", "KMS")
- **Query-adaptive fusion**: Auto-detects query type (identifier, keyword, semantic, mixed) and adjusts weights
- **Code-aware tokenizer**: Splits camelCase/snake_case, emits sub-tokens + joined forms

### Performance

- **16ms warm search** (daemon mode, no rerank)
- **4x faster than grep** on average
- **222x fewer tokens** in compact output mode (7 vs 1,555 tokens)
- **Identifier recall**: 0.90+ on exact keyword queries
- **F1 score**: 0.610 (beats grep's 0.454 on 30-query benchmark)

### Fully Local & Private

- **No cloud dependencies**: All processing runs locally on your machine
- **No telemetry**: Your code never leaves your device
- **Offline-first**: Works without internet after initial model download

### Agent & LLM Friendly

- **File-type filtering**: Scope searches with `-t/--type` flags (e.g., `-t rs`, `-t py`)
- **Compact output modes**: Token-efficient formats (`--grep`, `--no-content`, `--snippet`)
- **Unix domain socket daemon**: Persistent server for low-latency repeated queries
- **MCP server**: Model Context Protocol integration for AI coding assistants

## Getting Started

### Recommended: Claude Code Plugin

The fastest way to use sage is as a Claude Code plugin. This gives Claude semantic search as a built-in skill — it automatically uses sage instead of grep/glob for code exploration.

```bash
# 1. Install the sage binary
cargo install --path crates/sage-cli

# 2. Install the plugin into Claude Code
claude plugin install ./plugin

# 3. Index your project
sage index /path/to/your/project

# 4. Start using Claude Code — sage is now the default search tool
```

Once installed, Claude will automatically prefer `sage "query"` over Grep/Glob for conceptual searches, and `sage --grep "token"` for exact matches. The plugin also includes hooks that nudge Claude to use sage when it reaches for grep or glob.

### Manual CLI Usage

sage works as a standalone CLI tool without Claude Code:

```bash
# Install from source
cargo install --path crates/sage-cli

# Index your project
sage index /path/to/your/project

# Search semantically
sage "authentication logic" /path/to/your/project
```

### System Requirements

- **Rust**: 1.91 or later (edition 2024)
- **Platform**: macOS (ARM/Intel), Linux, Windows
- **Memory**: 2GB+ RAM recommended
- **Disk**: ~200MB for ColBERT model + index storage

### Model Downloads

On first index, sage automatically downloads the ColBERT model to `~/.sage/models/`:

- **ColBERT model**: LateOn-Code-edge (~200MB, 48d per-token embeddings)
- **Reranker model**: JINA Reranker v1 Turbo (~80MB, optional via `--rerank`)

Models are cached and shared across all sage instances.

## Usage

### Basic Search

```bash
# Search current directory
sage "handle user authentication"

# Search specific path
sage "database migration logic" ./backend/db

# Limit results
sage --max-count 5 "error handling"

# Show code snippets in results
sage --content "API endpoint for users"
```

### File-Type Filtering

```bash
# Search only Rust files
sage -t rs "error handling"

# Search TypeScript and JavaScript
sage -t ts -t js "API endpoint"

# Multiple types
sage -t rs -t py -t go "factory pattern"
```

### Compact Output Modes

```bash
# Grep-like one-line format (222x token reduction vs grep multi-iteration)
sage --grep "authentication"

# JSON output for programmatic use
sage --json "database"

# JSON without content (compact)
sage --json --no-content "error handling"
```

### Search Modes

```bash
# Default: Hybrid search (dense + sparse + fusion)
sage "authentication middleware"

# Dense-only (semantic search only)
sage --dense-only "user verification"

# Sparse-only (keyword search only)
sage --sparse-only "authenticate"

# Grep mode (exact + BM25, no dense, exhaustive)
sage -G "authenticate"

# Regex + semantic hybrid
sage -e "Promise\.allSettled" "parallel failure handling"
```

### Daemon Server

```bash
# Start daemon server (16ms warm search)
sage serve /path/to/project

# Search via daemon (auto-detected)
sage "authentication" /path/to/project

# Stop daemon
sage stop /path/to/project

# Auto-reindex on file changes
sage watch /path/to/project
```

### MCP Integration

```bash
# Run as MCP server (stdio transport)
sage mcp

# Install Claude Code integration
sage install-claude-code
```

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "sage": {
      "command": "sage",
      "args": ["mcp"]
    }
  }
}
```

## Architecture

### Search Pipeline

```
Query -> Query Classification -> Query Expansion (stemmed synonyms)
                                       |
                                       v
Query -> ColBERT Embedding -> PLAID Search ----+
                                               |-> Triple CC Fusion -> File Role Boost -> Results
Query -> Stemmed CodeTokenizer -> BM25 --------+
                                               |
Query -> Exact string match -------------------+
```

### Components

- **Indexer**: tree-sitter AST parsing, intelligent chunking with 6-layer NL annotations
- **Dense Embedder**: ColBERT LateOn-Code-edge via next-plaid (48d per-token, MaxSim)
- **Sparse Index**: Tantivy BM25 with Snowball stemmer + CodeTokenizer
- **Query Expander**: Stemmed synonym table bridging NL concepts to code tokens
- **Fusion**: Triple CC (Convex Combination) with query-adaptive weights
- **File Classifier**: 11-role path heuristics with semantic boost multipliers

### Project Structure

```
sage/
├── crates/
│   ├── sage-core/      # Core search engine (indexing, search, embeddings)
│   ├── sage-cli/       # Command-line interface
│   └── sage-mcp/       # MCP server
├── plugin/              # Claude Code plugin (skills, hooks)
│   ├── .claude-plugin/  #   Plugin manifest
│   ├── skills/          #   Agent skill definitions
│   └── hooks/           #   Session and tool hooks
├── benchmarks/          # Benchmark scripts and ground truth
└── docs/                # Architecture and design documents
```

## Supported Languages

sage indexes 27 file types, including 23 with tree-sitter AST-aware chunking:

- **AST-parsed (23):** Rust, Python, JavaScript, TypeScript, Go, Java, C, C++, Ruby, PHP, C#, Dart, Scala, Kotlin, Swift, Elixir, Lua, Haskell, OCaml, Zig, R, HTML, Svelte
- **Text-chunked (4):** Markdown, JSON, TOML, YAML

Plus fallback text chunking for any other file type.

## Benchmarks

On a 30-query benchmark (8 exact, 14 semantic, 8 architectural):

| Metric | grep | sage |
|--------|------|-------|
| **Overall F1** | 0.454 | **0.610** |
| **Speed (warm)** | 64ms | **17ms** |
| **Tokens (compact)** | 1,555 | **7** |
| **Exact F1** | 0.606 | **0.645** |
| **Semantic F1** | 0.463 | **0.568** |
| **Architectural F1** | 0.285 | **0.650** |

## Development

### Building from Source

```bash
git clone https://github.com/tk/sage.git
cd sage
cargo build --release
```

### Running Tests

```bash
cargo test --all
cargo clippy --all
```

### Environment Variables

```bash
SAGE_ORT_THREADS=4     # ONNX Runtime thread count (default: 4)
SAGE_COREML=1          # Enable CoreML acceleration on macOS
RUST_LOG=info            # Logging level (error, warn, info, debug, trace)
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache-2.0

## Credits

Built with:
- [next-plaid](https://github.com/lightonai/next-plaid) — ColBERT/PLAID late interaction search
- [Tantivy](https://github.com/quickwit-oss/tantivy) — Full-text search with BM25
- [fastembed-rs](https://github.com/Anush008/fastembed-rs) — Cross-encoder reranking
- [tree-sitter](https://tree-sitter.github.io/) — Code parsing
- [ONNX Runtime](https://onnxruntime.ai/) — Neural model inference
