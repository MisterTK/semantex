# semantex Configuration Reference

Complete guide to configuring semantex for optimal performance and results.

## Configuration Hierarchy

semantex uses a multi-layered configuration system with the following priority (highest to lowest):

1. **CLI flags**: Command-line arguments
2. **Environment variables**: `SEMANTEX_*` variables
3. **Project config**: `.semantexrc.yaml` in project root
4. **Global config**: `~/.config/semantex/config.yaml`
5. **Defaults**: Built-in defaults

Higher priority settings override lower priority ones.

## Configuration File Format

Configuration files use YAML format:

```yaml
# ~/.config/semantex/config.yaml
max_count: 15
content: false
rerank: false
```

## Complete Configuration Reference

### Search Settings

#### `max_count`

Maximum number of results to return.

- **Type**: Integer
- **Default**: `15`
- **Range**: 1-1000
- **CLI flag**: `--max-count`, `-m`
- **Environment**: `SEMANTEX_MAX_COUNT`

```yaml
max_count: 15
```

```bash
semantex --max-count 20 "authentication"
export SEMANTEX_MAX_COUNT=15
```

**Note**: Adaptive result sizing is enabled by default, so the actual number of results returned may be fewer than `max_count` when scores drop below the threshold for the query type.

#### `content`

Show content snippets in search results.

- **Type**: Boolean
- **Default**: `false`
- **CLI flag**: `--content`, `-c`
- **Environment**: `SEMANTEX_CONTENT`

```yaml
content: true
```

```bash
semantex --content "database pool"
export SEMANTEX_CONTENT=true
```

#### `context_lines`

Number of lines of context to show around matches.

- **Type**: Integer
- **Default**: `0`
- **Range**: 0-50
- **CLI flag**: `--context`, `-C`

```yaml
context_lines: 3
```

```bash
semantex --context 5 "error handling"
```

#### `rerank`

Enable cross-encoder reranking (JINA Reranker v1 Turbo via fastembed-rs).

- **Type**: Boolean
- **Default**: `false`
- **CLI flag**: `--rerank` (to enable)
- **Environment**: `SEMANTEX_RERANK`

```yaml
rerank: false
```

```bash
semantex --rerank "authentication"  # Enable reranking
export SEMANTEX_RERANK=true
```

**Trade-offs**:
- **Enabled**: ~1,200ms overhead, slightly different ranking
- **Disabled**: Fast queries (~17ms warm), hybrid fusion is already effective

**Note**: Benchmark testing showed the cross-encoder reranker (trained on web documents, not code) slightly hurts code search F1. Not recommended for general use.

#### `--refs`

Compact reference output for agent consumption.

- **Type**: Boolean flag
- **Default**: `false`
- **Conflicts with**: `--json`, `--grep`, `--content`, `--verbose`

Output format: `file:start-end name [kind] "summary"` grouped by file. Top result includes auto-peek (5-line preview). Footer shows confidence hint.

```bash
semantex --refs "authentication"
semantex --refs "auth flow" "token validation"  # batch mode
```

#### `--peek`

Refs plus first 5 lines of code per result.

- **Type**: Boolean flag
- **Default**: `false`
- **Conflicts with**: `--json`, `--grep`, `--content`, `--verbose`, `--refs`

```bash
semantex --peek "error handling"
```

#### `--around <SYMBOL>`

Graph walk: show callers, callees, type references, and hierarchy for a named symbol.

- **Type**: String (symbol name)
- **Default**: None
- **Standalone mode**: skips normal search

```bash
semantex --around handle_search
```

#### `--deep`

Deep search: internal search->read->summarize pipeline returning a curated prose answer.

- **Type**: Boolean flag
- **Default**: `false`
- **Conflicts with**: `--json`, `--grep`, `--refs`, `--peek`, `--around`, `--content`

Returns `<answer>...</answer>` with `Sources:` listing. Extractive summarization, no LLM required. Typically <50ms.

```bash
semantex --deep "how does authentication work end-to-end"
semantex --deep -v "query"  # prints timing metrics to stderr
```

#### `-e/--pattern <REGEX>`

Hybrid regex + semantic search.

- **Type**: String (regex pattern)
- **Default**: None

```bash
semantex -e "async fn" "database pool"
```

### Indexing Settings

#### `max_file_size`

Maximum file size to index (bytes). Files larger than this are skipped.

- **Type**: Integer (bytes)
- **Default**: `1048576` (1MB)
- **CLI flag**: `--max-file-size`
- **Environment**: `SEMANTEX_MAX_FILE_SIZE`

```yaml
max_file_size: 2097152  # 2MB
```

```bash
semantex index --max-file-size 5242880 .  # 5MB
export SEMANTEX_MAX_FILE_SIZE=2097152
```

**Recommendations**:
- Small projects: 1-2MB
- Medium projects: 2-5MB
- Large projects: Consider excluding large generated files

#### `max_file_count`

Maximum number of files to index per project.

- **Type**: Integer
- **Default**: `50000`
- **CLI flag**: `--max-file-count`

```yaml
max_file_count: 100000
```

```bash
semantex index --max-file-count 10000 .
```

#### `chunk_size`

Target chunk size in tokens.

- **Type**: Integer
- **Default**: `512`
- **Range**: 64-2048

```yaml
chunk_size: 256  # Smaller chunks, more granular search
```

**Trade-offs**:
- **Smaller** (256): More granular, more chunks, slower indexing
- **Larger** (1024): Coarser, fewer chunks, faster indexing

**Recommendations**:
- Most projects: 512
- Fine-grained search: 256
- Large files: 1024

#### `chunk_overlap`

Number of overlapping tokens between chunks.

- **Type**: Integer
- **Default**: `128`
- **Range**: 0-512

```yaml
chunk_overlap: 64  # Less overlap
```

**Trade-offs**:
- **More overlap**: Better context preservation, more chunks
- **Less overlap**: Fewer chunks, faster indexing

**Recommendations**:
- Default: 128 (25% of chunk_size)
- No overlap: 0 (for fully independent chunks)

#### `use_bm25_stemmer`

Whether to apply the English Snowball stemmer when building the BM25
sparse index (v0.4 Item 18).

- **Type**: Boolean
- **Default**: `true` (legacy behavior preserved)

```yaml
use_bm25_stemmer: true
```

**What stemming does**: The Snowball stemmer reduces inflected English
words to a shared stem. Example mappings: `retry` -> `retri`,
`handles` -> `handl`, `running` -> `run`. With stemming on, a query for
`retries` matches documents containing `retry` (helpful for natural-
language queries, harmful for exact identifier matching).

**When to disable**:
- Code-identifier-heavy corpora where exact symbol matches matter more
  than English morphology.
- Workloads dominated by literal identifier or API-name queries.

**Reindex required**: This flag is respected at index-build time only.
The value used when the index was built MUST match the value used at
search time. Changing the flag without running `semantex index` will
silently produce a query/index token mismatch (queries are tokenized
with the new analyzer, but documents were tokenized with the old one),
which degrades recall without raising an error.

```yaml
# Project that prefers literal identifier matches:
use_bm25_stemmer: false
```

After changing this flag, run:

```bash
semantex index .
```

### Embedding Settings

semantex uses ColBERT LateOn-Code-edge with fixed 48-dimension per-token embeddings via the `next-plaid` PLAID index. The embedding dimensions are not configurable — the model produces 48d INT8 vectors.

#### `colbert_model`

ColBERT model selection.

- **Type**: Enum
- **Default**: `LateOnCodeEdge`
- **Currently available**: `LateOnCodeEdge` (~17MB INT8 ONNX)

#### `plaid_nbits`

PLAID quantization bits for the dense index.

- **Type**: Integer
- **Default**: `4`
- **Valid values**: `2`, `4`

```yaml
plaid_nbits: 4
```

**Trade-offs**:
- **4-bit** (default): Better accuracy, larger index
- **2-bit**: Smaller index, slightly lower accuracy

### Fusion Settings

#### `rrf_k`

Fusion constant used internally in score normalization.

- **Type**: Float
- **Default**: `30.0`
- **Range**: 1.0-1000.0

```yaml
rrf_k: 30.0
```

semantex uses **Triple CC (Convex Combination)** fusion with query-adaptive weights across three channels (dense, sparse, exact). Weights are tunable per query type via environment variables:

```bash
SEMANTEX_WEIGHTS_IDENTIFIER=0.15,0.35,0.50  # dense,sparse,exact
SEMANTEX_WEIGHTS_KEYWORD=0.30,0.50,0.20
SEMANTEX_WEIGHTS_SEMANTIC=0.65,0.25,0.10
SEMANTEX_WEIGHTS_MIXED=0.45,0.35,0.20
```

See [docs/RRF_FUSION.md](docs/RRF_FUSION.md) for detailed analysis of the fusion approach.

### Reranking Settings

#### `rerank_candidates`

Number of candidates to pass to cross-encoder reranking (when `--rerank` is enabled).

- **Type**: Integer
- **Default**: `100`
- **Range**: 10-200

```yaml
rerank_candidates: 100
```

**Trade-offs**:
- **More candidates**: Better recall, slower reranking
- **Fewer candidates**: Faster, might miss relevant results

### File-Type Filtering Settings

#### `include_types`

File extensions to include in search results.

- **Type**: Array of strings
- **Default**: `[]` (all file types)
- **CLI flag**: `-t/--type <TYPE>` (repeatable)
- **Environment**: `SEMANTEX_INCLUDE_TYPES` (comma-separated)

```yaml
include_types: ["rs", "py", "js", "ts"]
```

```bash
semantex -t rs -t py "error handling"
export SEMANTEX_INCLUDE_TYPES="rs,py,js,ts"
```

#### `exclude_types`

File extensions to exclude from search results.

- **Type**: Array of strings
- **Default**: `[]` (no exclusions)
- **CLI flag**: `--exclude-type <TYPE>` (repeatable)
- **Environment**: `SEMANTEX_EXCLUDE_TYPES` (comma-separated)

```yaml
exclude_types: ["test", "md", "json"]
```

```bash
semantex --exclude-type test --exclude-type md "authentication"
export SEMANTEX_EXCLUDE_TYPES="test,md,json"
```

#### `code_only`

Exclude non-code files (documentation, config files, etc.).

- **Type**: Boolean
- **Default**: `false`
- **CLI flag**: `--code-only`

```yaml
code_only: true
```

```bash
semantex --code-only "database connection"
```

**Impact**: File-type filtering improves precision from 0.304 → 0.550 (+81%) in real-world benchmarks.

### Daemon Settings (WI1)

#### `daemon_idle_timeout`

Auto-stop daemon after N seconds of inactivity.

- **Type**: Integer (seconds)
- **Default**: `1800` (30 minutes)
- **CLI flag**: `--idle-timeout <SECONDS>`

```yaml
daemon_idle_timeout: 3600  # 1 hour
```

```bash
semantex serve --idle-timeout 3600 /path/to/project
```

**Performance**: Daemon reduces search latency to ~17ms warm.

### Model Settings

#### `model_dir`

Custom directory for ONNX models (optional).

- **Type**: Path
- **Default**: `~/.semantex/models/`
- **Environment**: `SEMANTEX_MODEL_DIR`

```yaml
model_dir: /custom/path/to/models
```

```bash
export SEMANTEX_MODEL_DIR=/mnt/shared/models
```

**Note**: The ColBERT model (~17MB) is cached at `~/.semantex/models/` by default.

## Configuration Examples

### High-Performance Setup (Speed Priority)

```yaml
# Fast search, minimal overhead
rerank: false
plaid_nbits: 2
chunk_size: 512
max_count: 10
```

### Large Codebase Setup (>50K files)

```yaml
# Optimized for scale
max_file_count: 200000
max_file_size: 2097152  # 2MB
chunk_size: 512
```

### Memory-Constrained Setup

```yaml
# Minimal memory footprint
plaid_nbits: 2
max_file_size: 524288  # 512KB
max_file_count: 20000
```

## Project-Specific Configuration

Create `.semantexrc.yaml` in your project root to override global settings:

```yaml
# .semantexrc.yaml — Project-specific overrides

# Example: Large monorepo
max_file_count: 500000
max_file_size: 2097152  # 2MB
plaid_nbits: 2          # Smaller index for large repos
```

This configuration applies only to searches within this project.

## Environment Variables

All configuration options can be set via environment variables:

```bash
# Search settings
export SEMANTEX_MAX_COUNT=5
export SEMANTEX_CONTENT=true
export SEMANTEX_RERANK=true

# File-type filtering
export SEMANTEX_INCLUDE_TYPES="rs,py,js,ts"
export SEMANTEX_EXCLUDE_TYPES="test,md,json"

# Indexing settings
export SEMANTEX_MAX_FILE_SIZE=2097152
export SEMANTEX_MODEL_DIR=/custom/models

# PLAID engine tuning
export SEMANTEX_PLAID_PROBE=32             # IVF centroid clusters probed per query
export SEMANTEX_PLAID_CENTROID_THRESHOLD=0.4  # Centroid score prune threshold

# Logging
export RUST_LOG=semantex=debug
```

### PLAID engine tuning

#### `SEMANTEX_PLAID_PROBE`

Number of IVF centroid clusters probed during PLAID approximate dense search.
Higher values increase recall at the cost of additional centroid comparisons
per query token.

- **Type**: Integer
- **Default**: `32`
- **Recommended range**: `16`–`64`
- **Environment**: `SEMANTEX_PLAID_PROBE`

```bash
export SEMANTEX_PLAID_PROBE=32   # Default
export SEMANTEX_PLAID_PROBE=8    # Restore pre-v0.4 next-plaid default (low recall)
```

ColBERT literature recommends 32–64 probes. semantex v0.3 used the upstream
`SearchParameters::default()` of 8, which under-probed for short queries.
v0.4 raises the default to 32; the env var is provided so benchmarks and
power users can override without recompiling.

#### `SEMANTEX_PLAID_CENTROID_THRESHOLD`

Centroid score threshold used by PLAID to prune candidate cells. Cells
with maximum query-token similarity below the threshold are skipped.

- **Type**: Float
- **Default**: `0.4`
- **Environment**: `SEMANTEX_PLAID_CENTROID_THRESHOLD`

```bash
export SEMANTEX_PLAID_CENTROID_THRESHOLD=0.4   # Default
export SEMANTEX_PLAID_CENTROID_THRESHOLD=0.0   # Disable pruning (worse precision, fuller recall)
```

Lower values consider more centroids (slower, broader recall); higher values
prune more aggressively (faster, may miss matches). The default is conservative.

## Logging Configuration

Control semantex logging via `RUST_LOG`:

```bash
# Levels: error, warn, info, debug, trace

# Production: Info-level logging
export RUST_LOG=info

# Development: Debug-level logging
export RUST_LOG=debug

# Performance debugging: Trace ONNX operations
export RUST_LOG=semantex::embedding=trace

# Module-specific logging
export RUST_LOG=semantex::search=debug,semantex::index=info
```

See [MONITORING.md](MONITORING.md) for detailed observability configuration.

## Performance Tuning Guide

### Tuning for Indexing Speed

```yaml
chunk_size: 1024           # Larger chunks, fewer total
chunk_overlap: 0           # No overlap
plaid_nbits: 2             # Smaller index
```

### Tuning for Search Speed

```yaml
rerank: false              # Skip reranking (default)
max_count: 10              # Fewer results
```

### Tuning for Memory Efficiency

```yaml
plaid_nbits: 2             # Smaller dense index
max_file_count: 20000      # Limit files
max_file_size: 524288      # 512KB max file size
```

## Troubleshooting

### Search is too slow

1. Ensure reranking is disabled: `rerank: false` (default)
2. Use daemon mode (`semantex serve`) for warm cache (~17ms)
3. Reduce `max_count` for fewer results

### Search quality is poor

1. Try different search modes: `--dense-only`, `--sparse-only`, or default hybrid
2. Use `--deep` for complex questions (returns complete answers)
3. Increase `max_count` for more candidates

### Indexing is too slow

1. Reduce `chunk_overlap`: Try 64 or 0
2. Use `plaid_nbits: 2` for faster index building
3. Reduce `max_file_count` to limit scope

### Out of memory during indexing

1. Set `max_file_count` limit
2. Reduce `max_file_size`
3. Use `plaid_nbits: 2` for smaller index

### Models not downloading

1. Check internet connection
2. Verify `~/.semantex/models/` is writable
3. Set custom model directory: `SEMANTEX_MODEL_DIR=/path`

## References

- [Tantivy BM25](https://github.com/quickwit-oss/tantivy)
- [next-plaid (ColBERT/PLAID)](https://github.com/lightonai/next-plaid)
- [fastembed-rs](https://docs.rs/fastembed) (reranking)
