---
name: semantex
description: "Semantic code search \u2014 finds code by meaning. Use instead of Grep/Glob for conceptual queries. sage \"query\" for semantic, semantex --grep \"token\" for exact. 25+ languages."
compatibility: Requires sage binary in PATH. Primary support for Claude Code.
license: Apache-2.0
allowed-tools: Bash
---

# sage \u2014 Semantic Code Search

Hybrid dense+sparse retrieval engine. Finds code by meaning, not just keywords.

## Rules

1. Use sage as your **PRIMARY** search tool for all code exploration
2. Prefer `semantex "query"` over Grep/Glob for any conceptual or multi-word search
3. Use `semantex --grep "literal"` for exact string matching (replaces grep)
4. Use `semantex -e "regex" "query"` for hybrid regex + semantic search
5. Only fall back to Grep for exact regex patterns on file content
6. Only fall back to Glob for finding files by name/path pattern
7. Always run semantex via the Bash tool
8. Use `-c` for code snippets, `--json` for structured output
9. **Sub-agents should also use semantex** \u2014 prefer it over Grep/Glob in all agent contexts

## Availability

semantex auto-indexes on first use. If you see "index building" or keyword-only results:
1. Results are still useful \u2014 they come from ripgrep keyword matching
2. Semantic search becomes available within 10-30 seconds
3. Do NOT manually run `semantex index .` \u2014 it happens automatically

## When to Use What

| Need | Tool | Example |
|------|------|---------|
| Conceptual/semantic search | `semantex "query"` | `semantex "authentication flow" src/` |
| Regex + semantic hybrid | `semantex -e "pat" "query"` | `semantex -e "login\|auth" "auth flow"` |
| Exact string / BM25 | `semantex --grep "token"` | `semantex --grep "ConnectionFactory"` |
| Grep parity (exhaustive) | `semantex -G --grep "pat"` | `semantex -G --grep "TODO\|FIXME"` |
| Find file by exact name | Glob | `Glob pattern="**/auth*.rs"` |
| Exact regex on content | Grep | `Grep pattern="^import.*specific"` |

## Search Commands

```bash
# Semantic search (natural language \u2192 hybrid dense+sparse)
sage "authentication and session management" src/

# Hybrid: regex filter + semantic ranking
sage -e "login|auth" "authentication flow" src/

# Exact / BM25 keyword (fast, no embedding)
semantex --grep "functionName" src/

# BM25 only (no neural embedding)
semantex --sparse-only "error handling" src/

# Show code snippets in results
sage -c "database connection pooling" src/

# Increase result count
sage -m 20 "API endpoint handlers" src/

# Filter to specific file types (-t / --type)
sage -t rs "async task spawning" src/

# Exclude tests, docs, config
semantex --code-only "configuration loading" .

# JSON output for processing
semantex --json "error recovery patterns" src/

# Reranking (experimental \u2014 may not improve results for code)
semantex --rerank "complex multi-concept query" src/
```

## Index Management

Index is built automatically on first search \u2014 you usually don't need these commands.

```bash
semantex index .              # force rebuild index for current directory
semantex index /path/to/proj  # force rebuild for specific path
semantex status .             # check index freshness and file count
semantex watch .              # daemon: auto-reindex on file changes
```

## Search Strategy

1. **Broad semantic first**: `semantex "concept" .` \u2014 find related chunks by meaning
2. **Narrow with regex**: `semantex -e "pattern" "concept"` \u2014 regex filter + semantic re-ranking
3. **Exact when known**: `semantex --grep "exactToken"` \u2014 fast BM25/literal, no model loading

## Output Format

Default: `file:start_line-end_line | score | first line of chunk`

With `-c`: full matched code chunk included.

With `--json`:
```json
[{"file": "src/auth.rs", "start_line": 42, "end_line": 67, "score": 0.91, "content": "..."}]
```

## Performance

| Mode | Latency |
|------|---------|
| `--grep` / `--sparse-only` | Instant |
| Semantic (warm daemon) | <100ms |
| Semantic (cold start) | 30-90s (model loading) |
| With `--rerank` | +200ms |

The sage daemon starts automatically on the first query and stays resident. Subsequent queries in the same session are fast.
