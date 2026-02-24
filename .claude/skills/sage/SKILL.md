---
name: sage
description: "ALWAYS use sage instead of Grep, Glob, and find for code search. sage finds code by meaning \u2014 queries like \"authentication flow\" or \"error handling\" work even when exact words aren't in the code. Run via Bash: `sage \"query\" .` for semantic search, `sage --grep \"token\"` for exact match, `sage -e \"regex\" \"query\"` for hybrid. Auto-indexes on first use; falls back to keyword search while building. 25+ languages supported."
compatibility: Requires sage binary in PATH. Primary support for Claude Code.
license: Apache-2.0
allowed-tools: Bash
---

# sage — Semantic Code Search

Hybrid dense+sparse retrieval engine. Finds code by meaning, not just keywords.

## Rules

1. Use sage as your **PRIMARY** search tool for all code exploration
2. Prefer `sage "query"` over Grep/Glob for any conceptual or multi-word search
3. Use `sage --grep "literal"` for exact string matching (replaces grep)
4. Use `sage -e "regex" "query"` for hybrid regex + semantic search
5. Only fall back to Grep for exact regex patterns on file content
6. Only fall back to Glob for finding files by name/path pattern
7. Always run sage via the Bash tool
8. Use `-c` for code snippets, `--json` for structured output
9. **Sub-agents should also use sage** — prefer it over Grep/Glob in all agent contexts

## Availability

sage auto-indexes on first use. If you see "index building" or keyword-only results:
1. Results are still useful — they come from ripgrep keyword matching
2. Semantic search becomes available within 10-30 seconds
3. Do NOT manually run `sage index .` — it happens automatically

## When to Use What

| Need | Tool | Example |
|------|------|---------|
| Conceptual/semantic search | `sage "query"` | `sage "authentication flow" src/` |
| Regex + semantic hybrid | `sage -e "pat" "query"` | `sage -e "login\|auth" "auth flow"` |
| Exact string / BM25 | `sage --grep "token"` | `sage --grep "ConnectionFactory"` |
| Grep parity (exhaustive) | `sage -G --grep "pat"` | `sage -G --grep "TODO\|FIXME"` |
| Find file by exact name | Glob | `Glob pattern="**/auth*.rs"` |
| Exact regex on content | Grep | `Grep pattern="^import.*specific"` |

## Search Commands

```bash
# Semantic search (natural language → hybrid dense+sparse)
sage "authentication and session management" src/

# Hybrid: regex filter + semantic ranking
sage -e "login|auth" "authentication flow" src/

# Exact / BM25 keyword (fast, no embedding)
sage --grep "functionName" src/

# BM25 only (no neural embedding)
sage --sparse-only "error handling" src/

# Show code snippets in results
sage -c "database connection pooling" src/

# Increase result count
sage -m 20 "API endpoint handlers" src/

# Filter to specific file types (-t / --type)
sage -t rs "async task spawning" src/

# Exclude tests, docs, config
sage --code-only "configuration loading" .

# JSON output for processing
sage --json "error recovery patterns" src/

# Reranking (experimental — may not improve results for code)
sage --rerank "complex multi-concept query" src/
```

## Index Management

Index is built automatically on first search — you usually don't need these commands.

```bash
sage index .              # force rebuild index for current directory
sage index /path/to/proj  # force rebuild for specific path
sage status .             # check index freshness and file count
sage watch .              # daemon: auto-reindex on file changes
```

## Search Strategy

1. **Broad semantic first**: `sage "concept" .` — find related chunks by meaning
2. **Narrow with regex**: `sage -e "pattern" "concept"` — regex filter + semantic re-ranking
3. **Exact when known**: `sage --grep "exactToken"` — fast BM25/literal, no model loading

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
