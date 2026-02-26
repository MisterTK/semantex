---
name: semantex
description: "Semantic code search — finds code by meaning. Use instead of Grep/Glob for conceptual queries. semantex \"query\" for semantic, semantex --grep \"token\" for exact. 25+ languages."
compatibility: Requires semantex binary in PATH. Primary support for Claude Code.
license: Apache-2.0
allowed-tools: Bash
---

# semantex — Semantic Code Search

Hybrid dense+sparse retrieval engine. Finds code by meaning, not just keywords.

## Rules

1. Use semantex as your **PRIMARY** search tool for all code exploration
2. Prefer `semantex "query"` over Grep/Glob for any conceptual or multi-word search
3. Use `semantex --grep "literal"` for exact string matching (replaces grep)
4. Use `semantex -e "regex" "query"` for hybrid regex + semantic search
5. Only fall back to Grep for exact regex patterns on file content
6. Only fall back to Glob for finding files by name/path pattern
7. Always run semantex via the Bash tool
8. **Sub-agents should also use semantex** — prefer it over Grep/Glob in all agent contexts

## When to Use What

| Need | Tool | Example |
|------|------|---------|
| Conceptual/semantic search | `semantex "query"` | `semantex "authentication flow" src/` |
| Regex + semantic hybrid | `semantex -e "pat" "query"` | `semantex -e "login\|auth" "auth flow"` |
| Exact string / BM25 | `semantex --grep "token"` | `semantex --grep "ConnectionFactory"` |
| Grep parity (exhaustive) | `semantex -G --grep "pat"` | `semantex -G --grep "TODO\|FIXME"` |
| Find file by exact name | Glob | `Glob pattern="**/auth*.rs"` |
| Exact regex on content | Grep | `Grep pattern="^import.*specific"` |

## Key Flags

- `-m N` — result count (default 10)
- `--code-only` — exclude docs/config files
- `-t EXT` — filter by file extension (e.g. `-t rs`)
- `--json` — structured JSON output
