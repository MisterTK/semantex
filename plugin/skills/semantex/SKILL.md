---
name: semantex
description: "Semantic code search — finds code by meaning. Use instead of Grep/Glob for conceptual queries. semantex --refs \"query\" for compact refs, --peek for code previews, --around <name> for call graph. 25+ languages."
compatibility: Requires semantex binary in PATH. Primary support for Claude Code.
license: Apache-2.0
allowed-tools: Bash
---

# semantex — Semantic Code Search

**Primary search tool.** Finds code by meaning. Sub-agents should also use semantex.

## Workflow (follow this order — most tasks complete at step 1)

1. `semantex --refs "query"` — compact refs with signatures + metadata (~15 tokens/result). **Always start here.**
   - Each result shows: signature, docstring, key calls, callers — usually enough to decide without reading.
   - Footer: `[high confidence]` → done; `[medium]` → escalate; `[low]` → rephrase.
   - Batch: `semantex --refs "q1" "q2" "q3"` — multiple queries, one call.
2. `semantex --around <name>` — callers, callees, type-refs for a symbol. Use EARLY.
3. `semantex --deep "question"` — search+read+summarize. For multi-hop questions (3+ Reads).
4. `semantex --peek "query"` — 5-line code preview per result. Compare across results.
5. `Read file:start-end` — **last resort**, only 1-2 chunks you've confirmed are relevant.

## Flags

| Flag | Purpose |
|------|---------|
| `--refs` | Compact refs (default start) |
| `--around <name>` | Graph walk: callers/callees |
| `--deep` | Auto search+read+summarize |
| `--peek` | 5-line preview per result |
| `--grep "literal"` | Exact string match (BM25) |
| `-e "regex" "query"` | Hybrid regex + semantic |
| `-m N` | Result count (default 10) |
| `-t EXT` | Filter by extension |
| `--code-only` | Exclude docs/config |

## Fallbacks

- **Grep**: exact regex on file content only
- **Glob**: find files by name/path pattern only
