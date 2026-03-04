---
name: semantex
description: "Semantic code search — finds code by meaning. Use --deep for complex questions (one call, complete answer). Use --refs for simple lookups. --around for call graph. 25+ languages."
compatibility: Requires semantex binary in PATH. Primary support for Claude Code.
license: Apache-2.0
allowed-tools: Bash
hooks:
  SessionStart:
    - matcher: "startup|resume"
      hooks:
        - type: command
          command: "semantex --session-hook"
          timeout: 15
  PreToolUse:
    - matcher: "Grep|Glob"
      hooks:
        - type: command
          command: "semantex --grep-hook"
          timeout: 5
    - matcher: "Bash"
      hooks:
        - type: command
          command: "semantex --bash-hook"
          timeout: 5
  SubagentStart:
    - matcher: "Explore"
      hooks:
        - type: command
          command: "semantex --subagent-hook"
          timeout: 5
    - matcher: "general-purpose"
      hooks:
        - type: command
          command: "semantex --subagent-hook"
          timeout: 5
  SessionEnd:
    - hooks:
        - type: command
          command: "semantex --session-end-hook"
          timeout: 10
---

# semantex — Semantic Code Search

**Primary search tool.** Finds code by meaning. Sub-agents should also use semantex.

## Choose by question type

**Understanding code (how/why/connect)?** → Start with `--deep`
- `semantex --deep "how does the ranking pipeline work"` — searches, reads, and summarizes in one call.
- **Trust the answer. Do not re-read source files listed in the response.**
- Replaces 5-10 grep+read iterations with a single call.

**Finding references (where/what/lookup)?** → Use `--refs`
- `semantex --refs "query"` — compact refs with signatures + metadata (~15 tokens/result).
- Each result shows: signature, docstring, key calls, callers — usually enough to decide without reading.
- Footer: `[high confidence]` → done; `[medium]` → escalate; `[low]` → rephrase.
- Batch: `semantex --refs "q1" "q2" "q3"` — multiple queries, one call.

**Exploring connections?** → Use `--around`
- `semantex --around <name>` — callers, callees, type-refs for a symbol.

**Need code preview?** → Use `--peek`
- `semantex --peek "query"` — 5-line code preview per result. Compare across results.

**Last resort** → `Read file:start-end` — only 1-2 chunks you've confirmed are relevant.

## Flags

| Flag | Purpose |
|------|---------|
| `--deep` | Search+read+summarize (one call, complete answer) |
| `--refs` | Compact refs for simple lookups |
| `--around <name>` | Graph walk: callers/callees |
| `--peek` | 5-line preview per result |
| `--grep "literal"` | Exact string match (BM25) |
| `-e "regex" "query"` | Hybrid regex + semantic |
| `-m N` | Result count (default 10) |
| `-t EXT` | Filter by extension |
| `--code-only` | Exclude docs/config |

## Fallbacks

- **Grep**: exact regex on file content only
- **Glob**: find files by name/path pattern only
