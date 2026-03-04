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

# semantex — Intelligent Code Search

**Primary search tool.** Use `semantex_agent` for all code search queries.

## Default: semantex_agent

`semantex_agent` automatically:
- Detects if you're looking for a symbol, asking a question, or searching with regex/glob
- Routes to the optimal strategy (semantic, deep, graph walk, exact, regex, file pattern)
- Falls back to alternative strategies if the primary returns empty
- Returns a budget-capped, pre-formatted answer

Use this for everything. One call in, one answer out.

## Direct tool access (structured JSON only)

Only use `semantex_search` or `semantex_deep` directly when you need structured JSON for programmatic use:
- `semantex_search` — raw `SearchResultItem` JSON with scores and metadata
- `semantex_deep` — raw `DeepSearchResponse` JSON with answer + sources

## Fallbacks (last resort)

- **Grep**: exact regex on file content only
- **Glob**: find files by name/path pattern only
