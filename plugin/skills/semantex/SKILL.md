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

**One call replaces 5-10 Grep + Read iterations.** Use `semantex_agent` as your primary search
tool for all code questions. It auto-routes to the right strategy and returns a complete,
pre-formatted answer.

## Why semantex_agent beats Grep + Read

Every tool call re-sends the full accumulated context (O(N²) cost). Halving tool calls cuts
context burden by ~75%, not ~50%. `semantex_agent` with `depth="deep"` collapses a full
grep-then-read investigation into a single call that returns full function bodies, callers,
and callees — nothing left to look up.

**After `depth="deep"` or `semantex_deep`: do NOT call Read on the listed files. The full
code is already in the response. The `[COMPLETE]` marker confirms this.**

## Choosing depth

| depth | When to use | Latency |
|-------|-------------|---------|
| `"quick"` | Symbol lookup, exact name, find definition | ~50 ms |
| `"search"` | Broad search, list all X, find usages | ~100 ms |
| `"deep"` | "How does X work?", "Where to add Y?", architecture questions | ~200 ms |
| *(omit)* | Auto-detect from query text | varies |

**Use `depth="deep"` for:**
- "How does X work?" — returns full implementation with call flow
- "Where should I add X?" — returns injection points and existing patterns
- Architecture questions — returns the relevant subsystem with context
- Any question where you would otherwise call Read after searching

**Use `depth="search"` for:**
- "List all handlers for X"
- "Show me every place Y is called"
- Broad enumeration queries

**Use `depth="quick"` for:**
- "Where is function foo defined?"
- Exact symbol lookups by name

## Multi-part questions: use `queries` (one call, not two)

```json
{
  "queries": ["authentication flow", "session token validation"],
  "depth": "search"
}
```

Use `queries` (array) instead of calling `semantex_agent` twice. Results are merged.
Use for 2-3 related concepts that you would otherwise search separately.

## Focus: tell semantex what you need

```json
{
  "query": "HTTP request handler",
  "depth": "deep",
  "focus": "signatures"
}
```

| focus | Effect |
|-------|--------|
| `"implementation"` | Full code bodies (default for deep) |
| `"callers"` | Who calls these functions (call graph edges) |
| `"signatures"` | Function signatures only — no bodies |
| `"patterns"` | Usage examples |

## Quick reference

```json
// Understand how something works
{"query": "how does auth work", "depth": "deep"}

// Find where to add a feature
{"query": "where is middleware registered", "depth": "deep", "focus": "patterns"}

// List all implementations of an interface
{"query": "implements Handler interface", "depth": "search"}

// Exact symbol lookup
{"query": "NewSessionManager", "depth": "quick"}

// Two related questions in one call
{"queries": ["rate limiting logic", "retry backoff"], "depth": "search"}

// Architecture question
{"query": "how does the indexing pipeline work end to end", "depth": "deep"}
```

## Direct tools (structured JSON only)

Use `semantex_search` or `semantex_deep` only when you need raw JSON for programmatic
processing. For all human-readable results, use `semantex_agent`.

## Fallbacks (last resort)

- **Grep**: exact regex on file content only — no semantic understanding
- **Glob**: find files by name/path pattern only
