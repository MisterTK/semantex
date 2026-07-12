---
name: semantex-docs
description: "Generate and maintain repo documentation from semantex's deterministic docs scaffolds. Use when asked to document a codebase, write architecture docs, or keep docs in sync with code. Calls semantex_docs_context (no LLM inside semantex) — you write the prose."
compatibility: Requires the semantex MCP server (semantex_docs_context tool). Primary support for Claude Code.
license: Apache-2.0
allowed-tools: Read, Write, Edit, Glob, Grep
---

# semantex-docs — maintained documentation from deterministic scaffolds

`semantex_docs_context` is **not an LLM call**. It never writes prose. It returns a
structurally-complete JSON scaffold — symbol inventory, call-graph edges, import edges,
existing doc-comment text, and file:line provenance for every claim — built deterministically
from the semantex index. **You** (the calling agent) turn that scaffold into maintained
markdown. This split is intentional: semantex ships zero LLM wiring, so documentation quality
is bounded by your judgment, not a hidden model call you can't inspect or control.

Docs land in the **user's repo** under `semantex_docs/` — a plain directory of markdown files,
not a semantex-internal artifact. Whether the user commits that directory or adds it to
`.gitignore` is their call, not yours; don't decide it for them unless asked.

## The workflow

### 1. Overview pass (once per session, or when asked for "the docs")

Call:

```json
{"scope": "overview"}
```

This returns:
- `god_nodes` — highest-PageRank-centrality symbols (the load-bearing code)
- `communities` — connected components over the call graph, with entry points
- `boundaries` — cross-directory import coupling (which top-level dirs depend on which)
- `module_inventory` — every indexed file, with symbol count / language / inferred role,
  sorted by symbol count descending
- `language_stats` — file/chunk counts per language

Write or refresh `semantex_docs/README.md` from this. Good structure:
- A one-paragraph "what this repo is" (infer from god_nodes + top modules + any existing
  README — don't invent a purpose that isn't evidenced by the code)
- An architecture section built from `communities` (one subsection per community: what it's
  for, its entry points with `file:line` links) and `boundaries` (a short dependency-direction
  note — "X depends on Y, not the reverse")
- A "where to look" table from `module_inventory`'s top entries (highest symbol_count first —
  that's usually where the interesting logic lives)
- A language breakdown if the repo is polyglot (`language_stats`)

### 2. Per-module pass

For each module worth a dedicated doc (start with the top of `module_inventory`, or whatever
the user pointed you at), call:

```json
{"scope": {"module": "path/as/indexed.rs"}}
```

This returns:
- `symbols` — every function/method/class/etc. with signature, params, return type, existing
  `docstring` + `doc_tags` (already-written doc comments — **preserve and refresh these,
  don't discard them**), semantic role, and `provenance` (file:line)
- `imports` / `imported_by` — the file-level dependency edges in both directions, **project-internal
  only**: external packages, stdlib, and third-party dependencies are never resolved to a path and
  never appear here. A short or empty list means "no *other project file* depends on this one this
  way" — not "this file has no dependencies." Don't write "this file has no imports" from an empty
  list; say what the scaffold actually shows (no internal import edges found) or omit the claim.
- `calls_out` / `calls_in` — call-graph edges in both directions, resolved to file:line where
  the graph resolution succeeded (unresolved edges are still listed by name)

Write `semantex_docs/<module-path-with-slashes-flattened-to-double-underscore>.md`
(e.g. `src/index/storage.rs` → `semantex_docs/src__index__storage.md`) — flattening avoids
directory-creation churn and keeps every doc discoverable with one `Glob`. Good structure:
- What this module is for (inferred from symbol names, existing docstrings, semantic roles —
  cite the module's own doc comment if `symbols` includes a module-level entry)
- Public surface: one subsection per public symbol worth documenting, each ending with its
  `file:line` citation
- Dependencies: `imports`/`imported_by` as a short "depends on / depended on by" list
- Call graph highlights: only the `calls_out`/`calls_in` edges that aren't obvious from the
  symbol list alone (e.g. a resolved call into a different module — that's the interesting
  cross-cutting edge; a call within the same file usually isn't worth a doc line)

### 3. Citing provenance

Every scaffold item carries `file:line` (a `provenance` field, or nested `location`/`provenance`
on graph edges). **Cite it** — end factual sentences with `(file.rs:12-40)` — and **verify it**
before you write: a `Read` of the cited range should back up the sentence. If the scaffold's
`docstring` for a symbol already says what you were about to write, don't duplicate — quote or
lightly edit it instead of re-deriving prose from scratch.

Never state something the scaffold + a source read doesn't support. If you want to say *why*
something is designed a certain way and neither the scaffold nor the code shows it, say "the
reason isn't evident from the code" rather than guessing.

### 4. Refresh passes (docs already exist)

When `semantex_docs/<file>.md` already exists, don't regenerate it wholesale — diff scaffold
vs. doc and update only what drifted:

1. Read the existing doc.
2. Call `semantex_docs_context` for the same scope again (index is live — this reflects the
   current tree, not whatever the doc was written against).
3. For a module doc: compare each documented symbol's cited `file:line` range against the
   fresh scaffold's `provenance` for that symbol name.
   - Same name, same (or near-same) line range → likely still accurate; leave the prose,
     but skim it against the current `docstring`/`signature` for a quick sanity check.
   - Same name, moved/resized range → the symbol shifted; re-verify the prose against the
     new range and update the citation.
   - Name gone from `symbols` → the symbol was removed or renamed; remove or update that
     section (check `calls_out`/`imports` of related modules for a rename hint before
     deleting outright).
   - New name in `symbols` with no matching doc section → add one.
4. For the overview doc: recheck `community` membership and `god_nodes` — if the top
   god_nodes or a community's entry points changed meaningfully, that's worth a rewrite of
   the affected subsection; stable sections don't need touching.

This keeps the diff small and the docs trustworthy — a doc that's 90% untouched and 10%
freshly-verified is more useful than one fully regenerated (and therefore fully unreviewed)
every run.

### Budget

`semantex_docs_context` accepts an optional `budget` (approximate tokens, default 6000). The
tool trims deterministically when a scaffold is large — highest-signal items (most-referenced
symbols, most-connected modules) survive the trim, so you don't need to pre-filter yourself.
Raise it for a single very large module; leave it default otherwise.

### What NOT to do

- Don't call an LLM API yourself to "generate the docs" in one shot from the scaffold's raw
  JSON without reading the cited source — that reintroduces exactly the hallucination risk
  this split was designed to avoid.
- Don't skip the overview pass and jump straight to module docs — the architecture section
  is what makes individual module docs navigable.
- Don't write `semantex_docs/` files as a side effect of an unrelated task unless asked;
  this skill is for explicit documentation requests.
