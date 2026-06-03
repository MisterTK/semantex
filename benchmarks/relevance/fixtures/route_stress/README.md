# route-stress evaluation corpus

A **multi-gold, mechanism-tagged** retrieval corpus for measuring a code-search
engine's **query router** — which mechanism it picks per query, and whether each
mechanism earns its keep.

## Why this exists

The existing relevance corpora (`coir`, `csn`) are all NL→code **semantic**,
**single-gold** queries. They cannot exercise the router's non-semantic
mechanisms — glob, regex, exact-symbol, structural call-graph — at all. This
corpus fills that gap. Each record is tagged with the **mechanism it is meant to
stress** and carries a **multi-gold** answer set, so a later harness can ask:

1. Does the router classify the query into the right mechanism?
2. Does each mechanism out-retrieve the others on the queries that favour it?

## The 6 mechanisms

| mechanism    | example query                          | gold derivation                              | source   |
|--------------|----------------------------------------|----------------------------------------------|----------|
| `glob`       | "all Go test files (`*_test.go`)"      | filesystem walk (`git ls-files`) by path glob| derived  |
| `regex`      | "lines containing TODO/FIXME"          | ripgrep over source; files with ≥1 match     | derived  |
| `lexical`    | "the `RouterGroup` struct"             | ripgrep the language def-pattern; def file(s)| derived  |
| `structural` | "what calls `handleHTTPRequest`"       | ripgrep `sym(` call-sites minus def lines    | derived  |
| `semantic`   | "how does the middleware chain work"   | hand-judged relevant files                   | curated  |
| `usage`      | "how do I register a route handler"    | hand-judged idiomatic-usage / public-API file| curated  |

## Gold convention (READ THIS)

**Gold ids are repo-relative file paths; matching is FILE-granularity**
(`match_granularity: "file"` on every record). Justification, with the harness
code that pins it:

- semantex stores and returns each chunk's `file_path` **relative to the project
  root** — `crates/semantex-core/src/index/builder.rs:251`
  (`let rel_path = file_path.strip_prefix(&project_path)…`), surfaced verbatim as
  the `"file"` field of `semantex search --json`
  (`crates/semantex-cli/src/commands/search.rs:743`).
- The harness file-mode matcher compares a result's `file` field **verbatim**
  against the gold set —
  `benchmarks/relevance/src/relevance_harness/runner.py` `_relevance_vector`,
  `match_mode="file"` (`[1 if f in gold else 0 for f in rr.ranked_files]`).
  `RankedResult.rank_of_file` (`types.py`) does the same.
- This matches the existing file-level corpora: SWE-loc gold is bare
  repo-relative paths (`datasets/swe_loc.py`, `changed_files_from_patch`), and
  the `tiny_eval_corpus.json` fixture uses `gold_files` + `match_mode: "file"`.

### Why NOT chunk-level (`file:start-end`) doc_ids for the derived mechanisms

The other match mode is `doc_id`, where gold = `{relpath}:{start_line}-{end_line}`
and must equal a chunk semantex actually emits (`semantex_client.py` `_doc_id`,
`csn.py`). We deliberately **do not** use it here because a chunk's span is
**AST-chunker-determined and not reproducible from source alone**:

  | symbol (gin)        | def line | actual chunk span |
  |---------------------|----------|-------------------|
  | `func New`          | 202      | `202-233`         |
  | `type Context struct` | 61     | `61-97`           |
  | `handleHTTPRequest` | 690      | `690-760`         |

A grep-derived `file:line-line` id would silently fail to match the real chunk
span, and the span changes whenever the chunker / index schema changes. **File
granularity is reproducible, chunker-version-stable, and unambiguous** — the
right altitude for a router-routing eval (we are testing *which mechanism finds
the right file*, not sub-file ranking). If a future task needs chunk-level gold,
derive the spans from a freshly built index at generation time and pin the
schema version — do not hand-write line ranges.

## Favor-unambiguous-gold principle

Mechanically-derived targets are chosen to have a **single or small,
clearly-enumerable** gold set. Avoid queries whose "right answer" is debatable —
those belong only in the small hand-curated `semantic` set.

Concrete pitfalls handled in the gin pilot:

- **Symbol-name collisions (lexical).** "the `Engine` struct" was *rejected*: a
  method `func (v *defaultValidator) Engine()` in `binding/default_validator.go`
  also matches the def-pattern, polluting the gold. Replaced with
  `ResponseWriter` (one clean `type … interface`, no method collision). When
  picking a lexical symbol, prefer a distinctive name with exactly one
  definition.
- **Generic call patterns (structural).** "callers of `New`" was *rejected*:
  `New(` also matches `template.New(`, `errors.New(`, etc. Picked
  distinctively-named methods (`handleHTTPRequest`, `combineHandlers`,
  `calculateAbsolutePath`, `allocateContext`, `redirectTrailingSlash`) whose
  `sym(` pattern is unambiguous. The def line is always excluded from call-sites.
- **This matters even more for the later monorepo slice (Platform):** with many
  packages, a bare symbol can be defined/called in several unrelated modules.
  Qualify targets (package-scoped names, path-anchored globs) so the gold stays
  a small, defensible set; push anything genuinely fuzzy into the curated
  semantic set with an explicit `note` justifying each gold file.

## Structural cross-check (graph.json)

When a repo has `graphify-out/graph.json` (gin does), the generator
independently re-derives each structural target's call-site files from the
graph's `calls` edges (`source_file` of edges whose `target` node matches the
symbol) and records whether the graph **agrees** with the ripgrep derivation.
All 5 gin structural records show `graph-crosscheck agrees=True`. The ripgrep
derivation remains authoritative; the graph is a sanity check only.

## Files

| file                     | role                                                                 |
|--------------------------|----------------------------------------------------------------------|
| `gen_route_stress.py`    | repo-agnostic generator: derives glob/regex/lexical/structural gold, merges curated records, validates, emits the corpus |
| `spec_gin.json`          | **derivation spec** for gin — names the targets (globs/regexes/symbols); gold is derived mechanically, so it's regenerated |
| `curated_gin.json`       | **hand-curated** semantic + usage records (committed data, NOT regenerated); each gold file carries a `note` justifying it |
| `gin_route_stress.json`  | the generated gin pilot corpus (30 records, 5 per mechanism)         |

## Record schema

```json
{
  "id": "gin-struct-1",
  "repo": "gin",
  "query": "what calls handleHTTPRequest",
  "intended_mechanism": "structural",
  "gold": ["gin.go"],
  "match_granularity": "file",
  "source": "derived",
  "note": "callers of handleHTTPRequest; graph-crosscheck agrees=True"
}
```

- `source: "derived"` — gold is reproduced by re-running the generator.
- `source: "curated"` — gold is committed data (semantic/usage); edit
  `curated_gin.json` by hand, never regenerate it.

## Regenerate

```bash
cd benchmarks/relevance/fixtures/route_stress
python3 gen_route_stress.py \
  --repo /path/to/gin --lang go --repo-name gin \
  --spec spec_gin.json --curated curated_gin.json \
  --out gin_route_stress.json
```

The derived gold is reproducible: a clean re-run is byte-identical (verified).
The generator exits non-zero if any record has empty gold (a sign the target
moved). Requires `ripgrep` (`rg`) on PATH and a `git` checkout of the target
repo (it uses `git ls-files`; falls back to a dot-dir-skipping walk otherwise).

### Adding a repo / language

Add a `LANG_PATTERNS[<lang>]` entry (globs + `def` + `is_def_line` regexes),
write a `spec_<repo>.json` naming the targets, hand-author
`curated_<repo>.json`, and run the generator. Nothing in the generator is
gin-specific beyond the per-language def/call patterns.

## Status

**PILOT — gin only.** flask (Python) and Platform (monorepo slice) come later,
pending review of this methodology.
