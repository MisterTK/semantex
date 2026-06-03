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

### `glob_kind` discriminator (literal vs nl)

`glob`-mechanism records carry a `glob_kind` field to distinguish two subtypes:

| `glob_kind` | example query        | router implication                          |
|-------------|----------------------|---------------------------------------------|
| `"nl"`      | "all Go test files (`*_test.go`)" | NL description of a glob; semantic/NL router path |
| `"literal"` | `*_test.go`          | query IS a glob pattern; fires `file_pattern` route |

**Why this matters:** the `file_pattern` route fires **only** when the query is itself a literal glob pattern (contains `*` or `?` as a wildcard operator). A natural-language description like "all Go test files" will be routed through the semantic/NL path, not `file_pattern` — scoring 0 on a `file_pattern`-route evaluation is expected, not a bug. The NL variants (`glob_kind: "nl"`) legitimately test whether the router handles NL glob descriptions; the literal variants (`glob_kind: "literal"`) test whether `--route file_pattern` returns the right files. Both subtypes share the same `intended_mechanism: "glob"` and identical gold derivation logic. A harness can slice by `glob_kind` to evaluate these two code-paths independently.

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

## Global test-inclusion policy

Test files are **never excluded** from any derived gold (glob / regex / lexical
/ structural). A test file that matches a regex, defines a symbol, or *calls* a
symbol IS a real match/def/caller — completeness beats an arbitrary test/non-test
split, and a silent per-target exclusion would make gold inconsistent. There is
deliberately **no** `exclude_tests` toggle in the generator or specs. If a
production-only variant is ever needed, author it as a separate, clearly
labelled query — not a hidden flag. (This policy is why e.g. gin `combineHandlers`
callers include `routergroup_test.go` and flask `get_flashed_messages` callers
are `tests/test_basic.py`.)

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
- **This matters even more for the monorepo slice (platform):** with many
  packages and two languages, a bare symbol can be defined/called in several
  unrelated modules. Qualify targets (package-scoped names, path-anchored globs)
  so the gold stays a small, defensible set; push anything genuinely fuzzy into
  the curated semantic set with an explicit `note` justifying each gold file.

### Symbols rejected on platform (the monorepo pitfall, applied)

The platform slice is **deliberately small** (~3-4 per mechanism). Every
lexical/structural symbol was grepped **repo-wide across BOTH languages first**
and kept only if it had exactly one definition in one file/one language. Rejected:

- **`Config` / `Service` / `Logger` / `User` / generic suffix names** — too
  common across `apps/api`, `apps/fhir-sync`, `apps/ddt`, and the Flutter app;
  a bare query would have a debatable multi-module gold. Rejected for lexical.
- **`create(` / `validate(` (≈14 / ≈5 files) and other generic method names** —
  far too broad as a structural call pattern. Rejected for structural; picked
  distinctively-named services/classes (`DataAvailabilityCache`,
  `FhirPersonaGeneratorService`, `HealthService`) whose `new X(` /`X(` pattern is
  unambiguous and whose callers are a small, clean set.
- **Dart class constructors as structural targets** — *rejected for a generator
  reason*: the Dart `is_def_line` recognises type defs (`class/enum/mixin/…`)
  but NOT a constructor decl (`const ConnectionCard({` / `FamilyApiService(this._dio)`)
  or a method def (`StreamingMessage createStreamingMessage()`), so a Dart
  `Sym(` call-derivation would mis-count the signature line as a call site.
  Rather than build fragile Dart method/ctor-def detection, **all platform
  structural targets are TS** (TS instantiation is `new X(`, and the def lines
  `class X` / `constructor(` never match `X(`, so TS call-sites derive cleanly).
  Dart still contributes glob, regex, and **lexical** targets (type defs derive
  correctly) plus curated semantic/usage — so cross-language coverage holds.

## Structural cross-check (graph.json)

When a repo has `graphify-out/graph.json` (gin does; flask and platform do not),
the generator independently re-derives each structural target's call-site files
from the graph's `calls` edges (`source_file` of edges whose `target` node
matches the symbol) and records whether the graph **agrees** with the ripgrep
derivation. All 5 gin structural records show `graph-crosscheck agrees=True`.
The ripgrep derivation remains authoritative; the graph is a sanity check only,
and is simply absent (no `graph-crosscheck` in the note) for repos without a
graph.

## Files

| file                          | role                                                                 |
|-------------------------------|----------------------------------------------------------------------|
| `gen_route_stress.py`         | repo-agnostic generator: derives glob/regex/lexical/structural gold, merges curated records, validates, emits the corpus. Languages: `go`, `python`, `typescript`, `dart` (in `LANG_PATTERNS`) |
| `spec_<repo>.json`            | **derivation spec** — names the targets (globs/regexes/symbols); gold is derived mechanically, so it's regenerated. A monorepo spec can set a per-item `"lang"` / `"globs"` to mix languages in one file |
| `curated_<repo>.json`         | **hand-curated** semantic + usage records (committed data, NOT regenerated); each gold file carries a `note` justifying it |
| `<repo>_route_stress.json`    | the generated corpus for that repo                                   |

Repos in the corpus:

| repo       | lang        | records | glob (nl+literal)                              | other mechanisms             |
|------------|-------------|---------|------------------------------------------------|------------------------------|
| `gin`      | Go          | 34      | 9 (5 nl + 4 literal)                           | 5 × {regex, lexical, structural, semantic, usage} |
| `flask`    | Python      | 34      | 9 (5 nl + 4 literal)                           | 5 × each                     |
| `platform` | TS + Dart (monorepo slice) | 27 | 9 (4 nl + 5 literal)              | regex 4, lexical 5, structural 3, semantic 3, usage 3 |

Per-mechanism totals across the 3 repos: glob 27 (14 nl + 13 literal), regex 14, lexical 15,
structural 13, semantic 13, usage 13. **Expandable** — add more
targets to the specs if the harness shows noisy per-mechanism signal.

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

Glob records additionally carry `glob_kind`:

```json
{
  "id": "gin-glob-lit-1",
  "repo": "gin",
  "query": "internal/**/*.go",
  "intended_mechanism": "glob",
  "glob_kind": "literal",
  "gold": ["internal/bytesconv/bytesconv.go", "..."],
  "match_granularity": "file",
  "source": "derived",
  "note": "glob=internal/**/*.go"
}
```

- `source: "derived"` — gold is reproduced by re-running the generator.
- `source: "curated"` — gold is committed data (semantic/usage); edit
  `curated_<repo>.json` by hand, never regenerate it.
- `glob_kind: "literal"` — query is a literal glob pattern; tests the
  `file_pattern` router dispatch.
- `glob_kind: "nl"` — query is a natural-language description of a glob;
  tests NL routing. `glob_kind` is only present on `intended_mechanism: "glob"`
  records.

## Regenerate

```bash
cd benchmarks/relevance/fixtures/route_stress

# gin (Go)
python3 gen_route_stress.py \
  --repo /path/to/gin --lang go --repo-name gin \
  --spec spec_gin.json --curated curated_gin.json \
  --out gin_route_stress.json

# flask (Python)
python3 gen_route_stress.py \
  --repo /path/to/flask --lang python --repo-name flask \
  --spec spec_flask.json --curated curated_flask.json \
  --out flask_route_stress.json

# platform (TS+Dart monorepo). --lang typescript is the DEFAULT; Dart targets
# in the spec carry "lang":"dart" and Dart regexes carry their own "lang", so a
# single run derives both languages correctly.
python3 gen_route_stress.py \
  --repo /path/to/platform --lang typescript --repo-name platform \
  --spec spec_platform.json --curated curated_platform.json \
  --out platform_route_stress.json
```

The derived gold is reproducible: a clean re-run is byte-identical (verified for
all 3 repos). The generator exits non-zero if any record has empty gold (a sign
the target moved). Requires `ripgrep` (`rg`) on PATH and a `git` checkout of the
target repo (it uses `git ls-files`; falls back to a dot-dir-skipping walk
otherwise).

### Adding a repo / language

Add a `LANG_PATTERNS[<lang>]` entry (globs + `def` + `is_def_line` regexes),
write a `spec_<repo>.json` naming the targets, hand-author `curated_<repo>.json`,
and run the generator. For a **monorepo**, put targets for every language in one
spec and tag each lexical/structural item (and each language-specific regex) with
`"lang": "<lang>"`; regexes may also carry an explicit `"globs"` list. Nothing in
the generator is repo-specific beyond the per-language def/call patterns.

## Status

**Three repos shipped:** gin (Go, 34), flask (Python, 34), platform (TS+Dart
monorepo slice, 27). The platform slice validates cross-language + monorepo
generalization with a small, deliberately-unambiguous target set. The corpus is
expandable per mechanism once the route-running harness (a later task) shows
where the per-mechanism signal is noisy.

Literal-glob (`glob_kind: "literal"`) variants were added in a second pass
(4 gin, 4 flask, 5 platform = 13 new records). These allow the `file_pattern`
router to be evaluated cleanly: the original NL-glob queries always scored 0 on
a `file_pattern`-route eval, making the route untestable. Both subtypes are
retained — NL variants test router NL-handling, literal variants test the
`file_pattern` dispatch path. Note: `**`-in-middle patterns are avoided where
Python's `Path.match` does not traverse all depth levels; patterns were chosen
so the generator's filesystem walk produces a **complete**, reproducible gold set.
