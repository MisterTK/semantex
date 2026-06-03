#!/usr/bin/env python3
"""Generate the route-stress evaluation corpus for one repo.

WHY this corpus exists
----------------------
The existing relevance corpora (CoIR / CodeSearchNet) are all NL->code
*semantic*, single-gold queries. They cannot exercise — let alone measure — a
query router's NON-semantic mechanisms (glob, regex, exact-symbol, structural
call-graph). This corpus fills that gap with MULTI-GOLD queries, each tagged
with the retrieval *mechanism* it is meant to stress, so a later harness can ask
two questions: (1) does the router pick the right mechanism, and (2) does each
mechanism earn its keep on the queries that should favour it.

WHAT this script does
---------------------
For a given repo it DERIVES the mechanical gold for four mechanisms directly
from the repo source (no index, no search engine), then merges in the
hand-curated records (semantic / usage) from a sibling JSON file:

  glob       — filesystem walk; gold = matching file paths
  regex      — ripgrep over source; gold = files containing >=1 match
  lexical    — ripgrep for the language's definition pattern; gold = def file(s)
  structural — ripgrep for `<symbol>(` call sites MINUS definition lines;
               gold = files that CALL the symbol (cross-checked against
               graphify-out/graph.json `calls` edges when present)

GOLD CONVENTION (see README.md §"Gold convention" for the full justification)
----------------------------------------------------------------------------
Gold ids are **repo-relative file paths** and matching is FILE-granularity
(`match_granularity: "file"`). semantex stores and returns chunk file_path
relative to the project root (crates/.../index/builder.rs:251
`strip_prefix(&project_path)`), and the harness's file-mode matcher
(relevance_harness/runner.py `_relevance_vector`, match_mode="file") compares a
result's `file` field verbatim against gold. We deliberately do NOT use
chunk-level `file:start-end` doc_ids for the DERIVED mechanisms: a chunk span is
AST-chunker-determined (e.g. gin `func New` at line 202 lives in chunk 202-233,
`Context struct` at line 61 in chunk 61-97) and is NOT reproducible from source
alone, so a grep-derived `file:line-line` id would silently fail to match.
File-granularity gold is reproducible, chunker-version-stable, and unambiguous.

Repo-agnostic: pass --repo and --lang. Only the per-language def/call regexes
(LANG_PATTERNS) are language-specific; nothing is gin-specific.

Usage:
  python3 gen_route_stress.py --repo /path/to/gin --lang go \
      --curated curated_gin.json --out gin_route_stress.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Per-language patterns. `def` builds a ripgrep regex that locates a symbol's
# DEFINITION line; `is_def_line` recognises a definition line so structural
# derivation can exclude it from call sites. Adding a language = adding an entry
# here; nothing else in this file is language-aware.
# --------------------------------------------------------------------------- #
LANG_PATTERNS = {
    "go": {
        "globs": ["*.go"],
        # type X <anything> (struct/interface/func/slice/map/alias) OR a free
        # function `func X(` OR a method `func (recv) X(`. NOTE: a method whose
        # NAME equals `sym` will also match — pick distinctively-named symbols
        # (the favor-unambiguous-gold rule) or accept the small method-collision
        # gold. The structural derivation, by contrast, excludes def lines.
        "def": lambda sym: (
            rf"^(?:type\s+{re.escape(sym)}\b"
            rf"|func\s+{re.escape(sym)}\s*\("
            rf"|func\s+\([^)]*\)\s+{re.escape(sym)}\s*\()"
        ),
        # a line that defines (rather than calls) the symbol
        "is_def_line": lambda sym: re.compile(
            rf"^\s*(?:type\s+{re.escape(sym)}\b|func\s+(?:\([^)]*\)\s+)?{re.escape(sym)}\s*\()"
        ),
    },
    "python": {
        "globs": ["*.py"],
        "def": lambda sym: rf"^\s*(?:def|class)\s+{re.escape(sym)}\b",
        "is_def_line": lambda sym: re.compile(rf"^\s*(?:def|class)\s+{re.escape(sym)}\b"),
    },
    # Extend as needed (ts/java/...); the pilot only uses "go".
}


def _walk_files(repo: Path) -> list[str]:
    """All tracked source-ish files as repo-relative posix paths.

    Uses `git ls-files` when the repo is a git checkout (respects .gitignore and
    excludes build/index artefacts like .semantex/ and graphify-out/);
    otherwise falls back to a plain walk that skips dotdirs.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "ls-files"],
            capture_output=True, text=True, check=True,
        ).stdout
        files = [ln for ln in out.splitlines() if ln.strip()]
        if files:
            return files
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    files = []
    for p in repo.rglob("*"):
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(repo).parts):
            files.append(p.relative_to(repo).as_posix())
    return files


def _rg_files(repo: Path, pattern: str, globs: list[str], *, exclude_tests: bool = False) -> list[str]:
    """Repo-relative files with >=1 line matching `pattern` (ripgrep -l)."""
    cmd = ["rg", "-l", "--no-heading", "--sort", "path"]
    for g in globs:
        cmd += ["--glob", g]
    if exclude_tests:
        cmd += ["--glob", "!*_test.*", "--glob", "!test_*"]
    cmd += [pattern, str(repo)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode not in (0, 1):  # 1 = no matches, not an error
        raise RuntimeError(f"ripgrep failed ({proc.returncode}): {proc.stderr.strip()}")
    rel = []
    for ln in proc.stdout.splitlines():
        if ln.strip():
            rel.append(Path(ln).resolve().relative_to(repo.resolve()).as_posix())
    return sorted(set(rel))


def _rg_hits(repo: Path, pattern: str, globs: list[str], *, exclude_tests: bool = False) -> list[tuple[str, int, str]]:
    """(relpath, line_no, line_text) for every match of `pattern`."""
    cmd = ["rg", "-n", "--no-heading", "--sort", "path"]
    for g in globs:
        cmd += ["--glob", g]
    if exclude_tests:
        cmd += ["--glob", "!*_test.*", "--glob", "!test_*"]
    cmd += [pattern, str(repo)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"ripgrep failed ({proc.returncode}): {proc.stderr.strip()}")
    hits = []
    for ln in proc.stdout.splitlines():
        # format: <path>:<lineno>:<text>
        m = re.match(r"^(.*?):(\d+):(.*)$", ln)
        if not m:
            continue
        rel = Path(m.group(1)).resolve().relative_to(repo.resolve()).as_posix()
        hits.append((rel, int(m.group(2)), m.group(3)))
    return hits


# --------------------------------------------------------------------------- #
# Per-mechanism derivation
# --------------------------------------------------------------------------- #
def derive_glob(repo: Path, glob_pattern: str) -> list[str]:
    """Gold = repo-relative files whose path matches `glob_pattern`.

    `glob_pattern` is a path glob like "*_test.go" or "render/*.go".
    """
    files = _walk_files(repo)
    pat = glob_pattern
    matched = [f for f in files if Path(f).match(pat) or Path(f).match(f"**/{pat}")]
    return sorted(set(matched))


def derive_regex(repo: Path, pattern: str, globs: list[str], *, exclude_tests: bool) -> list[str]:
    """Gold = repo-relative files containing >=1 line matching `pattern`."""
    return _rg_files(repo, pattern, globs, exclude_tests=exclude_tests)


def derive_lexical(repo: Path, symbol: str, lang: str) -> list[str]:
    """Gold = file(s) holding the DEFINITION of `symbol` (def-pattern grep)."""
    spec = LANG_PATTERNS[lang]
    return _rg_files(repo, spec["def"](symbol), spec["globs"])


def derive_structural(repo: Path, symbol: str, lang: str, *, exclude_tests: bool) -> list[str]:
    """Gold = file(s) that CALL `symbol` (call sites), excluding its definition.

    A call site is a `<symbol>(` occurrence on a line that is NOT a definition
    of `symbol`. We match `\\b<symbol>\\(`, then drop definition lines via the
    language's `is_def_line`. Returns repo-relative call-site files.
    """
    spec = LANG_PATTERNS[lang]
    call_re = rf"\b{re.escape(symbol)}\("
    is_def = spec["is_def_line"](symbol)
    files = set()
    for rel, _ln, text in _rg_hits(repo, call_re, spec["globs"], exclude_tests=exclude_tests):
        if is_def.search(text):
            continue
        files.add(rel)
    return sorted(files)


def crosscheck_structural_graph(repo: Path, symbol: str, derived: list[str]) -> Optional[dict]:
    """If graphify-out/graph.json exists, return the call-site files its `calls`
    edges attribute to `symbol`, for a sanity cross-check (informational only —
    the ripgrep derivation is authoritative). None if no graph present.
    """
    gpath = repo / "graphify-out" / "graph.json"
    if not gpath.is_file():
        return None
    g = json.loads(gpath.read_text())
    norm = symbol.lower()
    graph_files = set()
    for link in g.get("links", []):
        if link.get("relation") != "calls":
            continue
        # node ids are normalised like gin_routergroup_combinehandlers
        if norm in str(link.get("target", "")).lower():
            sf = link.get("source_file")
            if sf:
                graph_files.add(Path(sf).as_posix())
    return {
        "graph_call_site_files": sorted(graph_files),
        "agrees_with_derived": graph_files.issubset(set(derived)) if graph_files else None,
    }


# --------------------------------------------------------------------------- #
# Record assembly
# --------------------------------------------------------------------------- #
def _rec(id_, repo_name, query, mechanism, gold, *, granularity="file", source="derived", note=None):
    r = {
        "id": id_,
        "repo": repo_name,
        "query": query,
        "intended_mechanism": mechanism,
        "gold": sorted(gold) if isinstance(gold, (list, set, tuple)) else gold,
        "match_granularity": granularity,
        "source": source,
    }
    if note:
        r["note"] = note
    return r


def build_records(repo: Path, repo_name: str, lang: str, spec_path: Path) -> list[dict]:
    """Read the derivation spec (which symbols/patterns/globs to use for this
    repo) and emit the derived records. The spec is small, human-authored DATA
    that names the targets; the GOLD for each is derived mechanically here, so
    re-running reproduces it exactly.
    """
    spec = json.loads(spec_path.read_text())
    records: list[dict] = []

    # glob
    for item in spec.get("glob", []):
        gold = derive_glob(repo, item["pattern"])
        records.append(_rec(item["id"], repo_name, item["query"], "glob", gold,
                            granularity="file", note=f"glob={item['pattern']}"))

    # regex
    for item in spec.get("regex", []):
        gold = derive_regex(repo, item["pattern"], LANG_PATTERNS[lang]["globs"],
                            exclude_tests=item.get("exclude_tests", False))
        records.append(_rec(item["id"], repo_name, item["query"], "regex", gold,
                            granularity="file", note=f"regex={item['pattern']!r}"))

    # lexical / exact-symbol
    for item in spec.get("lexical", []):
        gold = derive_lexical(repo, item["symbol"], lang)
        records.append(_rec(item["id"], repo_name, item["query"], "lexical", gold,
                            granularity="file", note=f"def of {item['symbol']}"))

    # structural / callers
    for item in spec.get("structural", []):
        gold = derive_structural(repo, item["symbol"], lang,
                                 exclude_tests=item.get("exclude_tests", False))
        note = f"callers of {item['symbol']}"
        xc = crosscheck_structural_graph(repo, item["symbol"], gold)
        if xc is not None:
            note += f"; graph-crosscheck agrees={xc['agrees_with_derived']}"
        records.append(_rec(item["id"], repo_name, item["query"], "structural", gold,
                            granularity="file", note=note))

    return records


def merge_curated(records: list[dict], curated_path: Path) -> list[dict]:
    """Append the hand-curated semantic/usage records (committed data, NOT
    regenerated). They already carry gold + source=='curated'.
    """
    if not curated_path.is_file():
        return records
    curated = json.loads(curated_path.read_text())
    return records + curated


def validate(records: list[dict]) -> list[str]:
    """Return a list of problems (empty gold, etc). Empty list == all good."""
    problems = []
    seen = set()
    for r in records:
        if r["id"] in seen:
            problems.append(f"{r['id']}: duplicate id")
        seen.add(r["id"])
        if not r.get("gold"):
            problems.append(f"{r['id']} ({r['intended_mechanism']}): EMPTY gold")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, type=Path, help="path to the target repo")
    ap.add_argument("--lang", required=True, choices=sorted(LANG_PATTERNS), help="primary language")
    ap.add_argument("--repo-name", default=None, help="short repo name (default: dir name)")
    ap.add_argument("--spec", required=True, type=Path,
                    help="derivation spec JSON (targets to derive gold for)")
    ap.add_argument("--curated", type=Path, default=None,
                    help="hand-curated semantic/usage records JSON to merge in")
    ap.add_argument("--out", required=True, type=Path, help="output corpus JSON")
    args = ap.parse_args(argv)

    repo = args.repo.resolve()
    if not repo.is_dir():
        ap.error(f"repo not found: {repo}")
    repo_name = args.repo_name or repo.name

    records = build_records(repo, repo_name, args.lang, args.spec)
    if args.curated:
        records = merge_curated(records, args.curated)

    problems = validate(records)
    if problems:
        print("VALIDATION PROBLEMS:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        # empty derived gold is a hard error (the target is wrong); still write
        # so the author can inspect, but exit non-zero.

    out = {
        "schema": "route_stress/v1",
        "repo": repo_name,
        "repo_path_at_gen": str(repo),
        "match_granularity_default": "file",
        "mechanisms": ["glob", "regex", "lexical", "structural", "semantic", "usage"],
        "records": records,
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    # summary
    from collections import Counter
    counts = Counter(r["intended_mechanism"] for r in records)
    print(f"wrote {len(records)} records to {args.out}")
    for mech in out["mechanisms"]:
        print(f"  {mech:11s}: {counts.get(mech, 0)}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
