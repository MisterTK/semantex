"""External ripgrep keyword baseline — NOT semantex.

A naive "what would grep alone find" comparison point for the SWE-bench
file-localisation task: derive keywords from the issue title+body, count
keyword hits per file with `rg`, and rank files by hit count. No semantex, no
index, no daemon — pure lexical presence over the raw repo tree at its base
commit. This is the floor every retrieval-quality claim should be measured
against.

Deterministic: a fixed keyword-extraction regex + stopword list, and ties in
the ripgrep hit-count ranking are broken by file path, so re-running against
an unchanged repo tree always produces the same ranked list.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

# Anything vaguely identifier-shaped: snake_case, camelCase/PascalCase, dotted
# module paths, CONSTANT_CASE — the tokens an issue report actually names
# (function/class/variable names, module paths) rather than prose filler.
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}(?:\.[A-Za-z_][A-Za-z0-9_]{2,})*")

# Generic English + issue-boilerplate stopwords that would otherwise dominate
# every query's keyword set and swamp the file ranking with noise hits.
_STOPWORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "when", "does", "not", "into",
    "from", "have", "has", "been", "were", "was", "are", "you", "your", "but",
    "should", "would", "could", "will", "can", "cannot", "error", "issue",
    "bug", "fix", "fixed", "please", "thanks", "thank", "code", "example",
    "test", "tests", "file", "files", "line", "lines", "raise", "raises",
    "raised", "return", "returns", "returned", "value", "values", "true",
    "false", "none", "self", "def", "class", "import", "use", "used", "using",
    "also", "then", "than", "these", "those", "what", "which", "who", "how",
    "why", "there", "here", "about", "expected", "actual", "output", "input",
})


def extract_keywords(text: str, *, max_keywords: int = 12) -> list[str]:
    """Pull salient identifier-like tokens from issue title+body.

    Deterministic: first-seen order, deduped case-insensitively (stopword
    filtering is case-insensitive) but original casing is preserved so
    ripgrep's exact match still hits CamelCase symbol names in code.
    """
    seen: set[str] = set()
    out: list[str] = []
    for m in _IDENTIFIER_RE.finditer(text):
        tok = m.group(0)
        low = tok.lower()
        if low in _STOPWORDS or low.isdigit():
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(tok)
        if len(out) >= max_keywords:
            break
    return out


def rank_files_by_keyword_hits(
    corpus_dir: Path, keywords: list[str], *, rg_binary: str = "rg", timeout_secs: int = 60
) -> list[str]:
    """Rank corpus files by total ripgrep match count across `keywords`.

    One `rg --json` invocation with one `-e <kw>` per keyword (fixed-string,
    case-insensitive, one match counted per hit line, `--no-heading`). ripgrep
    skips hidden directories (`.git`, `.semantex`) by default, so no explicit
    excludes are needed. Ties in match count are broken by file path
    (ascending) so the ranking is fully deterministic.

    Raises FileNotFoundError if `rg_binary` isn't on PATH (the baseline is
    then reported as unavailable rather than silently empty).
    """
    if not keywords:
        return []
    if shutil.which(rg_binary) is None:
        raise FileNotFoundError(
            f"{rg_binary!r} not found on PATH; install ripgrep to run the "
            f"keyword baseline"
        )
    cmd = [rg_binary, "--json", "--fixed-strings", "--ignore-case", "--no-heading"]
    for kw in keywords:
        cmd += ["-e", kw]
    cmd.append(".")
    proc = subprocess.run(
        cmd, cwd=corpus_dir, capture_output=True, text=True, timeout=timeout_secs
    )
    # rg exits 1 when a search completes with zero matches — not an error.
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"rg failed (rc={proc.returncode}): {proc.stderr.strip()}")

    counts: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") != "match":
            continue
        path = evt["data"]["path"]["text"]
        path = path[2:] if path.startswith("./") else path
        counts[path] = counts.get(path, 0) + 1

    return [p for p, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
