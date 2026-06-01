"""SWE-bench localization loader.

Gold = the set of files changed by an instance's gold `patch`. Each Verified
instance becomes one query (its problem_statement); matching is file-level.
The corpus is the checked-out repo at $SWE_BENCH_REPO_CACHE/<instance_id>/,
shared with benchmarks/swe_bench's Phase-A cache.

Note: swe_bench_harness.dataset.Instance carries no `patch` field, so this
loader reads the raw HF rows / local JSON directly (Task 0.1 Step 6).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from ..indexing import ensure_index
from ..types import EvalCorpus, Query


# Matches the "+++ b/<path>" header line of each file in a unified diff.
_PLUS_FILE_RE = re.compile(r"^\+\+\+ (?:b/)?(.+?)\s*$", re.MULTILINE)


def changed_files_from_patch(patch: str) -> list[str]:
    """Files touched by a unified-diff patch, in first-seen order, deduped.

    Uses the "+++ " header (post-image path). Skips /dev/null (deletions);
    new files take their b/ path. Strips a leading 'b/'.
    """
    files: list[str] = []
    for m in _PLUS_FILE_RE.finditer(patch):
        path = m.group(1).strip()
        if path == "/dev/null":
            continue
        if path not in files:
            files.append(path)
    return files


def _repo_cache_dir() -> Path:
    return Path(os.environ.get("SWE_BENCH_REPO_CACHE", Path.home() / ".swe_bench_repos"))


def load_swe_loc_queries(
    *,
    local_path: Optional[Path] = None,
    instance_ids: Optional[set[str]] = None,
) -> list[Query]:
    """Load Verified rows (local JSON or HF) → one file-localisation Query each.

    `instance_ids`, if given, restricts to those ids (e.g. the Phase-A 100).
    """
    if local_path is not None:
        rows = json.loads(Path(local_path).read_text())
    else:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        rows = list(ds)

    queries: list[Query] = []
    for r in rows:
        if instance_ids is not None and r["instance_id"] not in instance_ids:
            continue
        gold = tuple(changed_files_from_patch(r["patch"]))
        if not gold:
            continue
        queries.append(
            Query(query_id=r["instance_id"], text=r["problem_statement"], gold_doc_ids=gold)
        )
    return queries


def build_swe_loc_corpus(
    *,
    instance_id: str,
    query: Query,
    semantex_binary: str,
) -> EvalCorpus:
    """Wrap one already-checked-out, indexed instance repo as a single-query corpus.

    The repo must already exist under $SWE_BENCH_REPO_CACHE/<instance_id>/ (run
    benchmarks/swe_bench/scripts/pre_index.py first, or ensure_index builds it).
    """
    corpus_dir = _repo_cache_dir() / instance_id
    if not corpus_dir.is_dir():
        raise FileNotFoundError(
            f"repo for {instance_id} not found at {corpus_dir}; "
            f"run benchmarks/swe_bench/scripts/pre_index.py first"
        )
    ensure_index(corpus_dir=corpus_dir, semantex_binary=semantex_binary)
    return EvalCorpus(
        name=f"swe-loc/{instance_id}",
        documents=(),
        queries=(query,),
        corpus_dir=corpus_dir,
    )
