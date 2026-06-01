"""Locate-or-build a semantex index for a corpus directory.

Reuses swe_bench's indexer so the SWE-loc path shares the Phase-A index cache.
A completed index is marked by `.semantex/meta.json` with chunk_count > 0
(same convention as benchmarks/swe_bench/scripts/pre_index.py)."""
from __future__ import annotations

import json
from pathlib import Path

from swe_bench_harness.indexer import index_repo


def index_is_complete(corpus_dir: Path) -> bool:
    meta = Path(corpus_dir) / ".semantex" / "meta.json"
    if not meta.exists():
        return False
    try:
        data = json.loads(meta.read_text())
    except (ValueError, OSError):
        return False
    return int(data.get("chunk_count", 0)) > 0


def ensure_index(
    *, corpus_dir: Path, semantex_binary: str, timeout_secs: int = 7200
) -> Path:
    """Build a semantex index in `corpus_dir` if one isn't already complete.

    Returns the `.semantex` dir path. Raises RuntimeError on a failed build.
    """
    corpus_dir = Path(corpus_dir)
    sx = corpus_dir / ".semantex"
    if index_is_complete(corpus_dir):
        return sx
    result = index_repo(
        repo_path=corpus_dir, semantex_binary=semantex_binary, timeout_secs=timeout_secs
    )
    if not result.ok:
        raise RuntimeError(f"semantex index failed for {corpus_dir}: {result.error}")
    return sx
