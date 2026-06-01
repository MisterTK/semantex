"""Locate-or-build a semantex index for a corpus directory.

A completed index is marked by `.semantex/meta.json` with chunk_count > 0
(same convention as benchmarks/swe_bench/scripts/pre_index.py).

IMPORTANT (D4 embedder A/B): the embedder is authoritative at INDEX time. The
dense backend is persisted in `.semantex/meta.json` (`dense_backend`) and the
searcher honors the on-disk backend regardless of the search-time
`SEMANTEX_EMBEDDER` env. So to actually compare `lateon-colbert` (colbert-plaid)
vs `coderank-137m` (coderank-hnsw) we MUST set SEMANTEX_EMBEDDER in the index
subprocess env, and each embedder must build into its OWN corpus dir (a shared
.semantex would be reused across arms and silently measure the first embedder).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class IndexBuild:
    """Result of an index build (or cache hit)."""
    path: Path
    built: bool          # True if we actually invoked `semantex index`
    duration_secs: float
    peak_rss_mb: Optional[float] = None


def index_is_complete(corpus_dir: Path) -> bool:
    meta = Path(corpus_dir) / ".semantex" / "meta.json"
    if not meta.exists():
        return False
    try:
        data = json.loads(meta.read_text())
    except (ValueError, OSError):
        return False
    return int(data.get("chunk_count", 0)) > 0


def _index_env(embedder: Optional[str]) -> dict:
    env = os.environ.copy()
    env["SEMANTEX_QUIET_LIMITS"] = "1"
    if embedder:
        # Canonical embedder selector (integration §4 D-env-knob). Set at INDEX
        # time so the right dense backend is built + persisted into meta.json.
        env["SEMANTEX_EMBEDDER"] = embedder
    return env


def ensure_index(
    *,
    corpus_dir: Path,
    semantex_binary: str,
    timeout_secs: int = 7200,
    embedder: Optional[str] = None,
) -> IndexBuild:
    """Build a semantex index in `corpus_dir` if one isn't already complete.

    The embedder (if given) is set in the index subprocess env via
    SEMANTEX_EMBEDDER so the matching dense backend is built. Returns an
    IndexBuild (path, whether we built, wall time, and peak RSS via /usr/bin/time
    -l when available). Raises RuntimeError on a failed build.
    """
    corpus_dir = Path(corpus_dir)
    sx = corpus_dir / ".semantex"
    if index_is_complete(corpus_dir):
        return IndexBuild(path=sx, built=False, duration_secs=0.0)

    # Prefer /usr/bin/time -l on macOS to capture peak RSS; fall back to a plain
    # invocation if unavailable. We parse "maximum resident set size" (bytes).
    use_time = Path("/usr/bin/time").exists()
    cmd = ([ "/usr/bin/time", "-l", semantex_binary, "index", "." ]
           if use_time else [semantex_binary, "index", "."])
    start = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=corpus_dir,
        capture_output=True,
        text=True,
        timeout=timeout_secs,
        env=_index_env(embedder),
    )
    duration = time.monotonic() - start
    if proc.returncode != 0:
        raise RuntimeError(
            f"semantex index failed for {corpus_dir} "
            f"(embedder={embedder!r}, rc={proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    peak_rss_mb: Optional[float] = None
    if use_time:
        for ln in proc.stderr.splitlines():
            if "maximum resident set size" in ln:
                try:
                    peak_rss_mb = int(ln.strip().split()[0]) / (1024 * 1024)
                except (ValueError, IndexError):
                    pass
                break
    return IndexBuild(path=sx, built=True, duration_secs=duration, peak_rss_mb=peak_rss_mb)
