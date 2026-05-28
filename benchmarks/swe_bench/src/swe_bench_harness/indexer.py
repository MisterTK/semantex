"""Drive `semantex index` per repo. Records success/failure cleanly so the
orchestrator can fall back to baseline-tools-only for failed indexes."""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IndexResult:
    ok: bool
    path: Path
    error: str = ""
    duration_secs: float = 0.0


def index_repo(
    *, repo_path: Path, semantex_binary: str, timeout_secs: int = 600
) -> IndexResult:
    """Run `semantex index .` in repo_path. Returns IndexResult."""
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [semantex_binary, "index", "."],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout_secs,
        )
    except subprocess.TimeoutExpired:
        return IndexResult(
            ok=False,
            path=repo_path / ".semantex",
            error=f"timeout after {timeout_secs}s",
            duration_secs=time.monotonic() - start,
        )

    duration = time.monotonic() - start
    if proc.returncode != 0:
        return IndexResult(
            ok=False,
            path=repo_path / ".semantex",
            error=proc.stderr.strip() or proc.stdout.strip(),
            duration_secs=duration,
        )
    return IndexResult(ok=True, path=repo_path / ".semantex", duration_secs=duration)
