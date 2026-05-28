"""Git checkout helper. Idempotent — safe to call repeatedly per instance."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepoCheckout:
    path: Path
    sha: str


def _run(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


def checkout(*, repo_url: str, sha: str, dest: Path) -> RepoCheckout:
    """Clone repo_url to dest (if absent) and hard-reset to sha. Idempotent."""
    dest = Path(dest)
    if not (dest / ".git").is_dir():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--quiet", repo_url, str(dest)])
    _run(["git", "fetch", "--quiet", "origin", sha], cwd=dest)
    _run(["git", "reset", "--quiet", "--hard", sha], cwd=dest)
    _run(["git", "clean", "-qfdx"], cwd=dest)
    return RepoCheckout(path=dest, sha=sha)
