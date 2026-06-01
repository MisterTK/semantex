"""Subprocess wrapper around `semantex search --json`. One job: run a query
under a given ablation/backend and return a normalised RankedResult.

Ablation -> CLI mapping (verified in Task 0.1):
  sparse-only -> --sparse-only
  dense-only  -> --dense-only
  hybrid      -> (neither flag; hybrid is the default)
  rerank      -> hybrid + --rerank
Embedder selection -> SEMANTEX_EMBEDDER env var (lateon-colbert | coderank-137m).
Canonical per integration §4 D-env-knob; SEMANTEX_DENSE_BACKEND is a kept-live
deprecated alias. All runs add: --json --no-content -m <k>.
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

from .types import RankedResult


_ABLATIONS = {"sparse-only", "dense-only", "hybrid", "rerank"}


def _doc_id(item: dict) -> str:
    """Stable doc id minted from file + line range; loaders mint ids the same way."""
    return f"{item['file']}:{item.get('start_line', 0)}-{item.get('end_line', 0)}"


def parse_results(query_id: str, stdout: str) -> RankedResult:
    """Parse the `semantex search --json` array into a RankedResult."""
    data = json.loads(stdout) if stdout.strip() else []
    doc_ids = tuple(_doc_id(it) for it in data)
    files = tuple(it["file"] for it in data)
    return RankedResult(query_id=query_id, ranked_doc_ids=doc_ids, ranked_files=files)


class SemantexClient:
    def __init__(
        self,
        *,
        semantex_binary: str,
        corpus_dir: str,
        embedder: Optional[str] = None,
        timeout_secs: int = 120,
    ):
        self.binary = semantex_binary
        self.corpus_dir = corpus_dir
        # Embedder id (e.g. "lateon-colbert" | "coderank-137m"), set via the
        # canonical SEMANTEX_EMBEDDER env var (integration §4 D-env-knob).
        self.embedder = embedder
        self.timeout_secs = timeout_secs

    def _build_args(self, query: str, *, ablation: str, k: int) -> list[str]:
        if ablation not in _ABLATIONS:
            raise ValueError(f"unknown ablation {ablation!r}; expected one of {_ABLATIONS}")
        args = [self.binary, query, "--json", "--no-content", "-m", str(k)]
        if ablation == "sparse-only":
            args.append("--sparse-only")
        elif ablation == "dense-only":
            args.append("--dense-only")
        elif ablation == "rerank":
            args.append("--rerank")
        # hybrid: no extra flag
        return args

    def _build_env(self) -> dict:
        env = os.environ.copy()
        env["SEMANTEX_QUIET_LIMITS"] = "1"
        if self.embedder:
            # Canonical embedder selector (integration §4 D-env-knob).
            env["SEMANTEX_EMBEDDER"] = self.embedder
        return env

    def search(self, query_id: str, query: str, *, ablation: str, k: int) -> RankedResult:
        args = self._build_args(query, ablation=ablation, k=k)
        proc = subprocess.run(
            args,
            cwd=self.corpus_dir,
            capture_output=True,
            text=True,
            timeout=self.timeout_secs,
            env=self._build_env(),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"semantex search failed (rc={proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        return parse_results(query_id, proc.stdout)
