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

Forced-route mode (route-stress eval): pass `route=<name>` to `search(...)` to
measure a SPECIFIC retrieval route instead of the default hybrid search. The
client then invokes `semantex agent <query> --route <name> --json-hits`, which
returns the engine's structured ranked hits as a JSON array in the SAME shape
as `search --json` (each item has a repo-relative `file`). `parse_results`
consumes it unchanged, so the file-level gold matcher works as-is. The
`ablation` arg is ignored in route mode — the route owns dense/sparse mechanism
selection. Retrieval routes: file_pattern (files), regex, exact_symbol (exact),
structural, semantic (search), analytical, exhaustive. Synthesis routes (deep,
architecture, feature_planning) are prose-only and return an empty hit list.
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

from .types import RankedResult


_ABLATIONS = {"sparse-only", "dense-only", "hybrid", "rerank"}

# Forced retrieval routes the agent `--json-hits` path returns file-bearing
# ranked hits for. Synthesis routes (deep / architecture / feature_planning)
# are prose-only and excluded — they return an empty hit list.
_RETRIEVAL_ROUTES = {
    "file_pattern",
    "files",  # alias for file_pattern
    "regex",
    "exact_symbol",
    "exact",  # alias for exact_symbol
    "structural",
    "semantic",
    "search",  # alias for semantic
    "analytical",
    "exhaustive",
}


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
        args = [self.binary, "--json", "--no-content", "-m", str(k)]
        if ablation == "sparse-only":
            args.append("--sparse-only")
        elif ablation == "dense-only":
            args.append("--dense-only")
        elif ablation == "rerank":
            args.append("--rerank")
        # hybrid: no extra flag.
        # Put the query LAST, after a `--` end-of-options separator, so a query
        # that starts with dashes (real CSN docstrings can, e.g. "---- utils ----")
        # is never mis-parsed by clap as a CLI flag.
        args += ["--", query]
        return args

    def _build_route_args(self, query: str, *, route: str) -> list[str]:
        """Build the forced-route command:
        `semantex agent --route <route> --json-hits -- <query>`.

        `--json-hits` returns the engine's structured ranked hits (a JSON array
        of SearchResultItem, same shape as `search --json`). The route owns
        mechanism selection, so no ablation/-m flags are passed; budget controls
        how many hits a route returns. The query is the LAST arg behind `--`.
        """
        if route not in _RETRIEVAL_ROUTES:
            raise ValueError(
                f"unknown/unsupported route {route!r}; expected a retrieval route "
                f"from {sorted(_RETRIEVAL_ROUTES)}"
            )
        return [
            self.binary,
            "agent",
            "--route",
            route,
            "--json-hits",
            "--",
            query,
        ]

    def _build_env(self) -> dict:
        env = os.environ.copy()
        env["SEMANTEX_QUIET_LIMITS"] = "1"
        # Canonical A/B measurement config: adaptive result sizing OFF. Adaptive
        # pruning (confidence threshold + per-file dedup) clips ~45% of
        # recoverable recall before any feature runs, so feature A/Bs measured
        # with it ON are invalid. It stays ON in the product (the -18% agent-CCB
        # feature) but must be OFF for relevance A/Bs. setdefault lets an explicit
        # export still measure the adaptive-ON behaviour. See
        # docs/superpowers/plans/2026-06-01-why-no-feature-uplift-rootcause.md §2.
        env.setdefault("SEMANTEX_ADAPTIVE_SIZING", "0")
        if self.embedder:
            # Canonical embedder selector (integration §4 D-env-knob).
            env["SEMANTEX_EMBEDDER"] = self.embedder
        return env

    def reset_daemon(self) -> None:
        """Stop any running daemon for this corpus so the next search spawns a
        fresh one under the locked A/B env.

        The daemon caches its config (incl. adaptive_sizing) at spawn time and
        lives 30 min idle, so a stale daemon from a prior run would silently
        serve A/B queries under the wrong config. Best-effort: `stop` is a no-op
        when no daemon is running, and we never raise on its result.
        """
        subprocess.run(
            [self.binary, "stop", "."],
            cwd=self.corpus_dir,
            capture_output=True,
            text=True,
            env=self._build_env(),
            check=False,
        )

    def search(
        self,
        query_id: str,
        query: str,
        *,
        ablation: str,
        k: int,
        route: Optional[str] = None,
    ) -> RankedResult:
        """Run a query and return its ranked, repo-relative-file results.

        Default (route=None): the hybrid `search --json` path under `ablation`.
        Forced-route (route set): the `agent --route <route> --json-hits` path,
        which returns the engine's ranked hits for a SPECIFIC retrieval route in
        the same JSON shape. `ablation` is ignored when `route` is set — the
        route owns mechanism selection. The result shape is identical either
        way, so the file-level gold matcher consumes it unchanged.
        """
        if route is not None:
            args = self._build_route_args(query, route=route)
            failure_label = "agent --json-hits"
        else:
            args = self._build_args(query, ablation=ablation, k=k)
            failure_label = "search"
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
                f"semantex {failure_label} failed (rc={proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        return parse_results(query_id, proc.stdout)
