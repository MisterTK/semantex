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

The `rerank` ablation ALSO sets SEMANTEX_RERANKER=on in the subprocess env
(see `_build_env`) — the CLI's `--rerank` flag alone is not sufficient: it is
off-by-default at TWO independent gates (config.rerank/query.use_rerank from
`--rerank`, and the SEMANTEX_RERANKER master switch that guards model-weight
loading), and this client must satisfy both or the rerank stage silently
no-ops. Because SEMANTEX_RERANKER is read once at daemon-process spawn time,
not per request, a caller measuring the `rerank` ablation must
`reset_daemon()` first if a daemon from an earlier (non-rerank) arm may
already be running — see `swe_loc_runner.run_instance`. First use downloads
the cross-encoder model weights (network required); subsequent runs are
cached.

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
    """Parse the `semantex search --json` array into a RankedResult.

    Always stashes the raw per-hit dicts on `.raw` (empty list -> empty tuple)
    so a caller who asked for content (via `with_content=True`) can measure
    payload size (tokens-returned) without a second parse pass.
    """
    data = json.loads(stdout) if stdout.strip() else []
    doc_ids = tuple(_doc_id(it) for it in data)
    files = tuple(it["file"] for it in data)
    return RankedResult(query_id=query_id, ranked_doc_ids=doc_ids, ranked_files=files, raw=tuple(data))


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

    def _build_args(
        self, query: str, *, ablation: str, k: int, with_content: bool = False
    ) -> list[str]:
        if ablation not in _ABLATIONS:
            raise ValueError(f"unknown ablation {ablation!r}; expected one of {_ABLATIONS}")
        args = [self.binary, "--json", "-m", str(k)]
        if not with_content:
            # Default: paths + metadata only (smaller, faster subprocess I/O).
            # with_content=True omits this flag so `.raw[i]["content"]` is
            # populated, for tokens-returned instrumentation.
            args.append("--no-content")
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

    def _build_env(self, *, rerank: bool = False) -> dict:
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
        if rerank:
            # TWO independent gates must both be true, and neither is carried by
            # the `--rerank` CLI flag alone when a request goes through the
            # daemon (confirmed empirically — see rerank-experiment.md):
            #
            # 1. SEMANTEX_RERANKER=on — the master switch RerankerEngine::
            #    from_config checks before it will construct a reranker or
            #    download weights at all (S3 off-by-default safety contract;
            #    search/fastembed_reranker.rs::reranker_enabled).
            # 2. SEMANTEX_RERANK=1 — sets `SemantexConfig.rerank` on whichever
            #    process actually EVALUATES `query.use_rerank && self.config.
            #    rerank` (hybrid.rs). For a daemon-served query this is the
            #    DAEMON's own config, loaded independently at ITS spawn time —
            #    NOT the short-lived CLI client process's config (the client's
            #    `--rerank` flag only sets the per-query `use_rerank` field
            #    forwarded over the wire; the daemon is auto-spawned with just
            #    `serve <path>`, no `--rerank`, so its static config.rerank
            #    stays false, and `use_rerank && config.rerank` short-circuits
            #    false, WITHOUT even attempting to load the reranker — no
            #    error, no weight download, just silently identical to
            #    `hybrid`) unless this env var is also set.
            #
            # Both are read once at process spawn time (never per-request), so
            # the caller (swe_loc_runner) must force a fresh daemon
            # (reset_daemon) before the first rerank-ablation query — an
            # already-running daemon spawned by an earlier arm fixed its env
            # before these flags existed. setdefault so an explicit caller
            # override still wins.
            env.setdefault("SEMANTEX_RERANKER", "on")
            env.setdefault("SEMANTEX_RERANK", "1")
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
        with_content: bool = False,
    ) -> RankedResult:
        """Run a query and return its ranked, repo-relative-file results.

        Default (route=None): the hybrid `search --json` path under `ablation`.
        Forced-route (route set): the `agent --route <route> --json-hits` path,
        which returns the engine's ranked hits for a SPECIFIC retrieval route in
        the same JSON shape. `ablation` is ignored when `route` is set — the
        route owns mechanism selection. The result shape is identical either
        way, so the file-level gold matcher consumes it unchanged.

        `with_content=True` (search path only; `agent --json-hits` always
        returns content, there's no equivalent flag) omits `--no-content` so
        `.raw[i]["content"]` is populated — used to estimate tokens-returned.
        """
        if route is not None:
            args = self._build_route_args(query, route=route)
            failure_label = "agent --json-hits"
        else:
            args = self._build_args(query, ablation=ablation, k=k, with_content=with_content)
            failure_label = "search"
        return self._run(args, failure_label, query_id, rerank=(ablation == "rerank"))

    def search_agent_auto(self, query_id: str, query: str) -> RankedResult:
        """Run `agent --json-hits` with NO forced route: the engine's own
        keyword classifier (see semantex_core::search::agent_classifier)
        picks the retrieval mechanism, same as a real user's unforced `agent`
        call. This is the "semantex agent routed search" arm — distinct from
        `search(..., route=<name>)`, which OVERRIDES the classifier to force
        one specific mechanism (used by route_eval's oracle-regret sweep).

        Requires an already-running daemon (`agent` has no search-style
        auto-spawn); the caller must start one first (see
        relevance_harness.route_eval.start_daemon).
        """
        args = [self.binary, "agent", "--json-hits", "--", query]
        return self._run(args, "agent --json-hits (auto-routed)", query_id)

    def _run(
        self, args: list[str], failure_label: str, query_id: str, *, rerank: bool = False
    ) -> RankedResult:
        proc = subprocess.run(
            args,
            cwd=self.corpus_dir,
            capture_output=True,
            text=True,
            timeout=self.timeout_secs,
            env=self._build_env(rerank=rerank),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"semantex {failure_label} failed (rc={proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        return parse_results(query_id, proc.stdout)
