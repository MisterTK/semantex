"""Oracle-regret evaluation harness for the semantex query router.

Loads the route_stress corpora (gin / flask / platform), runs every retrieval
route over every query, then aggregates the oracle nDCG, regret, router accuracy,
and a confusion matrix.  Pure computation module — I/O lives in scripts/route_stress.py.

Design notes
------------
* **File dedup**: the ranked file list from each route is deduped by first-occurrence
  before scoring.  Structural + graph routes can surface the same file via multiple
  sections; without dedup, nDCG can exceed 1.0.  dedup_ranked_files() is the fix.
* **Router choice**: obtained via `semantex agent --json <query>` (no --route); the
  `.route` field of the JSON response is the keyword classifier's choice.
* **Synthesis routes** (deep / architecture / feature_planning / analytical / exhaustive):
  when the router picks one of these for a file-gold query we record the outcome as
  "router chose synthesis", set its file-retrieval nDCG to 0.0, and report the count
  separately.
* **Retrieval routes tested**: file_pattern, regex, exact_symbol, structural, semantic.
  (These map to the five `intended_mechanism` categories in the corpus.)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .indexing import ensure_index
from .metrics import ndcg_at_k, recall_at_k


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Retrieval routes we exhaustively score per query.
RETRIEVAL_ROUTES: tuple[str, ...] = (
    "file_pattern",
    "regex",
    "exact_symbol",
    "structural",
    "semantic",
)

#: Routes whose output is prose-only; they return zero file hits.
SYNTHESIS_ROUTES: frozenset[str] = frozenset({
    "deep",
    "architecture",
    "feature_planning",
    "analytical",
    "exhaustive",
})

# Regex to extract "file:start-end" tokens from agent formatted output.
# Matches the first token on result lines like:
#   utils.go:51-53 [text] (0.06)
#   render/json.go:12-30 [method] (0.04)
#   recovery.go:114: func stack(...)        <- regex route: file:line: pattern
_FILE_RANGE_RE = re.compile(r'^([\w./\-]+\.\w+):(\d+)[-:](\d+)?')

# The file_pattern route emits BARE filenames (one per line, no ":line"):
#   auth_test.go
#   binding/binding_test.go
# Match a column-0 path-with-extension and nothing else on the line (no spaces),
# so code/prose lines (which contain spaces) are never mistaken for a filename.
_BARE_FILE_RE = re.compile(r'^([\w./\-]+\.\w+)\s*$')


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class QueryRouteResult:
    """All five route scores + router choice for one query."""
    query_id: str
    query: str
    repo: str
    intended_mechanism: str
    gold_files: frozenset[str]

    # Per-route nDCG@10 and Recall@10 (after file dedup).
    route_ndcg: dict[str, float] = field(default_factory=dict)
    route_recall: dict[str, float] = field(default_factory=dict)
    # Raw ranked (deduped) file lists per route.
    route_ranked_files: dict[str, list[str]] = field(default_factory=dict)

    # Router classifier's chosen route (from `agent --json` without --route).
    router_choice: Optional[str] = None
    # True when router_choice is a synthesis route (no file hits expected).
    router_chose_synthesis: bool = False

    @property
    def oracle_route(self) -> str:
        """Route with the highest nDCG@10 (ties: first RETRIEVAL_ROUTES order)."""
        if not self.route_ndcg:
            return RETRIEVAL_ROUTES[0]
        return max(RETRIEVAL_ROUTES, key=lambda r: self.route_ndcg.get(r, -1.0))

    @property
    def oracle_ndcg(self) -> float:
        return self.route_ndcg.get(self.oracle_route, 0.0)

    @property
    def oracle_routes_tied(self) -> list[str]:
        """All routes that share the oracle nDCG (may be a tie)."""
        best = self.oracle_ndcg
        if best == 0.0:
            return []
        return [r for r in RETRIEVAL_ROUTES if abs(self.route_ndcg.get(r, -1.0) - best) < 1e-9]

    @property
    def router_ndcg(self) -> float:
        """nDCG of the route the keyword router chose (0 if synthesis or unknown)."""
        if self.router_chose_synthesis or self.router_choice is None:
            return 0.0
        return self.route_ndcg.get(self.router_choice, 0.0)

    @property
    def regret(self) -> float:
        return self.oracle_ndcg - self.router_ndcg


@dataclass
class RouteEvalResult:
    """Full evaluation result: per-query breakdown + aggregates."""
    repo: str
    records: list[QueryRouteResult] = field(default_factory=list)

    # Aggregates (populated by aggregate())
    oracle_win_counts: dict[str, int] = field(default_factory=dict)
    per_route_mean_ndcg: dict[str, float] = field(default_factory=dict)
    per_mechanism_route_ndcg: dict[str, dict[str, float]] = field(default_factory=dict)
    per_mechanism_regret: dict[str, float] = field(default_factory=dict)
    per_mechanism_router_accuracy: dict[str, float] = field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    overall_regret: float = 0.0
    overall_router_accuracy: float = 0.0
    synthesis_count: int = 0
    router_matches_oracle_count: int = 0
    total_queries: int = 0


# ---------------------------------------------------------------------------
# File dedup (the critical fix)
# ---------------------------------------------------------------------------

def dedup_ranked_files(ranked_files: list[str]) -> list[str]:
    """Return ranked_files with duplicates removed (first occurrence kept).

    Per-query nDCG is computed over the deduped list so a gold file that recurs
    across structural graph sections is counted at most once, preventing nDCG > 1.0.
    """
    seen: set[str] = set()
    out: list[str] = []
    for f in ranked_files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _relevance_vector(ranked_files: list[str], gold: frozenset[str]) -> list[int]:
    return [1 if f in gold else 0 for f in ranked_files]


# ---------------------------------------------------------------------------
# Formatted output → file list parser
# ---------------------------------------------------------------------------

def parse_files_from_formatted(text: str) -> list[str]:
    """Extract an ordered, deduped file list from agent --json formatted output.

    Handles the line formats emitted by different routes:
      semantic/exact:    ``file.go:51-53 [text] (0.06)``   (column 0)
      regex:             ``file.go:114: func ...``         (column 0)
      structural:        ``  gin.go:662-675 ServeHTTP [method]``  (INDENTED, under
                         ``Target:`` / ``Callers (N):`` / ``Callees (N):`` headers)
      file_pattern:      ``binding/binding_test.go``       (BARE filename, no range)

    A result line is recognised by a ``path.ext:linenumber`` token at the START of
    the (whitespace-stripped) line. This catches both column-0 routes and the
    indented structural-route results, while code/prose lines (which do NOT begin
    with ``path.ext:digits``) are correctly ignored.

    Returns files in appearance order (first-occurrence kept = dedup in-place).
    """
    files: list[str] = []
    seen: set[str] = set()

    def _add(fp: str) -> None:
        if fp not in seen:
            seen.add(fp)
            files.append(fp)

    for raw in text.splitlines():
        if not raw:
            continue
        stripped = raw.lstrip()
        # Skip the route header and any section header / footer markers.
        if stripped.startswith('[route:') or stripped.startswith('['):
            continue
        # A "file:line-range" token at the start of the (stripped) line is a result,
        # whether at column 0 (semantic/regex/exact) or indented (structural).
        m = _FILE_RANGE_RE.match(stripped)
        if m:
            _add(m.group(1))
            continue
        # file_pattern route emits BARE filenames at column 0 (no line range). Only
        # match these when the ORIGINAL line is unindented — an indented bare token
        # would be a structural sub-detail / prose, not a file_pattern hit.
        if raw == stripped:
            bm = _BARE_FILE_RE.match(stripped)
            if bm:
                _add(bm.group(1))
    return files


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _build_env(embedder: Optional[str] = None) -> dict:
    env = os.environ.copy()
    env["SEMANTEX_QUIET_LIMITS"] = "1"
    env.setdefault("SEMANTEX_ADAPTIVE_SIZING", "0")
    if embedder:
        env["SEMANTEX_EMBEDDER"] = embedder
    return env


def _run_route(
    query: str,
    route: str,
    *,
    corpus_dir: str,
    semantex_binary: str,
    timeout_secs: int = 60,
    embedder: Optional[str] = None,
) -> list[str]:
    """Run a forced-route agent query and return a deduped ranked file list.

    Uses ``agent --route <route> --json -- <query>`` which returns an AgentResponse
    JSON with a ``formatted`` field.  We parse file paths from the formatted text.
    Returns an empty list for synthesis routes or on error.
    """
    if route in SYNTHESIS_ROUTES:
        return []
    cmd = [semantex_binary, "agent", "--route", route, "--json", "--", query]
    try:
        proc = subprocess.run(
            cmd,
            cwd=corpus_dir,
            capture_output=True,
            text=True,
            timeout=timeout_secs,
            env=_build_env(embedder),
        )
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode != 0:
        # Return empty list rather than raising — the harness reports zeros.
        return []
    try:
        data = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        return []
    formatted = data.get("formatted", "")
    return dedup_ranked_files(parse_files_from_formatted(formatted))


def _get_router_choice(
    query: str,
    *,
    corpus_dir: str,
    semantex_binary: str,
    timeout_secs: int = 60,
    embedder: Optional[str] = None,
) -> Optional[str]:
    """Return the keyword classifier's chosen route for `query`.

    Runs ``agent --json <query>`` (no --route) and reads the ``.route`` field.
    Returns None on failure.
    """
    cmd = [semantex_binary, "agent", "--json", "--", query]
    try:
        proc = subprocess.run(
            cmd,
            cwd=corpus_dir,
            capture_output=True,
            text=True,
            timeout=timeout_secs,
            env=_build_env(embedder),
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
        return data.get("route")
    except (json.JSONDecodeError, ValueError):
        return None


def reset_daemon(corpus_dir: str, semantex_binary: str, embedder: Optional[str] = None) -> None:
    """Stop any running daemon for corpus_dir (best-effort)."""
    subprocess.run(
        [semantex_binary, "stop", "."],
        cwd=corpus_dir,
        capture_output=True,
        text=True,
        env=_build_env(embedder),
        check=False,
    )


def start_daemon(
    corpus_dir: str,
    semantex_binary: str,
    *,
    embedder: Optional[str] = None,
    wait_secs: float = 30.0,
) -> subprocess.Popen:
    """Spawn a `semantex serve` daemon under the locked A/B env and wait for it.

    Unlike `search` (which has a fast-path auto-spawn), the `agent` subcommand
    requires an ALREADY-RUNNING daemon — it errors with "Daemon not running"
    otherwise. The route_eval harness drives the engine exclusively through
    `agent --route`, so we must start the daemon explicitly after a reset, under
    SEMANTEX_ADAPTIVE_SIZING=0 so the cached config is the canonical A/B lock.

    Returns the Popen handle; the caller stops it via `reset_daemon`. Raises
    RuntimeError if the port file does not appear within `wait_secs`.
    """
    port_file = Path(corpus_dir) / ".semantex" / "semantex.port"
    # Clear any stale port file so we wait for THIS daemon's port.
    if port_file.exists():
        port_file.unlink(missing_ok=True)
    proc = subprocess.Popen(
        [semantex_binary, "serve", "."],
        cwd=corpus_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_build_env(embedder),
    )
    deadline = time.monotonic() + wait_secs
    while time.monotonic() < deadline:
        if port_file.exists():
            return proc
        if proc.poll() is not None:
            raise RuntimeError(
                f"semantex serve exited early (rc={proc.returncode}) for {corpus_dir}"
            )
        time.sleep(0.1)
    proc.terminate()
    raise RuntimeError(
        f"semantex daemon for {corpus_dir} did not come up within {wait_secs}s"
    )


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteStressRecord:
    id: str
    repo: str
    query: str
    intended_mechanism: str
    gold: frozenset[str]


def load_route_stress_corpus(path: Path) -> tuple[str, list[RouteStressRecord]]:
    """Load a route_stress JSON fixture.  Returns (repo_name, records)."""
    data = json.loads(path.read_text())
    repo = data["repo"]
    records = []
    for r in data["records"]:
        records.append(RouteStressRecord(
            id=r["id"],
            repo=r["repo"],
            query=r["query"],
            intended_mechanism=r["intended_mechanism"],
            gold=frozenset(r["gold"]),
        ))
    return repo, records


# ---------------------------------------------------------------------------
# Per-repo evaluation runner
# ---------------------------------------------------------------------------

def evaluate_repo(
    corpus_path: Path,
    repo_dir: str,
    *,
    semantex_binary: str,
    k: int = 10,
    embedder: Optional[str] = None,
    verbose: bool = False,
) -> RouteEvalResult:
    """Run oracle-regret eval for one repo.  Ensures index, (re)starts daemon.

    `agent --route` requires an ALREADY-RUNNING daemon, so we reset any stale one,
    then explicitly spawn a fresh daemon under the SEMANTEX_ADAPTIVE_SIZING=0 lock.
    The daemon is stopped on exit (even on error) to avoid leaking processes.
    """
    repo_name, records = load_route_stress_corpus(corpus_path)

    # Ensure index is built.
    ensure_index(corpus_dir=Path(repo_dir), semantex_binary=semantex_binary, embedder=embedder)

    # Reset any stale daemon, then spawn a fresh one under the A/B lock. The
    # `agent` path does NOT auto-spawn (unlike `search`), so this is required.
    reset_daemon(repo_dir, semantex_binary, embedder)
    start_daemon(repo_dir, semantex_binary, embedder=embedder)

    result = RouteEvalResult(repo=repo_name)

    try:
        for rec in records:
            if verbose:
                print(f"  [{rec.id}] {rec.query[:60]}")

            qr = QueryRouteResult(
                query_id=rec.id,
                query=rec.query,
                repo=rec.repo,
                intended_mechanism=rec.intended_mechanism,
                gold_files=rec.gold,
            )

            # Score every retrieval route.
            for route in RETRIEVAL_ROUTES:
                t0 = time.monotonic()
                ranked = _run_route(
                    rec.query,
                    route,
                    corpus_dir=repo_dir,
                    semantex_binary=semantex_binary,
                    embedder=embedder,
                )
                elapsed = time.monotonic() - t0
                if verbose:
                    print(f"    route={route:<14} hits={len(ranked):3d}  ({elapsed*1000:.0f}ms)")
                qr.route_ranked_files[route] = ranked
                rels = _relevance_vector(ranked, rec.gold)
                n_rel = len(rec.gold)
                qr.route_ndcg[route] = ndcg_at_k([rels], k=k, n_relevant=[n_rel])
                qr.route_recall[route] = recall_at_k([rels], k=k, n_relevant=[n_rel])

            # Get router's classification.
            router_choice = _get_router_choice(
                rec.query,
                corpus_dir=repo_dir,
                semantex_binary=semantex_binary,
                embedder=embedder,
            )
            qr.router_choice = router_choice
            qr.router_chose_synthesis = (
                (router_choice in SYNTHESIS_ROUTES) if router_choice else False
            )

            result.records.append(qr)
    finally:
        reset_daemon(repo_dir, semantex_binary, embedder)

    result = aggregate(result, k=k)
    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(result: RouteEvalResult, *, k: int = 10) -> RouteEvalResult:
    """Populate all aggregate fields on a RouteEvalResult."""
    records = result.records
    if not records:
        return result

    result.total_queries = len(records)
    result.synthesis_count = sum(1 for r in records if r.router_chose_synthesis)

    # Per-route oracle-win counts (a route earns a win when it's among the tied best).
    result.oracle_win_counts = {r: 0 for r in RETRIEVAL_ROUTES}
    for rec in records:
        for r in rec.oracle_routes_tied:
            if r in result.oracle_win_counts:
                result.oracle_win_counts[r] += 1

    # Per-route mean nDCG across all queries.
    result.per_route_mean_ndcg = {
        r: sum(rec.route_ndcg.get(r, 0.0) for rec in records) / len(records)
        for r in RETRIEVAL_ROUTES
    }

    # Per-mechanism × per-route mean nDCG.
    mechanisms = sorted({rec.intended_mechanism for rec in records})
    result.per_mechanism_route_ndcg = {}
    for mech in mechanisms:
        mech_recs = [rec for rec in records if rec.intended_mechanism == mech]
        result.per_mechanism_route_ndcg[mech] = {
            r: sum(rec.route_ndcg.get(r, 0.0) for rec in mech_recs) / len(mech_recs)
            for r in RETRIEVAL_ROUTES
        }

    # Overall regret and per-mechanism regret.
    result.overall_regret = sum(rec.regret for rec in records) / len(records)
    result.per_mechanism_regret = {
        mech: sum(rec.regret for rec in records if rec.intended_mechanism == mech) /
              max(1, sum(1 for r in records if r.intended_mechanism == mech))
        for mech in mechanisms
    }

    # Router accuracy: fraction of queries where router_choice == oracle_route.
    router_correct = sum(
        1 for rec in records
        if (not rec.router_chose_synthesis)
        and rec.router_choice in rec.oracle_routes_tied
    )
    result.router_matches_oracle_count = router_correct
    result.overall_router_accuracy = router_correct / len(records)

    # Per-mechanism router accuracy.
    result.per_mechanism_router_accuracy = {}
    for mech in mechanisms:
        mech_recs = [rec for rec in records if rec.intended_mechanism == mech]
        correct = sum(
            1 for rec in mech_recs
            if (not rec.router_chose_synthesis)
            and rec.router_choice in rec.oracle_routes_tied
        )
        result.per_mechanism_router_accuracy[mech] = correct / max(1, len(mech_recs))

    # Confusion matrix: intended_mechanism × router_choice (sparse — only the
    # routes the router actually picked appear as columns; the renderer orders
    # them retrieval-first then synthesis).
    result.confusion_matrix = {mech: {} for mech in mechanisms}
    for rec in records:
        mech = rec.intended_mechanism
        chosen = rec.router_choice or "unknown"
        result.confusion_matrix[mech][chosen] = result.confusion_matrix[mech].get(chosen, 0) + 1

    return result


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

def _qr_to_dict(qr: QueryRouteResult) -> dict:
    return {
        "query_id": qr.query_id,
        "query": qr.query,
        "repo": qr.repo,
        "intended_mechanism": qr.intended_mechanism,
        "gold_count": len(qr.gold_files),
        "route_ndcg": qr.route_ndcg,
        "route_recall": qr.route_recall,
        "oracle_route": qr.oracle_route,
        "oracle_routes_tied": qr.oracle_routes_tied,
        "oracle_ndcg": round(qr.oracle_ndcg, 4),
        "router_choice": qr.router_choice,
        "router_chose_synthesis": qr.router_chose_synthesis,
        "router_ndcg": round(qr.router_ndcg, 4),
        "regret": round(qr.regret, 4),
    }


def result_to_dict(result: RouteEvalResult) -> dict:
    return {
        "repo": result.repo,
        "total_queries": result.total_queries,
        "synthesis_count": result.synthesis_count,
        "overall_regret": round(result.overall_regret, 4),
        "overall_router_accuracy": round(result.overall_router_accuracy, 4),
        "router_matches_oracle_count": result.router_matches_oracle_count,
        "oracle_win_counts": result.oracle_win_counts,
        "per_route_mean_ndcg": {k: round(v, 4) for k, v in result.per_route_mean_ndcg.items()},
        "per_mechanism_route_ndcg": {
            mech: {r: round(v, 4) for r, v in rmap.items()}
            for mech, rmap in result.per_mechanism_route_ndcg.items()
        },
        "per_mechanism_regret": {k: round(v, 4) for k, v in result.per_mechanism_regret.items()},
        "per_mechanism_router_accuracy": {
            k: round(v, 4) for k, v in result.per_mechanism_router_accuracy.items()
        },
        "confusion_matrix": result.confusion_matrix,
        "per_query": [_qr_to_dict(qr) for qr in result.records],
    }
