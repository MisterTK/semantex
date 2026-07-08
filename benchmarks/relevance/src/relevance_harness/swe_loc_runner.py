"""Drive all file-localisation arms for one SWE-bench-Verified instance.

Arms (see benchmarks/relevance/README.md "SWE-bench localisation" section for
the full write-up and how each maps to a SweRank/LocAgent-comparable number):

  hybrid        -- `semantex search` (dense+sparse fusion, semantex's shipped
                   default retrieval path). Rerank OFF (the shipped default).
  sparse-only   -- `semantex search --sparse-only` (BM25-only baseline).
  rerank        -- `semantex search --rerank` (hybrid fusion + the shipped
                   cross-encoder reranker). Same retrieval pool as `hybrid`;
                   isolates the reranker's effect on file-level ranking. This
                   is the ONLY arm that is not fully offline: the reranker
                   model downloads on first use (see semantex_client module
                   docstring for the two-gate enable story).
  agent-routed  -- `semantex agent --json-hits`, NO forced route: the engine's
                   own keyword classifier (agent_classifier.rs) picks the
                   retrieval mechanism, exactly as an unforced real `agent`
                   call would. Requires an already-running daemon (unlike
                   `search`, `agent` has no auto-spawn fast path).
  ripgrep       -- an external keyword baseline (ripgrep_baseline.py), no
                   semantex involved at all: "what would grep alone find".

`hybrid`, `sparse-only`, `agent-routed`, and `ripgrep` are OFFLINE once the
repo is indexed: no network, no LLM calls (the classifier is a pure keyword
heuristic; the default semantex build wires zero LLM deps — see
semantex-core/Cargo.toml `default = []`), and deterministic for a fixed repo
snapshot + query. `rerank` needs network on its first invocation ever (model
weight download, then cached).

Latency: `hybrid`, `sparse-only`, and `rerank` each record a "cold" duration
(the timed `search` call itself) and a "warm" duration (an immediate repeat of
the exact same query against the same, now-warm, daemon/model — no reindex,
no respawn). Cold/warm is a real distinction for `rerank` specifically: its
first query in an instance pays cross-encoder model load time that the second
does not.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time

from .indexing import ensure_index
from .ripgrep_baseline import extract_keywords, rank_files_by_keyword_hits
from .route_eval import reset_daemon, start_daemon
from .semantex_client import SemantexClient
from .tokens import estimate_tokens_returned
from .types import Query

#: Arms run per instance, in report-column order. `rerank` sits next to
#: `hybrid` (same retrieval pool, reranker on) so the two are easy to diff.
ARMS: tuple[str, ...] = ("hybrid", "sparse-only", "rerank", "agent-routed", "ripgrep")

#: Arms driven through the `search --json` path (as opposed to `agent` /
#: ripgrep), where cold/warm daemon timing is meaningful.
_SEARCH_PATH_ARMS: tuple[str, ...] = ("hybrid", "sparse-only", "rerank")


@dataclass(frozen=True)
class ArmQueryResult:
    """One arm's outcome for one query. `error` is set (ranked_files empty)
    when the arm failed to run (e.g. ripgrep not installed, daemon start
    timeout) — the caller records it as a miss rather than aborting the run.

    `duration_secs` is the wall-clock time of the (cold) scored search call.
    `warm_duration_secs` is set only for `_SEARCH_PATH_ARMS`: an immediate
    repeat of the same query against the same daemon, isolating steady-state
    latency from one-time costs (daemon spawn, model load).
    """
    arm: str
    query_id: str
    ranked_files: tuple[str, ...]
    tokens_returned: int = 0
    error: Optional[str] = None
    duration_secs: float = 0.0
    warm_duration_secs: Optional[float] = None


def run_instance(
    query: Query,
    *,
    corpus_dir: Path,
    semantex_binary: str,
    k: int,
) -> list[ArmQueryResult]:
    """Index `corpus_dir` if needed, then run all `ARMS` for `query`.

    One query per instance (SWE-bench-Verified localisation is one query per
    repo snapshot), so the daemon lifecycle for the rerank and agent-routed
    arms is scoped to this single call: reset any stale daemon, run the
    plain search-path arms (which auto-spawn their own daemon), force a fresh
    daemon spawn before `rerank` (SEMANTEX_RERANKER is read once at daemon
    spawn time — an already-running daemon from the `hybrid`/`sparse-only`
    arms above would silently keep reranking a no-op), then explicitly start
    a daemon for the agent-routed arm (which requires one already running),
    and stop it on the way out.
    """
    corpus_dir = Path(corpus_dir)
    ensure_index(corpus_dir=corpus_dir, semantex_binary=semantex_binary)
    # 300s (vs. the client's 120s default): on a CPU-constrained/contended host, a
    # cold `rerank` call (fresh daemon spawn + cross-encoder ONNX session build from
    # a ~2.2 GB model file, even when the weights are already cache-warm) measurably
    # exceeds 120s and was observed to time out here — a false "error" for that arm,
    # not a real engine failure. Scoped to this runner only (not the client default)
    # since other callers' timeout needs aren't this benchmark's concern.
    client = SemantexClient(semantex_binary=semantex_binary, corpus_dir=str(corpus_dir),
                             timeout_secs=300)
    # Defeat stale-daemon reuse from a prior instance's run (config, incl. the
    # adaptive-sizing A/B lock, is cached at spawn time — see semantex_client
    # module docstring).
    client.reset_daemon()

    results: list[ArmQueryResult] = []

    for ablation in _SEARCH_PATH_ARMS:
        if ablation == "rerank":
            # Force a fresh daemon spawn so SEMANTEX_RERANKER=on (set by
            # SemantexClient for this ablation) is actually inherited — the
            # daemon the `hybrid`/`sparse-only` calls above may have spawned
            # fixed its env before this flag existed.
            client.reset_daemon()
        try:
            t0 = time.monotonic()
            rr = client.search(
                query.query_id, query.text, ablation=ablation, k=k, with_content=True
            )
            duration = time.monotonic() - t0

            # Warm rerun: identical query, same (now warm) daemon/model. Cheap
            # (no reindex, no respawn) and best-effort — a failure here never
            # invalidates the (already-captured) cold result.
            warm_duration: Optional[float] = None
            try:
                t1 = time.monotonic()
                client.search(
                    query.query_id, query.text, ablation=ablation, k=k, with_content=True
                )
                warm_duration = time.monotonic() - t1
            except Exception:  # noqa: BLE001 -- warm timing is best-effort
                warm_duration = None

            results.append(ArmQueryResult(
                arm=ablation,
                query_id=query.query_id,
                ranked_files=rr.ranked_files,
                tokens_returned=estimate_tokens_returned(rr),
                duration_secs=duration,
                warm_duration_secs=warm_duration,
            ))
        except Exception as e:  # noqa: BLE001 -- recorded as a per-arm miss, not fatal
            results.append(ArmQueryResult(
                arm=ablation, query_id=query.query_id, ranked_files=(), error=str(e)
            ))

    # agent-routed: `agent` requires an already-running daemon (no auto-spawn),
    # so reset + explicitly start one under the canonical A/B env lock. This
    # also drops any reranker-enabled daemon left over from the `rerank` arm.
    reset_daemon(str(corpus_dir), semantex_binary)
    daemon_proc = None
    try:
        daemon_proc = start_daemon(str(corpus_dir), semantex_binary)
        t0 = time.monotonic()
        rr = client.search_agent_auto(query.query_id, query.text)
        duration = time.monotonic() - t0
        results.append(ArmQueryResult(
            arm="agent-routed",
            query_id=query.query_id,
            ranked_files=rr.ranked_files[:k],
            tokens_returned=estimate_tokens_returned(rr),
            duration_secs=duration,
        ))
    except Exception as e:  # noqa: BLE001
        results.append(ArmQueryResult(
            arm="agent-routed", query_id=query.query_id, ranked_files=(), error=str(e)
        ))
    finally:
        reset_daemon(str(corpus_dir), semantex_binary)
        if daemon_proc is not None and daemon_proc.poll() is None:
            daemon_proc.terminate()

    # ripgrep keyword baseline: no index, no daemon, pure lexical presence.
    try:
        keywords = extract_keywords(query.text)
        t0 = time.monotonic()
        ranked = rank_files_by_keyword_hits(corpus_dir, keywords)[:k]
        duration = time.monotonic() - t0
        results.append(ArmQueryResult(
            arm="ripgrep", query_id=query.query_id, ranked_files=tuple(ranked),
            duration_secs=duration,
        ))
    except (FileNotFoundError, RuntimeError) as e:
        results.append(ArmQueryResult(
            arm="ripgrep", query_id=query.query_id, ranked_files=(), error=str(e)
        ))

    return results


def relevance_vector(ranked_files: tuple[str, ...], gold: frozenset[str]) -> list[int]:
    """0/1 vector: 1 iff the ranked file at that position is a gold file."""
    return [1 if f in gold else 0 for f in ranked_files]
