"""Drive all four file-localisation arms for one SWE-bench-Verified instance.

Arms (see benchmarks/relevance/README.md "SWE-bench localisation" section for
the full write-up and how each maps to a SweRank/LocAgent-comparable number):

  hybrid        -- `semantex search` (dense+sparse fusion, semantex's shipped
                   default retrieval path).
  sparse-only   -- `semantex search --sparse-only` (BM25-only baseline).
  agent-routed  -- `semantex agent --json-hits`, NO forced route: the engine's
                   own keyword classifier (agent_classifier.rs) picks the
                   retrieval mechanism, exactly as an unforced real `agent`
                   call would. Requires an already-running daemon (unlike
                   `search`, `agent` has no auto-spawn fast path).
  ripgrep       -- an external keyword baseline (ripgrep_baseline.py), no
                   semantex involved at all: "what would grep alone find".

All four arms are OFFLINE once the repo is indexed: no network, no LLM calls
(the classifier is a pure keyword heuristic; the default semantex build wires
zero LLM deps — see WAVE0-CONTRACT.md and semantex-core/Cargo.toml `default =
[]`), and deterministic for a fixed repo snapshot + query.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .indexing import ensure_index
from .ripgrep_baseline import extract_keywords, rank_files_by_keyword_hits
from .route_eval import reset_daemon, start_daemon
from .semantex_client import SemantexClient
from .tokens import estimate_tokens_returned
from .types import Query

#: Arms run per instance, in report-column order.
ARMS: tuple[str, ...] = ("hybrid", "sparse-only", "agent-routed", "ripgrep")


@dataclass(frozen=True)
class ArmQueryResult:
    """One arm's outcome for one query. `error` is set (ranked_files empty)
    when the arm failed to run (e.g. ripgrep not installed, daemon start
    timeout) — the caller records it as a miss rather than aborting the run.
    """
    arm: str
    query_id: str
    ranked_files: tuple[str, ...]
    tokens_returned: int = 0
    error: Optional[str] = None


def run_instance(
    query: Query,
    *,
    corpus_dir: Path,
    semantex_binary: str,
    k: int,
) -> list[ArmQueryResult]:
    """Index `corpus_dir` if needed, then run all `ARMS` for `query`.

    One query per instance (SWE-bench-Verified localisation is one query per
    repo snapshot), so the daemon lifecycle for the agent-routed arm is
    scoped to this single call: reset any stale daemon, run search-path arms
    (which auto-spawn their own daemon), then explicitly start a daemon for
    the agent-routed arm (which requires one already running), and stop it
    on the way out.
    """
    corpus_dir = Path(corpus_dir)
    ensure_index(corpus_dir=corpus_dir, semantex_binary=semantex_binary)
    client = SemantexClient(semantex_binary=semantex_binary, corpus_dir=str(corpus_dir))
    # Defeat stale-daemon reuse from a prior instance's run (config, incl. the
    # adaptive-sizing A/B lock, is cached at spawn time — see semantex_client
    # module docstring).
    client.reset_daemon()

    results: list[ArmQueryResult] = []

    for ablation in ("hybrid", "sparse-only"):
        try:
            rr = client.search(
                query.query_id, query.text, ablation=ablation, k=k, with_content=True
            )
            results.append(ArmQueryResult(
                arm=ablation,
                query_id=query.query_id,
                ranked_files=rr.ranked_files,
                tokens_returned=estimate_tokens_returned(rr),
            ))
        except Exception as e:  # noqa: BLE001 -- recorded as a per-arm miss, not fatal
            results.append(ArmQueryResult(
                arm=ablation, query_id=query.query_id, ranked_files=(), error=str(e)
            ))

    # agent-routed: `agent` requires an already-running daemon (no auto-spawn),
    # so reset + explicitly start one under the canonical A/B env lock.
    reset_daemon(str(corpus_dir), semantex_binary)
    daemon_proc = None
    try:
        daemon_proc = start_daemon(str(corpus_dir), semantex_binary)
        rr = client.search_agent_auto(query.query_id, query.text)
        results.append(ArmQueryResult(
            arm="agent-routed",
            query_id=query.query_id,
            ranked_files=rr.ranked_files[:k],
            tokens_returned=estimate_tokens_returned(rr),
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
        ranked = rank_files_by_keyword_hits(corpus_dir, keywords)[:k]
        results.append(ArmQueryResult(
            arm="ripgrep", query_id=query.query_id, ranked_files=tuple(ranked)
        ))
    except (FileNotFoundError, RuntimeError) as e:
        results.append(ArmQueryResult(
            arm="ripgrep", query_id=query.query_id, ranked_files=(), error=str(e)
        ))

    return results


def relevance_vector(ranked_files: tuple[str, ...], gold: frozenset[str]) -> list[int]:
    """0/1 vector: 1 iff the ranked file at that position is a gold file."""
    return [1 if f in gold else 0 for f in ranked_files]
