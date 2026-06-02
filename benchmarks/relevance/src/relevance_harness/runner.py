"""Drive the semantex client over an EvalCorpus and emit the arrays the metrics
consume. No metric math here — just per-query ranked relevance vectors.

match_mode:
  "doc_id" — a result is relevant iff its "file:start-end" id is in gold_doc_ids
             (CSN / CoIR exact-target matching).
  "file"   — a result is relevant iff its file path is in gold_doc_ids
             (SWE-loc file-level / function-level localisation).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .indexing import ensure_index
from .semantex_client import SemantexClient
from .types import EvalCorpus, RankedResult


@dataclass
class RunOutput:
    """Everything metrics.py needs, plus raw per-query results for the report."""
    corpus_name: str
    ablation: str
    relevances: list[list[int]]
    n_relevant: list[int]
    per_query: list[RankedResult]
    # D4 instrumentation (optional; default-empty keeps unit tests stable).
    embedder: Optional[str] = None
    index_built: bool = False
    index_secs: float = 0.0
    index_peak_rss_mb: Optional[float] = None
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def cold_latency_ms(self) -> Optional[float]:
        return self.latencies_ms[0] if self.latencies_ms else None

    @property
    def warm_latency_ms(self) -> Optional[float]:
        """Median of all-but-first query latencies (warm path)."""
        warm = self.latencies_ms[1:]
        if not warm:
            return None
        s = sorted(warm)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _relevance_vector(rr: RankedResult, gold: set[str], *, match_mode: str) -> list[int]:
    if match_mode == "file":
        return [1 if f in gold else 0 for f in rr.ranked_files]
    return [1 if d in gold else 0 for d in rr.ranked_doc_ids]


def run_corpus(
    corpus: EvalCorpus,
    *,
    ablation: str,
    k: int,
    semantex_binary: str,
    embedder: Optional[str] = None,
    match_mode: str = "doc_id",
) -> RunOutput:
    if corpus.corpus_dir is None:
        raise ValueError("corpus.corpus_dir must be set to index + search")
    # The embedder is authoritative at INDEX time (it picks + persists the dense
    # backend in meta.json). Pass it through so arm B actually builds coderank-hnsw.
    build = ensure_index(
        corpus_dir=corpus.corpus_dir, semantex_binary=semantex_binary, embedder=embedder
    )

    client = SemantexClient(
        semantex_binary=semantex_binary,
        corpus_dir=str(corpus.corpus_dir),
        embedder=embedder,  # canonical SEMANTEX_EMBEDDER selector (integration §4)
    )

    # Defeat stale-daemon reuse: a daemon caches its config (incl. the canonical
    # adaptive-OFF A/B lock) at spawn time and lives 30 min idle, so stop any
    # existing daemon before searching to force a fresh spawn under the lock. The
    # rerank ablation is exempt — it relies on a manually pre-started daemon
    # (rerank env + a raised RSS cap) that a reset would kill, silently disabling
    # reranking.
    if ablation != "rerank":
        client.reset_daemon()

    relevances: list[list[int]] = []
    n_relevant: list[int] = []
    per_query: list[RankedResult] = []
    latencies_ms: list[float] = []
    for q in corpus.queries:
        t0 = time.monotonic()
        rr = client.search(q.query_id, q.text, ablation=ablation, k=k)
        latencies_ms.append((time.monotonic() - t0) * 1000.0)
        gold = set(q.gold_doc_ids)
        relevances.append(_relevance_vector(rr, gold, match_mode=match_mode))
        n_relevant.append(len(gold))
        per_query.append(rr)

    return RunOutput(
        corpus_name=corpus.name,
        ablation=ablation,
        relevances=relevances,
        n_relevant=n_relevant,
        per_query=per_query,
        embedder=embedder,
        index_built=build.built,
        index_secs=build.duration_secs,
        index_peak_rss_mb=build.peak_rss_mb,
        latencies_ms=latencies_ms,
    )
