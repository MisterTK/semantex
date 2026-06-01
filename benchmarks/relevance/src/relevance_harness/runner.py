"""Drive the semantex client over an EvalCorpus and emit the arrays the metrics
consume. No metric math here — just per-query ranked relevance vectors.

match_mode:
  "doc_id" — a result is relevant iff its "file:start-end" id is in gold_doc_ids
             (CSN / CoIR exact-target matching).
  "file"   — a result is relevant iff its file path is in gold_doc_ids
             (SWE-loc file-level / function-level localisation).
"""
from __future__ import annotations

from dataclasses import dataclass
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
    ensure_index(corpus_dir=corpus.corpus_dir, semantex_binary=semantex_binary)

    client = SemantexClient(
        semantex_binary=semantex_binary,
        corpus_dir=str(corpus.corpus_dir),
        embedder=embedder,  # canonical SEMANTEX_EMBEDDER selector (integration §4)
    )

    relevances: list[list[int]] = []
    n_relevant: list[int] = []
    per_query: list[RankedResult] = []
    for q in corpus.queries:
        rr = client.search(q.query_id, q.text, ablation=ablation, k=k)
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
    )
