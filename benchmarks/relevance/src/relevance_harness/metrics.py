"""Ranking metrics from canonical IR definitions. Pure functions, NumPy only.

Each function takes `relevances`: a list (one per query) of 0/1 ints in ranked
order, where a 1 at position i means the doc ranked i-th (1-based) is relevant.
`n_relevant` is the per-query count of relevant docs in the full qrels (needed
for Recall, nDCG ideal-DCG, and MAP — it can exceed the number of 1s present in
the truncated/returned list).
"""
from __future__ import annotations

import math


def _reciprocal_rank(rels: list[int], k: int) -> float:
    for i, r in enumerate(rels[:k], start=1):
        if r:
            return 1.0 / i
    return 0.0


def mrr_at_k(relevances: list[list[int]], *, k: int) -> float:
    """Mean reciprocal rank of the first relevant doc within top-k."""
    if not relevances:
        return 0.0
    return sum(_reciprocal_rank(r, k) for r in relevances) / len(relevances)


def acc_at_k(relevances: list[list[int]], *, k: int) -> float:
    """Fraction of queries with >=1 relevant doc in the top-k ("hit rate").

    This is the file-level Acc@k SweRank and LocAgent report for SWE-bench
    localisation: a query counts as correct iff ANY gold file is retrieved
    within the top-k, regardless of how many gold files exist or how many of
    them were found. It differs from recall_at_k, which averages the
    FRACTION of gold docs retrieved (relevant when a query has multiple gold
    docs and partial credit matters); Acc@k gives no partial credit, matching
    the published protocol so numbers are directly comparable.
    """
    if not relevances:
        return 0.0
    hits = sum(1 for rels in relevances if any(rels[:k]))
    return hits / len(relevances)


def recall_at_k(relevances: list[list[int]], *, k: int, n_relevant: list[int]) -> float:
    """Mean over queries of (relevant retrieved in top-k) / (total relevant)."""
    if not relevances:
        return 0.0
    total = 0.0
    for rels, nrel in zip(relevances, n_relevant):
        if nrel <= 0:
            total += 0.0
            continue
        hits = sum(1 for r in rels[:k] if r)
        total += hits / nrel
    return total / len(relevances)


def _dcg(rels: list[int], k: int) -> float:
    return sum(r / math.log2(i + 1) for i, r in enumerate(rels[:k], start=1))


def ndcg_at_k(relevances: list[list[int]], *, k: int, n_relevant: list[int]) -> float:
    """Mean normalised DCG@k with binary relevance.

    IDCG uses the ideal ranking: min(n_relevant, k) ones at the top.
    """
    if not relevances:
        return 0.0
    total = 0.0
    for rels, nrel in zip(relevances, n_relevant):
        dcg = _dcg(rels, k)
        ideal = [1] * min(nrel, k)
        idcg = _dcg(ideal, k)
        total += (dcg / idcg) if idcg > 0 else 0.0
    return total / len(relevances)


def average_precision(rels: list[int], *, n_relevant: int) -> float:
    """Average precision for a single query.

    AP = (1/n_relevant) * Σ_over_relevant_hits precision@(rank of that hit).
    """
    if n_relevant <= 0:
        return 0.0
    hits = 0
    score = 0.0
    for i, r in enumerate(rels, start=1):
        if r:
            hits += 1
            score += hits / i
    return score / n_relevant


def mean_average_precision(
    relevances: list[list[int]], *, n_relevant: list[int]
) -> float:
    """Mean of per-query average precision."""
    if not relevances:
        return 0.0
    return sum(
        average_precision(rels, n_relevant=nrel)
        for rels, nrel in zip(relevances, n_relevant)
    ) / len(relevances)
