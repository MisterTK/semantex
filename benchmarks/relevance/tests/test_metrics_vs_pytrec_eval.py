import pytest

pytrec_eval = pytest.importorskip("pytrec_eval")

from relevance_harness.metrics import mrr_at_k, ndcg_at_k, recall_at_k, mean_average_precision


def _rels_and_nrel(qrel: dict, run: dict, query_id: str):
    """Convert a qrel/run pair into (ranked 0/1 list, n_relevant) for one query."""
    relevant = {d for d, g in qrel[query_id].items() if g > 0}
    ranked = sorted(run[query_id].items(), key=lambda kv: kv[1], reverse=True)
    rels = [1 if doc in relevant else 0 for doc, _ in ranked]
    return rels, len(relevant)


def test_matches_pytrec_eval_on_a_few_queries():
    qrel = {
        "q1": {"d1": 1, "d2": 0, "d3": 1, "d4": 0},
        "q2": {"d1": 0, "d2": 1, "d5": 1},
    }
    run = {
        "q1": {"d1": 0.9, "d2": 0.5, "d3": 0.8, "d4": 0.1},
        "q2": {"d2": 0.7, "d1": 0.6, "d5": 0.2},
    }
    measures = {"recip_rank", "ndcg_cut.10", "recall.10", "map"}
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, measures)
    ref = evaluator.evaluate(run)

    ref_mrr = sum(ref[q]["recip_rank"] for q in qrel) / len(qrel)
    ref_ndcg = sum(ref[q]["ndcg_cut_10"] for q in qrel) / len(qrel)
    ref_recall = sum(ref[q]["recall_10"] for q in qrel) / len(qrel)
    ref_map = sum(ref[q]["map"] for q in qrel) / len(qrel)

    relevances, n_relevant = [], []
    for q in sorted(qrel):
        rels, nrel = _rels_and_nrel(qrel, run, q)
        relevances.append(rels)
        n_relevant.append(nrel)

    assert mrr_at_k(relevances, k=10) == pytest.approx(ref_mrr, abs=1e-9)
    assert ndcg_at_k(relevances, k=10, n_relevant=n_relevant) == pytest.approx(ref_ndcg, abs=1e-9)
    assert recall_at_k(relevances, k=10, n_relevant=n_relevant) == pytest.approx(ref_recall, abs=1e-9)
    assert mean_average_precision(relevances, n_relevant=n_relevant) == pytest.approx(ref_map, abs=1e-9)
