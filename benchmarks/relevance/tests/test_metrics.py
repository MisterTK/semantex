import math

import pytest

from relevance_harness.metrics import (
    acc_at_k, average_precision, mean_average_precision, mrr_at_k, ndcg_at_k, recall_at_k,
)


# relevances: a list per query of 0/1 in ranked order (1 = doc at that rank is relevant)
def test_mrr_first_relevant_at_rank_2():
    # first relevant is at rank 2 -> 1/2
    assert mrr_at_k([[0, 1, 0, 1]], k=10) == pytest.approx(0.5)


def test_mrr_no_relevant_in_top_k_is_zero():
    assert mrr_at_k([[0, 0, 0, 1]], k=3) == pytest.approx(0.0)


def test_mrr_averages_across_queries():
    # q1: first rel at rank 1 -> 1.0 ; q2: first rel at rank 4 -> 0.25
    assert mrr_at_k([[1, 0], [0, 0, 0, 1]], k=10) == pytest.approx((1.0 + 0.25) / 2)


def test_acc_at_k_hit_when_any_relevant_in_top_k():
    # one relevant doc at rank 2 -> hit at k=5, miss at k=1
    assert acc_at_k([[0, 1, 0]], k=5) == pytest.approx(1.0)
    assert acc_at_k([[0, 1, 0]], k=1) == pytest.approx(0.0)


def test_acc_at_k_no_partial_credit_for_multi_gold():
    # 2 gold docs, only 1 retrieved in top-k -> still a full hit (unlike recall)
    assert acc_at_k([[1, 0, 0]], k=3) == pytest.approx(1.0)


def test_acc_at_k_averages_across_queries():
    # q1 hits, q2 misses -> 0.5
    assert acc_at_k([[1, 0], [0, 0]], k=2) == pytest.approx(0.5)


def test_acc_at_k_empty_relevances_is_zero():
    assert acc_at_k([], k=10) == pytest.approx(0.0)


def test_recall_at_k_counts_relevant_in_top_k():
    # 3 relevant total (n_relevant given separately); 2 of them in top-3
    assert recall_at_k([[1, 0, 1, 1]], k=3, n_relevant=[3]) == pytest.approx(2 / 3)


def test_recall_at_k_zero_relevant_is_zero():
    assert recall_at_k([[0, 0]], k=2, n_relevant=[0]) == pytest.approx(0.0)


def test_ndcg_perfect_ranking_is_one():
    # 2 relevant, both at the top -> DCG == IDCG -> 1.0
    assert ndcg_at_k([[1, 1, 0, 0]], k=4, n_relevant=[2]) == pytest.approx(1.0)


def test_ndcg_known_value():
    # rels at ranks 2 and 3; n_relevant=2
    # DCG = 1/log2(3) + 1/log2(4) = 0.6309298 + 0.5 = 1.1309298
    # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309298 = 1.6309298
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k([[0, 1, 1, 0]], k=4, n_relevant=[2]) == pytest.approx(dcg / idcg)


def test_average_precision_known_value():
    # rels at ranks 1 and 3, n_relevant=2
    # precision@1 = 1/1 = 1.0 ; precision@3 = 2/3
    # AP = (1.0 + 0.6667) / 2 = 0.8333
    assert average_precision([1, 0, 1, 0], n_relevant=2) == pytest.approx((1.0 + 2 / 3) / 2)


def test_map_averages_average_precision():
    ap1 = average_precision([1, 0], n_relevant=1)       # 1.0
    ap2 = average_precision([0, 0, 1], n_relevant=1)    # 1/3
    assert mean_average_precision([[1, 0], [0, 0, 1]], n_relevant=[1, 1]) == pytest.approx(
        (ap1 + ap2) / 2
    )
