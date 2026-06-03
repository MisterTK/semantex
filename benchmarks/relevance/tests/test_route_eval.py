"""Unit tests for the oracle-regret route_eval aggregation logic.

These run WITHOUT a daemon: they exercise the pure aggregation/dedup/parsing math
on tiny synthetic fixtures, so the decision-table logic is verifiable in isolation.
"""

import pytest

from relevance_harness.metrics import ndcg_at_k
from relevance_harness.route_eval import (
    QueryRouteResult,
    RouteEvalResult,
    aggregate,
    dedup_ranked_files,
    parse_files_from_formatted,
)


# ---------------------------------------------------------------------------
# (a) dedup keeps first-occurrence rank + pulls inflated nDCG back to <= 1.0
# ---------------------------------------------------------------------------

def test_dedup_keeps_first_occurrence_rank():
    ranked = ["a.go", "b.go", "a.go", "c.go", "b.go"]
    assert dedup_ranked_files(ranked) == ["a.go", "b.go", "c.go"]


def test_dedup_preserves_order_no_dupes():
    ranked = ["x.go", "y.go", "z.go"]
    assert dedup_ranked_files(ranked) == ranked


def test_dedup_empty():
    assert dedup_ranked_files([]) == []


def test_undeduped_gold_repeat_inflates_ndcg_above_one():
    """Without dedup, a gold file repeated across sections inflates nDCG > 1.0.

    This is the bug the dedup fix prevents — assert the bug exists on raw input,
    then assert dedup pulls it back to <= 1.0.
    """
    gold = frozenset({"gold.go"})
    # gold.go appears at ranks 1, 2, 3 (e.g. 3 structural graph sections).
    raw = ["gold.go", "gold.go", "gold.go", "other.go"]
    raw_rels = [1 if f in gold else 0 for f in raw]
    # n_relevant=1, but 3 ones in the ranked list => DCG > IDCG => nDCG > 1.0
    raw_ndcg = ndcg_at_k([raw_rels], k=10, n_relevant=[1])
    assert raw_ndcg > 1.0, "expected the un-deduped vector to inflate nDCG above 1.0"

    deduped = dedup_ranked_files(raw)
    dd_rels = [1 if f in gold else 0 for f in deduped]
    dd_ndcg = ndcg_at_k([dd_rels], k=10, n_relevant=[1])
    assert dd_ndcg == pytest.approx(1.0)
    assert dd_ndcg <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# parse_files_from_formatted
# ---------------------------------------------------------------------------

def test_parse_semantic_formatted():
    text = (
        "[route: semantic]\n"
        "\n"
        "utils.go:51-53 [text] (0.06)\n"
        "    // some content\n"
        "render/json.go:12-30 [method] (0.04)\n"
        "  func ...\n"
        "\n"
        "[10 results, 44ms, confidence: low]\n"
    )
    files = parse_files_from_formatted(text)
    assert files == ["utils.go", "render/json.go"]


def test_parse_regex_formatted():
    text = (
        "[route: regex]\n"
        "\n"
        "recovery.go:114: func stack(skip int) []byte {\n"
        "context_test.go:1577: // TODO\n"
        "recovery.go:140: // readNthLine\n"  # dup file, second line -> deduped
        "\n"
        "[20 matches, 1ms]\n"
    )
    files = parse_files_from_formatted(text)
    assert files == ["recovery.go", "context_test.go"]


def test_parse_file_pattern_bare_filenames():
    text = (
        "[route: file_pattern]\n"
        "\n"
        "auth_test.go\n"
        "binding/binding_test.go\n"
        "render/json.go\n"
    )
    files = parse_files_from_formatted(text)
    assert files == ["auth_test.go", "binding/binding_test.go", "render/json.go"]


def test_parse_ignores_prose_and_code_lines():
    # content/prose lines contain spaces or are indented -> never matched as files
    text = (
        "[route: structural]\n"
        "\n"
        "context.go:188-196 Next [method]\n"
        "  Calls: safe int8, len\n"
        "    func (c *Context) Next() {\n"
        "this is prose with spaces.md mention\n"  # has spaces -> not a bare file
    )
    files = parse_files_from_formatted(text)
    assert files == ["context.go"]


def test_parse_structural_indented_callers():
    # The structural route nests INDENTED file:range result lines under
    # Target/Callers/Callees headers; the parser must pick those up (the bug
    # that made structural score 0 on structural queries was dropping these).
    text = (
        "[route: structural]\n"
        "\n"
        "Target:\n"
        "  gin.go:690-760 handleHTTPRequest [method]\n"
        "\n"
        "Callers (44):\n"
        "  auth_test.go:84-99 TestBasicAuthSucceed [fn]\n"
        "  gin.go:662-675 ServeHTTP [method]\n"  # dup file gin.go -> deduped
        "  ... and 34 more\n"
        "\n"
        "Callees (9):\n"
        "  context.go:188-196 Next [method]\n"
    )
    files = parse_files_from_formatted(text)
    # gin.go appears first (Target), auth_test.go, then context.go; gin.go deduped.
    assert files == ["gin.go", "auth_test.go", "context.go"]


def test_parse_empty_route():
    assert parse_files_from_formatted("[route: file_pattern]\n\n") == []


# ---------------------------------------------------------------------------
# (b) oracle_route / oracle_ndcg / regret on a synthetic QueryRouteResult
# ---------------------------------------------------------------------------

def _qr(mechanism, route_ndcg, router_choice, *, qid="q1"):
    qr = QueryRouteResult(
        query_id=qid,
        query="dummy",
        repo="testrepo",
        intended_mechanism=mechanism,
        gold_files=frozenset({"g.go"}),
    )
    qr.route_ndcg = dict(route_ndcg)
    qr.route_recall = {r: 0.0 for r in route_ndcg}
    qr.router_choice = router_choice
    from relevance_harness.route_eval import SYNTHESIS_ROUTES
    qr.router_chose_synthesis = router_choice in SYNTHESIS_ROUTES
    return qr


def test_oracle_route_picks_max_ndcg():
    qr = _qr(
        "structural",
        {"file_pattern": 0.0, "regex": 0.1, "exact_symbol": 0.2,
         "structural": 0.9, "semantic": 0.5},
        router_choice="structural",
    )
    assert qr.oracle_route == "structural"
    assert qr.oracle_ndcg == pytest.approx(0.9)
    assert qr.oracle_routes_tied == ["structural"]


def test_oracle_tie_lists_all_tied_routes():
    qr = _qr(
        "semantic",
        {"file_pattern": 0.0, "regex": 0.0, "exact_symbol": 0.0,
         "structural": 0.7, "semantic": 0.7},
        router_choice="semantic",
    )
    # ties resolved in RETRIEVAL_ROUTES order for oracle_route
    assert qr.oracle_route == "structural"  # earlier in RETRIEVAL_ROUTES
    assert set(qr.oracle_routes_tied) == {"structural", "semantic"}


def test_regret_is_oracle_minus_router_choice():
    qr = _qr(
        "structural",
        {"file_pattern": 0.0, "regex": 0.1, "exact_symbol": 0.2,
         "structural": 0.9, "semantic": 0.5},
        router_choice="semantic",  # router picked semantic (0.5), oracle is structural (0.9)
    )
    assert qr.router_ndcg == pytest.approx(0.5)
    assert qr.regret == pytest.approx(0.4)


def test_zero_regret_when_router_picks_oracle():
    qr = _qr(
        "structural",
        {"file_pattern": 0.0, "regex": 0.1, "exact_symbol": 0.2,
         "structural": 0.9, "semantic": 0.5},
        router_choice="structural",
    )
    assert qr.regret == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# (d) router-chose-synthesis -> retrieval nDCG scored 0 + counted separately
# ---------------------------------------------------------------------------

def test_router_chose_synthesis_scores_zero_router_ndcg():
    qr = _qr(
        "semantic",
        {"file_pattern": 0.0, "regex": 0.0, "exact_symbol": 0.0,
         "structural": 0.3, "semantic": 0.8},
        router_choice="deep",  # synthesis route
    )
    assert qr.router_chose_synthesis is True
    assert qr.router_ndcg == 0.0  # synthesis returns no file hits -> 0
    # regret = oracle (semantic 0.8) - router (0) = 0.8
    assert qr.regret == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# (c) confusion matrix + per-route oracle-win aggregation
# ---------------------------------------------------------------------------

def test_aggregate_oracle_win_counts_and_confusion():
    records = [
        # q1: structural wins; router picks structural (correct)
        _qr("structural",
            {"file_pattern": 0.0, "regex": 0.0, "exact_symbol": 0.0,
             "structural": 0.9, "semantic": 0.4},
            router_choice="structural", qid="q1"),
        # q2: semantic wins; router picks semantic (correct)
        _qr("semantic",
            {"file_pattern": 0.0, "regex": 0.0, "exact_symbol": 0.0,
             "structural": 0.2, "semantic": 0.8},
            router_choice="semantic", qid="q2"),
        # q3: semantic wins; router picks structural (WRONG)
        _qr("semantic",
            {"file_pattern": 0.0, "regex": 0.0, "exact_symbol": 0.0,
             "structural": 0.3, "semantic": 0.7},
            router_choice="structural", qid="q3"),
        # q4: file_pattern wins; router picks deep (synthesis)
        _qr("glob",
            {"file_pattern": 0.9, "regex": 0.0, "exact_symbol": 0.0,
             "structural": 0.0, "semantic": 0.1},
            router_choice="deep", qid="q4"),
    ]
    result = RouteEvalResult(repo="testrepo", records=records)
    result = aggregate(result, k=10)

    # oracle-win counts: structural 1 (q1), semantic 2 (q2,q3), file_pattern 1 (q4)
    assert result.oracle_win_counts["structural"] == 1
    assert result.oracle_win_counts["semantic"] == 2
    assert result.oracle_win_counts["file_pattern"] == 1
    assert result.oracle_win_counts["regex"] == 0
    assert result.oracle_win_counts["exact_symbol"] == 0

    # synthesis count = 1 (q4 deep)
    assert result.synthesis_count == 1
    assert result.total_queries == 4

    # router matches oracle: q1 (structural=structural) + q2 (semantic=semantic) = 2
    # q3 wrong, q4 synthesis -> not counted
    assert result.router_matches_oracle_count == 2
    assert result.overall_router_accuracy == pytest.approx(2 / 4)

    # confusion matrix: rows = intended_mechanism, cols = router choice
    assert result.confusion_matrix["structural"]["structural"] == 1
    assert result.confusion_matrix["semantic"]["semantic"] == 1
    assert result.confusion_matrix["semantic"]["structural"] == 1
    assert result.confusion_matrix["glob"]["deep"] == 1


def test_aggregate_oracle_win_counts_ties_credit_all():
    # one query where structural and semantic tie at the oracle nDCG
    records = [
        _qr("semantic",
            {"file_pattern": 0.0, "regex": 0.0, "exact_symbol": 0.0,
             "structural": 0.6, "semantic": 0.6},
            router_choice="semantic", qid="q1"),
    ]
    result = aggregate(RouteEvalResult(repo="t", records=records), k=10)
    # both tied routes earn an oracle win
    assert result.oracle_win_counts["structural"] == 1
    assert result.oracle_win_counts["semantic"] == 1
    # router picked semantic which is in the tie -> counted as matching oracle
    assert result.router_matches_oracle_count == 1


def test_aggregate_regret_overall_and_per_mechanism():
    records = [
        _qr("structural",
            {"file_pattern": 0.0, "regex": 0.0, "exact_symbol": 0.0,
             "structural": 0.9, "semantic": 0.4},
            router_choice="semantic", qid="q1"),  # regret 0.5
        _qr("semantic",
            {"file_pattern": 0.0, "regex": 0.0, "exact_symbol": 0.0,
             "structural": 0.2, "semantic": 0.8},
            router_choice="semantic", qid="q2"),  # regret 0.0
    ]
    result = aggregate(RouteEvalResult(repo="t", records=records), k=10)
    assert result.overall_regret == pytest.approx((0.5 + 0.0) / 2)
    assert result.per_mechanism_regret["structural"] == pytest.approx(0.5)
    assert result.per_mechanism_regret["semantic"] == pytest.approx(0.0)


def test_aggregate_per_route_mean_ndcg():
    records = [
        _qr("structural",
            {"file_pattern": 0.0, "regex": 0.0, "exact_symbol": 0.0,
             "structural": 0.8, "semantic": 0.4},
            router_choice="structural", qid="q1"),
        _qr("semantic",
            {"file_pattern": 0.0, "regex": 0.0, "exact_symbol": 0.0,
             "structural": 0.2, "semantic": 0.6},
            router_choice="semantic", qid="q2"),
    ]
    result = aggregate(RouteEvalResult(repo="t", records=records), k=10)
    assert result.per_route_mean_ndcg["structural"] == pytest.approx((0.8 + 0.2) / 2)
    assert result.per_route_mean_ndcg["semantic"] == pytest.approx((0.4 + 0.6) / 2)
    assert result.per_route_mean_ndcg["regex"] == pytest.approx(0.0)


def test_aggregate_empty_records_is_safe():
    result = aggregate(RouteEvalResult(repo="t", records=[]), k=10)
    assert result.total_queries == 0
