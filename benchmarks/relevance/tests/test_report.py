import json

from relevance_harness.report import (
    ReproStamp, compute_metrics_row, render_report_json, render_report_md,
)
from relevance_harness.runner import RunOutput


def _run(name="csn/python", ablation="hybrid"):
    # q1 gold at rank 1, q2 gold at rank 2
    return RunOutput(
        corpus_name=name, ablation=ablation,
        relevances=[[1, 0], [0, 1]], n_relevant=[1, 1], per_query=[],
    )


def test_compute_metrics_row_has_all_metrics():
    row = compute_metrics_row(_run(), k=10)
    assert row["dataset"] == "csn/python"
    assert row["ablation"] == "hybrid"
    assert row["n_queries"] == 2
    for key in ("mrr_at_10", "ndcg_at_10", "recall_at_1", "recall_at_5", "recall_at_10", "map"):
        assert key in row
    # mrr = (1/1 + 1/2)/2 = 0.75
    assert abs(row["mrr_at_10"] - 0.75) < 1e-9
    # recall@1 = (1 + 0)/2 = 0.5 ; recall@10 = 1.0
    assert abs(row["recall_at_1"] - 0.5) < 1e-9
    assert abs(row["recall_at_10"] - 1.0) < 1e-9


def test_report_json_includes_stamp_and_manifest(tmp_path):
    rows = [compute_metrics_row(_run(), k=10)]
    stamp = ReproStamp(git_rev="abc123", dense_backend="coderank-hnsw",
                       model_id="CodeRankEmbed", k=10)
    manifests = [{"dataset": "csn/python", "total": 1000, "selected": 200,
                  "seed": 20260531, "kept_ids": ["q1"], "dropped_ids": []}]
    out = render_report_json(rows=rows, stamp=stamp, manifests=manifests)
    data = json.loads(out)
    assert data["stamp"]["git_rev"] == "abc123"
    assert data["stamp"]["dense_backend"] == "coderank-hnsw"
    assert data["rows"][0]["dataset"] == "csn/python"
    assert data["manifests"][0]["selected"] == 200


def test_report_md_contains_table_and_backend():
    rows = [compute_metrics_row(_run(), k=10)]
    stamp = ReproStamp(git_rev="abc123", dense_backend="colbert-plaid",
                       model_id="colbert", k=10)
    md = render_report_md(rows=rows, stamp=stamp, manifests=[])
    assert "csn/python" in md
    assert "colbert-plaid" in md
    assert "mrr_at_10" in md or "MRR@10" in md
