import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import lateon_vs_coderank_report
import lateon_vs_coderank_report as lvc


def _write_track_a_run(tmp_path, dataset, embedder, rows, instr=None):
    run_dir = tmp_path / dataset / embedder
    run_dir.mkdir(parents=True)
    (run_dir / "report.json").write_text(
        json.dumps({"stamp": {}, "rows": rows, "manifests": []})
    )
    if instr is not None:
        (run_dir / "instr-x.json").write_text(json.dumps(instr))
    return run_dir


def test_load_track_a_cell_pairs_rows_with_instrumentation_positionally(tmp_path):
    rows = [
        {"dataset": "csn/python", "ablation": "hybrid", "n_queries": 10,
         "mrr_at_10": 0.5, "ndcg_at_10": 0.6, "recall_at_1": 0.3,
         "recall_at_5": 0.7, "recall_at_10": 0.8, "map": 0.55},
        {"dataset": "csn/go", "ablation": "hybrid", "n_queries": 10,
         "mrr_at_10": 0.4, "ndcg_at_10": 0.5, "recall_at_1": 0.2,
         "recall_at_5": 0.6, "recall_at_10": 0.7, "map": 0.45},
    ]
    instr = [
        {"cell": "csn/python", "cold_latency_ms": 120.0, "warm_latency_ms": 40.0},
        {"cell": "csn/go", "cold_latency_ms": 130.0, "warm_latency_ms": 45.0},
    ]
    run_dir = _write_track_a_run(tmp_path, "csn", "lateon-colbert", rows, instr)

    cells = lvc.load_track_a_cell(run_dir)

    assert len(cells) == 2
    assert cells[0]["dataset"] == "csn/python"
    assert cells[0]["cold_latency_ms"] == 120.0
    assert cells[1]["warm_latency_ms"] == 45.0


def test_load_track_a_cell_handles_missing_instrumentation(tmp_path):
    rows = [{"dataset": "coir/x", "ablation": "hybrid", "n_queries": 5,
             "mrr_at_10": 0.5, "ndcg_at_10": 0.5, "recall_at_1": 0.5,
             "recall_at_5": 0.5, "recall_at_10": 0.5, "map": 0.5}]
    run_dir = _write_track_a_run(tmp_path, "coir", "lateon-colbert", rows)

    cells = lvc.load_track_a_cell(run_dir)

    assert cells[0]["cold_latency_ms"] is None
    assert cells[0]["warm_latency_ms"] is None


def test_track_a_table_skips_dataset_embedder_pairs_not_yet_run(tmp_path):
    rows = [{"dataset": "coir/x", "ablation": "hybrid", "n_queries": 5,
             "mrr_at_10": 0.5, "ndcg_at_10": 0.5, "recall_at_1": 0.5,
             "recall_at_5": 0.5, "recall_at_10": 0.5, "map": 0.5}]
    _write_track_a_run(tmp_path, "coir", "lateon-colbert", rows,
                        [{"cell": "coir/x", "cold_latency_ms": 100.0, "warm_latency_ms": 30.0}])
    # coderank-137m for "coir", and both embedders for "csn", were never run.

    table = lvc.track_a_table(tmp_path)

    assert len(table) == 1
    assert table[0]["embedder"] == "lateon-colbert"
    assert table[0]["dataset"] == "coir/x"


def test_track_b_table_computes_mean_and_stdev_per_repo_question_type_arm():
    all_results = [
        {"arm": "sx-lateon", "repo": "/x/gin", "question_type": "architecture", "quality": 3},
        {"arm": "sx-lateon", "repo": "/x/gin", "question_type": "architecture", "quality": 5},
        {"arm": "sx-coderank", "repo": "/x/gin", "question_type": "architecture", "quality": 4},
        {"arm": "builtin", "repo": "/x/gin", "question_type": "architecture", "quality": 1},
        {"arm": "sx-lateon", "repo": "/x/gin", "question_type": "architecture", "quality": None},
    ]

    table = lvc.track_b_table(all_results)

    assert {"repo": "gin", "question_type": "architecture", "arm": "sx-lateon",
            "n": 2, "mean_quality": 4.0, "stdev": 1.41} in table
    assert {"repo": "gin", "question_type": "architecture", "arm": "sx-coderank",
            "n": 1, "mean_quality": 4.0, "stdev": 0.0} in table
    assert len(table) == 2  # builtin excluded, None-quality row dropped


def test_flag_ambiguous_cells_flags_overlapping_bands_and_skips_clear_wins():
    track_b_rows = [
        {"repo": "gin", "question_type": "architecture", "arm": "sx-lateon",
         "n": 3, "mean_quality": 4.0, "stdev": 0.5},
        {"repo": "gin", "question_type": "architecture", "arm": "sx-coderank",
         "n": 3, "mean_quality": 4.2, "stdev": 0.3},
        {"repo": "gin", "question_type": "deep_technical", "arm": "sx-lateon",
         "n": 3, "mean_quality": 2.0, "stdev": 0.2},
        {"repo": "gin", "question_type": "deep_technical", "arm": "sx-coderank",
         "n": 3, "mean_quality": 4.5, "stdev": 0.2},
    ]

    flags = lvc.flag_ambiguous_cells(track_b_rows)

    assert len(flags) == 1
    assert "gin/architecture" in flags[0]


def test_render_report_md_includes_all_sections():
    md = lvc.render_report_md(
        track_a_rows=[{"embedder": "lateon-colbert", "dataset": "csn/python",
                       "n_queries": 10, "mrr_at_10": 0.5, "ndcg_at_10": 0.6,
                       "recall_at_1": 0.3, "recall_at_5": 0.7, "recall_at_10": 0.8,
                       "map": 0.55, "cold_latency_ms": 120.0, "warm_latency_ms": 40.0}],
        track_b_rows=[{"repo": "gin", "question_type": "architecture", "arm": "sx-lateon",
                       "n": 3, "mean_quality": 4.0, "stdev": 0.5}],
        ambiguous=["gin/architecture: ambiguous (placeholder text for this test)"],
    )

    assert "Track A" in md
    assert "csn/python" in md
    assert "Track B" in md
    assert "gin" in md
    assert "Ambiguous cells" in md
    assert "gin/architecture: ambiguous" in md
