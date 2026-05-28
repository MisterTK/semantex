import json
from pathlib import Path

import pandas as pd

from swe_bench_harness.analyze import build_unit_table


def test_build_unit_table_joins_runs_with_eval(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for inst in ("r0", "r1"):
        for cond in ("c1", "c2"):
            (runs_dir / f"{inst}__{cond}__0.json").write_text(json.dumps({
                "instance_id": inst, "condition_id": cond, "replicate": 0,
                "patch": "diff", "wall_clock_secs": 100.0,
                "turns": [
                    {"input_tokens": 100, "output_tokens": 10,
                     "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                     "tool_calls": ["grep"]},
                ],
                "error": "",
            }))
    eval_reports = {
        "c1__0": {"resolved_ids": ["r0"], "unresolved_ids": ["r1"], "submitted_ids": ["r0","r1"], "error_ids": []},
        "c2__0": {"resolved_ids": ["r0","r1"], "unresolved_ids": [], "submitted_ids": ["r0","r1"], "error_ids": []},
    }
    pricing = {"claude-sonnet-4-6": {
        "input_per_mtok": 3.0, "output_per_mtok": 15.0,
        "cache_write_per_mtok": 3.75, "cache_read_per_mtok": 0.30,
    }}
    df = build_unit_table(
        runs_dir=runs_dir, eval_reports=eval_reports,
        agent_model="claude-sonnet-4-6", pricing=pricing,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert set(df.columns) >= {
        "instance_id", "condition_id", "replicate",
        "resolved", "num_turns", "ccb", "cost_usd", "wall_clock_secs",
    }
    row = df[(df.instance_id == "r0") & (df.condition_id == "c1")].iloc[0]
    assert row.resolved == True
    row2 = df[(df.instance_id == "r1") & (df.condition_id == "c1")].iloc[0]
    assert row2.resolved == False
