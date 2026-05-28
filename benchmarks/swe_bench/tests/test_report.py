import pandas as pd

from swe_bench_harness.report import (
    leaderboard_submission_dict, summarize_by_condition,
)


def _df():
    return pd.DataFrame([
        {"instance_id":"r0","condition_id":"c1","replicate":0,"resolved":True,"submitted":True,"num_turns":10,"ccb":1000,"cost_usd":0.5,"wall_clock_secs":120,"tool_distribution":{},"error":""},
        {"instance_id":"r1","condition_id":"c1","replicate":0,"resolved":False,"submitted":True,"num_turns":20,"ccb":4000,"cost_usd":1.5,"wall_clock_secs":300,"tool_distribution":{},"error":""},
        {"instance_id":"r0","condition_id":"c2","replicate":0,"resolved":True,"submitted":True,"num_turns":7,"ccb":700,"cost_usd":0.4,"wall_clock_secs":100,"tool_distribution":{},"error":""},
        {"instance_id":"r1","condition_id":"c2","replicate":0,"resolved":True,"submitted":True,"num_turns":12,"ccb":2400,"cost_usd":1.0,"wall_clock_secs":180,"tool_distribution":{},"error":""},
    ])


def test_summarize_by_condition_returns_means_and_resolution_rate():
    df = _df()
    summary = summarize_by_condition(df)
    assert set(summary.condition_id) == {"c1", "c2"}
    c1 = summary[summary.condition_id == "c1"].iloc[0]
    assert c1.resolution_rate == 0.5
    assert c1.mean_ccb == 2500
    c2 = summary[summary.condition_id == "c2"].iloc[0]
    assert c2.resolution_rate == 1.0


def test_leaderboard_submission_structure():
    df = _df()
    out = leaderboard_submission_dict(
        df=df, condition_id="c2", replicate=0, system_name="semantex+OpenHands+Sonnet-4.6",
    )
    assert out["system"] == "semantex+OpenHands+Sonnet-4.6"
    assert "resolved_instances" in out
    assert "resolution_rate" in out
    assert set(out["resolved_instances"]) == {"r0", "r1"}
