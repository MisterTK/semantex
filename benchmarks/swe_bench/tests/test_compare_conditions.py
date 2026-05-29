"""Unit test for scripts.compare_conditions.compare() — the C2-vs-C1 delta logic."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# scripts/ isn't a package on the harness install path; add it for the test.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from compare_conditions import _semantex_calls, _used_semantex, compare  # noqa: E402


def _df():
    """Synthetic 4-instance dataset, paired across C1 and C2."""
    return pd.DataFrame([
        # C1 baseline
        {"instance_id": "r0", "condition_id": "c1", "replicate": 0,
         "resolved": True,  "num_turns": 10, "ccb": 1000, "cost_usd": 0.5,
         "tool_distribution": {"terminal": 5, "file_editor": 3}},
        {"instance_id": "r1", "condition_id": "c1", "replicate": 0,
         "resolved": False, "num_turns": 75, "ccb": 8000, "cost_usd": 4.0,  # hard: failed
         "tool_distribution": {"terminal": 40, "file_editor": 20}},
        {"instance_id": "r2", "condition_id": "c1", "replicate": 0,
         "resolved": True,  "num_turns": 25, "ccb": 3000, "cost_usd": 1.5,  # hard: >20 turns
         "tool_distribution": {"terminal": 15, "file_editor": 8}},
        {"instance_id": "r3", "condition_id": "c1", "replicate": 0,
         "resolved": True,  "num_turns": 5,  "ccb": 500,  "cost_usd": 0.3,
         "tool_distribution": {"terminal": 3, "file_editor": 1}},
        # C2 semantex (treatment)
        {"instance_id": "r0", "condition_id": "c2", "replicate": 0,
         "resolved": True,  "num_turns": 8,  "ccb": 800,  "cost_usd": 0.4,
         "tool_distribution": {"terminal": 4, "file_editor": 2, "semantex_agent": 1}},
        {"instance_id": "r1", "condition_id": "c2", "replicate": 0,
         "resolved": True,  "num_turns": 30, "ccb": 5000, "cost_usd": 2.0,  # WAS hard, c2 fixed
         "tool_distribution": {"terminal": 12, "file_editor": 8, "semantex_agent": 3}},
        {"instance_id": "r2", "condition_id": "c2", "replicate": 0,
         "resolved": True,  "num_turns": 18, "ccb": 2000, "cost_usd": 1.0,  # WAS hard, c2 fixed faster
         "tool_distribution": {"terminal": 10, "file_editor": 5, "semantex_agent": 2}},
        {"instance_id": "r3", "condition_id": "c2", "replicate": 0,
         "resolved": True,  "num_turns": 5,  "ccb": 500,  "cost_usd": 0.3,  # easy, no semantex used
         "tool_distribution": {"terminal": 3, "file_editor": 1}},
    ])


def test_semantex_helpers():
    assert _semantex_calls({"terminal": 5, "semantex_agent": 2}) == 2
    assert _semantex_calls({"terminal": 5}) == 0
    assert _used_semantex({"semantex_agent": 1}) is True
    assert _used_semantex({"terminal": 5}) is False
    assert _used_semantex({}) is False


def test_compare_full_pipeline():
    r = compare(_df(), baseline="c1", treatment="c2")

    # Pairing
    assert r["n_paired_instances"] == 4

    # 1. Resolution rate: c1 = 3/4, c2 = 4/4
    assert r["rr_baseline"] == 0.75
    assert r["rr_treatment"] == 1.0
    assert r["rr_lift_pp"] == 25.0
    # McNemar discordant: b=baseline-only-resolved=0, c=treatment-only-resolved=1
    assert r["mcnemar_b"] == 0
    assert r["mcnemar_c"] == 1

    # 2. semantex usage: 3 of 4 c2 units called it
    assert r["semantex_use_rate"] == 0.75
    assert r["mean_semantex_calls"] == 1.5  # (1+3+2+0) / 4
    assert r["max_semantex_calls"] == 3

    # 3. Mean turns: c1=28.75, c2=15.25 → delta -13.5
    assert r["turns_baseline"] == 28.75
    assert r["turns_treatment"] == 15.25
    assert r["turns_delta_mean"] == -13.5
    # CI should bracket the point estimate
    assert r["turns_delta_ci"][0] <= -13.5 <= r["turns_delta_ci"][1] + 1e-6 or \
           r["turns_delta_ci"][0] - 1e-6 <= -13.5

    # 4. CCB: baseline=3125 avg, treatment=2075 avg → ~-33.6%
    assert abs(r["ccb_baseline"] - 3125.0) < 1e-6
    assert abs(r["ccb_treatment"] - 2075.0) < 1e-6
    assert r["ccb_delta_pct"] < 0  # treatment is cheaper

    # 5. Cost ratios
    assert r["cost_baseline"] == (0.5 + 4.0 + 1.5 + 0.3) / 4
    assert r["cost_treatment"] == (0.4 + 2.0 + 1.0 + 0.3) / 4

    # 6. Resolved per $: c1 = 3/6.3 = 0.476; c2 = 4/3.7 = 1.081
    assert r["rpd_baseline"] == 3 / (0.5 + 4.0 + 1.5 + 0.3)
    assert r["rpd_treatment"] == 4 / (0.4 + 2.0 + 1.0 + 0.3)

    # 7. Hard subset: r1 (failed) and r2 (>20 turns) → 2 hard instances
    assert r["hard_n"] == 2
    # c1 resolved 1 of 2 hard (r2); c2 resolved 2 of 2 (r1, r2)
    assert r["hard_rr_baseline"] == 0.5
    assert r["hard_rr_treatment"] == 1.0
    assert r["hard_lift_pp"] == 50.0


def test_compare_no_overlap_returns_zero_pairs():
    df = pd.DataFrame([
        {"instance_id": "a", "condition_id": "c1", "replicate": 0,
         "resolved": True, "num_turns": 5, "ccb": 100, "cost_usd": 0.1,
         "tool_distribution": {}},
        {"instance_id": "b", "condition_id": "c2", "replicate": 0,
         "resolved": True, "num_turns": 5, "ccb": 100, "cost_usd": 0.1,
         "tool_distribution": {}},
    ])
    r = compare(df, baseline="c1", treatment="c2")
    assert r["n_paired_instances"] == 0
