"""Join per-unit run outputs with swebench eval reports → tidy DataFrame."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .metrics import ccb, cost_usd, num_turns, tool_distribution


def build_unit_table(
    *,
    runs_dir: Path,
    eval_reports: dict[str, dict],   # key = f"{condition_id}__{replicate}"
    agent_model: str,
    pricing: dict,
) -> pd.DataFrame:
    """One row per (instance, condition, replicate) unit."""
    rows = []
    for jf in sorted(runs_dir.glob("*.json")):
        data = json.loads(jf.read_text())
        cond_rep = f"{data['condition_id']}__{data['replicate']}"
        report = eval_reports.get(cond_rep, {})
        resolved_ids = set(report.get("resolved_ids", []))
        submitted_ids = set(report.get("submitted_ids", []))
        turns = data.get("turns", [])
        rows.append({
            "instance_id": data["instance_id"],
            "condition_id": data["condition_id"],
            "replicate": data["replicate"],
            "resolved": data["instance_id"] in resolved_ids,
            "submitted": data["instance_id"] in submitted_ids,
            "num_turns": num_turns(turns),
            "ccb": ccb(turns),
            "cost_usd": cost_usd(turns, model=agent_model, pricing=pricing),
            "wall_clock_secs": data.get("wall_clock_secs", 0.0),
            "tool_distribution": tool_distribution(turns),
            "error": data.get("error", ""),
        })
    return pd.DataFrame(rows)
