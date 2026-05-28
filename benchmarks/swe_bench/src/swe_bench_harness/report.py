"""Aggregate the tidy DataFrame into per-condition summaries + leaderboard JSON."""
from __future__ import annotations

import pandas as pd


def summarize_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Per-condition: mean(metrics), resolution_rate across replicates."""
    return df.groupby("condition_id").agg(
        n_units=("instance_id", "count"),
        resolution_rate=("resolved", "mean"),
        mean_ccb=("ccb", "mean"),
        mean_turns=("num_turns", "mean"),
        mean_cost_usd=("cost_usd", "mean"),
        mean_wall_clock_secs=("wall_clock_secs", "mean"),
    ).reset_index()


def leaderboard_submission_dict(
    *,
    df: pd.DataFrame,
    condition_id: str,
    replicate: int,
    system_name: str,
) -> dict:
    """Format a single (condition, replicate) for SWE-bench leaderboard submission."""
    sub = df[(df.condition_id == condition_id) & (df.replicate == replicate)]
    resolved = sorted(sub[sub.resolved].instance_id.tolist())
    return {
        "system": system_name,
        "condition_id": condition_id,
        "replicate": replicate,
        "resolution_rate": float(sub.resolved.mean()) if len(sub) else 0.0,
        "resolved_instances": resolved,
        "submitted_count": int(sub.submitted.sum()),
        "total_count": int(len(sub)),
    }


def render_markdown_report(
    *,
    df: pd.DataFrame,
    summary: pd.DataFrame,
    paired_tests: list[dict],
) -> str:
    lines = []
    lines.append("# SWE-bench Verified Results\n")
    lines.append("## Per-condition summary\n")
    lines.append(summary.to_markdown(index=False, floatfmt=".3f") + "\n")
    lines.append("## Paired comparisons\n")
    for t in paired_tests:
        lines.append(
            f"- **{t['treatment']} vs {t['baseline']}**: "
            f"lift = {t['treatment_lift_pp']:+.2f}pp, p = {t['p_value']:.4f} "
            f"(b={t['b']}, c={t['c']})\n"
        )
    return "".join(lines)
