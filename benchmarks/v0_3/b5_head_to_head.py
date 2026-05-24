#!/usr/bin/env python3
"""
B5 — Cross-tool head-to-head.

Per spec §4.1, B5 takes all B3 questions and runs each through ≥4 alternative
tools, reporting win rate per question, magnitude of win, and paired t-test
significance.

This is a derived benchmark — it does no fresh measurements of its own,
just aggregates B3 outputs across tools. The aggregator lives here so
`run_public.py` can call it after every B3 invocation finishes.

Scaffolding only at W8: emits a placeholder JSON noting that aggregation
runs once B3 produces results for ≥2 tools.
"""

from __future__ import annotations

import json
from pathlib import Path


def run(*, output_dir: Path, b3_results_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    tools_seen: list = []
    if b3_results_dir.exists():
        for f in sorted(b3_results_dir.glob("b3_*_summary.json")):
            try:
                payload = json.loads(f.read_text())
                tools_seen.append(payload.get("tool", f.stem))
            except (OSError, json.JSONDecodeError):
                continue

    summary = {
        "ok": len(tools_seen) >= 2,
        "tools_with_b3_data": tools_seen,
        "reason": (
            "B5 aggregates across ≥2 tools; have data for "
            f"{len(tools_seen)}: {tools_seen}"
        ),
        "todo": [
            "Implement per-question paired t-test (n=3+, alpha=0.05)",
            "Compute win rate per question type (architecture, error_handling, ...)",
            "Render leaderboard table; mark statistically-insignificant rows",
            "Spec §4.7 check: SD > 30% of mean → 'noise-dominated, no claim'",
        ],
    }
    (output_dir / "b5_status.json").write_text(json.dumps(summary, indent=2))
    return summary
