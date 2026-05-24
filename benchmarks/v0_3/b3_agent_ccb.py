#!/usr/bin/env python3
"""
B3 — Multi-repo Agent CCB sub-benchmark wrapper.

Per spec §4.4: 5 repos × 5 question categories × 3 replications × N tools
= 375 agent runs per release. That's hours of real-LLM time and is the human's
job to invoke. This module is a structural wrapper around the existing
`benchmarks/agent_bench.py`.

Today: dispatches to `benchmarks/run_b3.py`, which itself wraps agent_bench.py
on a per-(repo,question,tool) basis. The wrapper here exists so run_public.py
has one uniform entry point per sub-benchmark.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run(*, tool: str, output_dir: Path, repos: list, dry_run: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    wrapper = Path(__file__).resolve().parent.parent / "run_b3.py"
    if not wrapper.exists():
        return {
            "tool": tool,
            "ok": False,
            "reason": "benchmarks/run_b3.py missing — should not happen on W8 scaffold",
        }

    if dry_run:
        out = {
            "tool": tool,
            "ok": True,
            "dry_run": True,
            "would_invoke": f"{sys.executable} {wrapper} --tool {tool} --repos {' '.join(repos)} --output {output_dir}",
            "n_repos": len(repos),
            "note": "B3 is multi-hour and human-driven. Scaffold confirms wiring only.",
        }
        (output_dir / f"b3_{tool}_dryrun.json").write_text(json.dumps(out, indent=2))
        return out

    # Real run: shell out to run_b3.py. The human will typically invoke that
    # script directly with replication count; we expose a one-rep default here
    # so smoke testing doesn't spend hours.
    cmd = [
        sys.executable, str(wrapper),
        "--tool", tool,
        "--reps", "1",
        "--output", str(output_dir),
        "--repos", *repos,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=60 * 60 * 6)
    except subprocess.TimeoutExpired:
        return {"tool": tool, "ok": False, "reason": "run_b3.py timed out after 6h"}
    summary_file = output_dir / f"b3_{tool}_summary.json"
    summary = {}
    if summary_file.exists():
        try:
            summary = json.loads(summary_file.read_text())
        except json.JSONDecodeError:
            pass
    return {
        "tool": tool,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout.decode("utf-8", errors="replace")[-1000:],
        "stderr_tail": proc.stderr.decode("utf-8", errors="replace")[-1000:],
        "summary": summary,
    }
