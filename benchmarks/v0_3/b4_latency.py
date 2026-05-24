#!/usr/bin/env python3
"""
B4 — Latency sub-benchmark wrapper.

This thin shim invokes `benchmarks/latency_bench.py` and writes the JSON
report into the output directory expected by run_public.py.

Keeping the heavy lifting in latency_bench.py means the standalone latency
bench works on its own (matching the W8 deliverable: "B4 latency micro-bench
(cargo bench style)") without depending on this shim.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def run(*, tool: str, target_repo: str, output_dir: Path, semantex_bin: str) -> dict:
    """Run B4 for one tool against one repo, write JSON, return summary dict."""
    if tool != "semantex":
        # B4 is currently semantex-only — the spec wants tool latency comparisons
        # under B5 (head-to-head). Other tools shell out to ad-hoc binaries
        # whose cold/warm semantics differ, so they live in B5.
        return {
            "tool": tool,
            "skipped": True,
            "reason": "B4 measures semantex cold/warm; other-tool latency lives under B5",
        }

    bench_script = Path(__file__).resolve().parent.parent / "latency_bench.py"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"b4_latency_{Path(target_repo).name}.json"

    env = os.environ.copy()
    env.setdefault("SEMANTEX_BIN", semantex_bin)

    cmd = [
        sys.executable,
        str(bench_script),
        "--target", target_repo,
        "--queries", "30",
        "--runs", "5",
        "--bin", semantex_bin,
        "--json",
        "--output", str(json_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, env=env, timeout=900)
    if proc.returncode != 0:
        return {
            "tool": tool,
            "ok": False,
            "stderr": proc.stderr.decode("utf-8", errors="replace")[:500],
            "stdout_head": proc.stdout.decode("utf-8", errors="replace")[:500],
        }

    try:
        report = json.loads(json_path.read_text())
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return {"tool": tool, "ok": False, "stderr": f"failed to parse report: {e}"}

    return {
        "tool": tool,
        "ok": True,
        "json_path": str(json_path),
        "summary": {
            "warm_p50_ms": report.get("warm", {}).get("p50"),
            "warm_p95_ms": report.get("warm", {}).get("p95"),
            "cold_p50_ms": report.get("cold", {}).get("p50"),
            "cold_p95_ms": report.get("cold", {}).get("p95"),
        },
    }
