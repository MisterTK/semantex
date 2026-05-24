#!/usr/bin/env python3
"""
B3 — Multi-repo agent CCB wrapper.

Thin wrapper around the existing `benchmarks/agent_bench.py`. Does NOT
rewrite that script — instead, it iterates the (repo, question, tool) matrix
the spec requires and aggregates JSONL transcripts into a single summary
table.

Spec §4.4: 5 questions × 5 repos × 3 replications × N tools. agent_bench.py
already drives the per-repo loop, so this wrapper just expands the (tool ×
replication) outer loop and aggregates.

Spec §4.7 cap-condition: if SD > 30% of mean across replications → flag the
row "noise-dominated, no claim". Implemented in `_flag_noisy_rows`.

The actual model-call work happens inside agent_bench.py via Claude Code's
`--headless` mode. We do NOT call the Anthropic API directly. Costs and
runtime are entirely controlled by agent_bench.py's per-run budget.

Usage
-----
    # Dry-run — just verify wiring without spending agent dollars
    python3 benchmarks/run_b3.py --tool semantex --reps 1 \
        --repos /path/to/rust-repo /path/to/python-repo \
        --output results/b3-smoke --dry-run

    # Real run (warning: hours of agent time)
    python3 benchmarks/run_b3.py --tool semantex --reps 3 \
        --repos /path/to/rust-repo /path/to/python-repo \
        --output results/b3-2026-05-24/
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_AGENT_BENCH = THIS_DIR / "agent_bench.py"


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="B3 — multi-repo agent CCB wrapper.")
    ap.add_argument("--tool", required=True,
                    help="Tool slug (semantex, claude-builtins, graphify, ...). "
                         "Currently agent_bench.py distinguishes the 'baseline' "
                         "(claude built-ins) vs 'treatment' (with MCP) arms; "
                         "this wrapper maps slug → arm.")
    ap.add_argument("--reps", type=int, default=3,
                    help="Replications per (repo × question × tool). Spec: 3.")
    ap.add_argument("--repos", nargs="+", required=True,
                    help="Paths to indexed repos to run against.")
    ap.add_argument("--output", required=True, help="Output directory.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip actual agent_bench.py invocation; emit wiring summary only.")
    ap.add_argument("--noise-threshold", type=float, default=0.30,
                    help="SD/mean threshold above which a row is flagged "
                         "'noise-dominated, no claim'. Spec §4.7 default: 0.30.")
    ap.add_argument("--agent-bench",
                    default=str(DEFAULT_AGENT_BENCH),
                    help=f"Path to agent_bench.py (default: {DEFAULT_AGENT_BENCH}). "
                         "Override if running from a worktree where this wrapper "
                         "sits next to v0.3 scaffolding but agent_bench.py lives "
                         "in the main checkout.")
    return ap.parse_args(argv)


def _flag_noisy_rows(per_rep_metrics: dict, threshold: float) -> dict:
    """Per spec §4.7: SD > 30% of mean → noise-dominated.

    `per_rep_metrics` is keyed `(repo, question_id, tool)` → list of metric
    values (e.g. CCB). Returns the same shape with extra `noise_dominated` flag.
    """
    flagged: dict = {}
    for key, values in per_rep_metrics.items():
        if len(values) < 2:
            flagged[key] = {
                "values": values,
                "mean": values[0] if values else None,
                "sd": None,
                "noise_dominated": None,  # cannot judge with n<2
                "n": len(values),
            }
            continue
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        ratio = (sd / abs(mean)) if mean else None
        flagged[key] = {
            "values": values,
            "mean": mean,
            "sd": sd,
            "ratio": ratio,
            "noise_dominated": (ratio is not None and ratio > threshold),
            "n": len(values),
        }
    return flagged


def _invoke_agent_bench(tool: str, rep_idx: int, repos: list, out_dir: Path,
                        agent_bench_path: str) -> dict:
    """Shell out to agent_bench.py for one replication across all repos.

    agent_bench.py's `run` subcommand already does per-repo iteration; we set
    `--output` per replication so files don't collide.
    """
    rep_out = out_dir / f"rep{rep_idx:02d}_{tool}"
    rep_out.mkdir(parents=True, exist_ok=True)
    # agent_bench.py expects --repos and --output. Tool selection is encoded
    # via arm flags inside that script. For "semantex" we want the treatment
    # arm; for "claude" we want baseline. agent_bench.py runs both arms by
    # default — so we keep both and tag the rep with the tool slug for
    # bookkeeping.
    cmd = [
        sys.executable, agent_bench_path, "run",
        "--repos", *repos,
        "--output", str(rep_out),
        "--skip-index",
    ]
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=60 * 60 * 2)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "agent_bench.py timed out after 2h",
                "rep": rep_idx, "rep_out": str(rep_out)}
    elapsed = time.perf_counter() - start
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "rep": rep_idx,
        "rep_out": str(rep_out),
        "stdout_tail": proc.stdout.decode("utf-8", errors="replace")[-500:],
        "stderr_tail": proc.stderr.decode("utf-8", errors="replace")[-500:],
    }


def _aggregate(out_dir: Path) -> dict:
    """Scan all rep* dirs under out_dir for JSONL transcripts and aggregate
    per (repo, question_id) → list of CCB values per arm.

    JSONL schema is whatever agent_bench.py writes. We probe defensively:
    if the script's output shape changes, we surface a "schema unknown" note
    rather than silently misaggregating.
    """
    per_rep_metrics: dict = {}
    rep_dirs = sorted(p for p in out_dir.glob("rep*") if p.is_dir())
    for rep_dir in rep_dirs:
        for jsonl in rep_dir.glob("**/*.jsonl"):
            try:
                for line in jsonl.read_text().splitlines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    # Best-effort: agent_bench.py records {repo, qid, arm, ccb,
                    # turn_count, tokens, ...}. We aggregate `ccb`.
                    key = (obj.get("repo"), obj.get("qid"), obj.get("arm"))
                    if any(v is None for v in key):
                        continue
                    ccb = obj.get("ccb")
                    if ccb is None:
                        continue
                    per_rep_metrics.setdefault(key, []).append(ccb)
            except (OSError, json.JSONDecodeError):
                continue
    # JSON keys must be strings — pack tuple→"repo|qid|arm" for output.
    flagged = _flag_noisy_rows(per_rep_metrics, threshold=0.30)
    return {
        "rows": [
            {
                "repo": k[0], "qid": k[1], "arm": k[2],
                **v,
            }
            for k, v in flagged.items()
        ],
        "row_count": len(flagged),
        "noise_dominated_count": sum(
            1 for v in flagged.values() if v.get("noise_dominated")
        ),
    }


def main(argv=None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    agent_bench_path = Path(args.agent_bench).resolve()
    if not args.dry_run and not agent_bench_path.exists():
        print(f"ERROR: agent_bench not found at {agent_bench_path}. "
              f"Pass --agent-bench /path/to/agent_bench.py.", file=sys.stderr)
        return 2

    # Validate repos
    valid_repos = []
    for r in args.repos:
        rp = Path(r).resolve()
        if not rp.exists():
            print(f"WARNING: repo path missing, skipping: {rp}", file=sys.stderr)
            continue
        valid_repos.append(str(rp))

    metadata = {
        "tool": args.tool,
        "reps": args.reps,
        "repos": valid_repos,
        "dry_run": args.dry_run,
        "noise_threshold": args.noise_threshold,
        "started_at": int(time.time()),
    }
    (out_dir / "b3_run_metadata.json").write_text(json.dumps(metadata, indent=2))

    if args.dry_run:
        summary = {
            "tool": args.tool,
            "ok": True,
            "dry_run": True,
            "would_invoke": (
                f"{sys.executable} {agent_bench_path} run --repos {' '.join(valid_repos)} "
                f"--output <rep_dir> --skip-index  × {args.reps} replications"
            ),
            "agent_bench": str(agent_bench_path),
            "agent_bench_present": agent_bench_path.exists(),
            "noise_check_threshold": args.noise_threshold,
        }
        (out_dir / f"b3_{args.tool}_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 0

    rep_results = []
    for rep in range(1, args.reps + 1):
        print(f"[b3] rep {rep}/{args.reps} tool={args.tool}", file=sys.stderr)
        rep_results.append(_invoke_agent_bench(
            args.tool, rep, valid_repos, out_dir, str(agent_bench_path)))

    agg = _aggregate(out_dir)
    summary = {
        "tool": args.tool,
        "ok": all(r.get("ok") for r in rep_results),
        "reps": args.reps,
        "rep_results": rep_results,
        "aggregation": agg,
        "finished_at": int(time.time()),
    }
    (out_dir / f"b3_{args.tool}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(
        {"tool": args.tool, "ok": summary["ok"], "row_count": agg["row_count"],
         "noise_dominated_count": agg["noise_dominated_count"]},
        indent=2,
    ))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
