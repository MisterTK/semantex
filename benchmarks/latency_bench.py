#!/usr/bin/env python3
"""
B4 — Latency micro-benchmark (semantex v0.3 SOTA release infra).

Times search calls against a pre-built semantex index and reports
p50/p95/p99 for both cold-start and warm-state latency.

Spec validation targets (from docs/superpowers/specs/2026-05-24-...):
  * p50 warm  ≤ 10ms   (today: ~17ms)
  * p50 cold  ≤ 300ms  (today: ~540ms)

This script measures what semantex actually does today; it does NOT
synthesize numbers. If the daemon is not warm, the first call pays
the cold-start cost; subsequent calls are warm.

Usage
-----
    # Default: 30 queries × 5 timed runs per query against the target repo
    python3 benchmarks/latency_bench.py --target /path/to/repo

    # Tiny smoke set (sample run requested by W8 spec)
    python3 benchmarks/latency_bench.py --queries 5 --target /path/to/repo

    # Custom binary
    SEMANTEX_BIN=/path/to/semantex python3 benchmarks/latency_bench.py \
        --target /path/to/repo

    # JSON output for downstream scripts / aggregation
    python3 benchmarks/latency_bench.py --target /path/to/repo --json

Query set
---------
The 30-query mini-set is intentionally identifier-style ASCII tokens —
universal across languages and codebases. The set is derived from the
generic categories already in `benchmarks/agent_bench.py` rather than
project-specific identifiers, so the bench works against any indexed repo.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


DEFAULT_BIN = (
    os.environ.get("SEMANTEX_BIN")
    or "/usr/local/bin/semantex"
)


# 30-query mini-set. Generic concepts, no project-specific identifiers.
# Curated to span: data flow, error handling, async, IO, parsing, testing,
# config, persistence — i.e. concepts present in nearly every codebase.
QUERIES_30 = [
    "error handling",
    "configuration loading",
    "logging setup",
    "async function",
    "database query",
    "http request handler",
    "json serialization",
    "file read write",
    "command line argument parsing",
    "test fixture",
    "main entry point",
    "retry logic",
    "cache invalidation",
    "rate limit",
    "authentication middleware",
    "input validation",
    "websocket connection",
    "thread safety",
    "memory allocation",
    "string formatting",
    "regular expression match",
    "iterator implementation",
    "trait implementation",
    "interface definition",
    "channel send receive",
    "graceful shutdown",
    "background worker",
    "schema migration",
    "metrics emission",
    "dependency injection",
]


@dataclass
class TimedCall:
    query: str
    elapsed_ms: float
    returncode: int
    bytes_out: int
    error: Optional[str] = None


@dataclass
class LatencyReport:
    semantex_bin: str
    target: str
    query_count: int
    runs_per_query: int
    cold_ms: list = field(default_factory=list)       # one per query (first run)
    warm_ms: list = field(default_factory=list)       # all subsequent runs across queries
    failed_calls: list = field(default_factory=list)  # TimedCall dicts for failures
    # Per-query first-run latency in source order. Always populated, so the text
    # report can show "this specific query took N ms on its first call."
    per_query_first_ms: list = field(default_factory=list)

    @property
    def cold_p50(self) -> Optional[float]:
        return statistics.median(self.cold_ms) if self.cold_ms else None

    @property
    def warm_p50(self) -> Optional[float]:
        return statistics.median(self.warm_ms) if self.warm_ms else None

    @staticmethod
    def percentile(data: list, pct: float) -> Optional[float]:
        if not data:
            return None
        s = sorted(data)
        if len(s) == 1:
            return s[0]
        # nearest-rank percentile (inclusive)
        k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
        return s[k]

    def to_dict(self) -> dict:
        return {
            "semantex_bin": self.semantex_bin,
            "target": self.target,
            "query_count": self.query_count,
            "runs_per_query": self.runs_per_query,
            "per_query_first_ms": self.per_query_first_ms,
            "cold": {
                "n": len(self.cold_ms),
                "p50": self.cold_p50,
                "p95": self.percentile(self.cold_ms, 95),
                "p99": self.percentile(self.cold_ms, 99),
                "min": min(self.cold_ms) if self.cold_ms else None,
                "max": max(self.cold_ms) if self.cold_ms else None,
            },
            "warm": {
                "n": len(self.warm_ms),
                "p50": self.warm_p50,
                "p95": self.percentile(self.warm_ms, 95),
                "p99": self.percentile(self.warm_ms, 99),
                "min": min(self.warm_ms) if self.warm_ms else None,
                "max": max(self.warm_ms) if self.warm_ms else None,
            },
            "failed_calls": self.failed_calls,
            "spec_targets": {
                "warm_p50_ms": 10,
                "cold_p50_ms": 300,
            },
        }


def run_one(semantex_bin: str, target: str, query: str, timeout_s: float = 30.0) -> TimedCall:
    """Single timed invocation of `semantex --refs ... -p <target> <query>`."""
    cmd = [
        semantex_bin,
        "--refs",            # smallest output payload — measures search latency, not I/O
        "--max-count", "10",
        "-p", target,
        query,
    ]
    start = time.perf_counter()
    err: Optional[str] = None
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout_s,
        )
        rc = proc.returncode
        out = proc.stdout
        if rc != 0:
            err = proc.stderr.decode("utf-8", errors="replace")[:300]
    except subprocess.TimeoutExpired:
        rc = -1
        out = b""
        err = f"timeout after {timeout_s}s"
    except FileNotFoundError as e:
        rc = -2
        out = b""
        err = f"binary not found: {e}"
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return TimedCall(
        query=query,
        elapsed_ms=elapsed_ms,
        returncode=rc,
        bytes_out=len(out),
        error=err,
    )


def stop_daemon(semantex_bin: str, target: str) -> None:
    """Best-effort daemon stop to reset cold-start state.

    semantex's `stop` subcommand sends a shutdown to a running daemon for the
    given project. Errors are swallowed — if the daemon isn't running, the next
    call is naturally cold.
    """
    try:
        subprocess.run(
            [semantex_bin, "stop", target],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass  # noqa: BLE001 — best-effort


def run_bench(
    semantex_bin: str,
    target: str,
    queries: list,
    runs_per_query: int,
    cold_reset_each: bool,
) -> LatencyReport:
    rep = LatencyReport(
        semantex_bin=semantex_bin,
        target=target,
        query_count=len(queries),
        runs_per_query=runs_per_query,
    )

    # Establish a known cold state once before the loop. If cold_reset_each is
    # True we also stop the daemon between queries (slower, more honest cold #s).
    stop_daemon(semantex_bin, target)
    time.sleep(0.2)  # let port file release; small fixed pause

    for i, q in enumerate(queries):
        if i > 0 and cold_reset_each:
            stop_daemon(semantex_bin, target)
            time.sleep(0.2)

        for run_idx in range(runs_per_query):
            call = run_one(semantex_bin, target, q)
            if call.returncode != 0:
                rep.failed_calls.append(asdict(call))
                if run_idx == 0:
                    rep.per_query_first_ms.append(None)
                continue
            if run_idx == 0:
                rep.per_query_first_ms.append(call.elapsed_ms)
                # First call after (potentially) cold start: counted as cold
                # only when cold_reset_each is True OR i == 0 (first query overall).
                if cold_reset_each or i == 0:
                    rep.cold_ms.append(call.elapsed_ms)
                else:
                    rep.warm_ms.append(call.elapsed_ms)
            else:
                rep.warm_ms.append(call.elapsed_ms)

    return rep


def fmt_ms(v: Optional[float]) -> str:
    return f"{v:.1f}ms" if v is not None else "n/a"


def print_text_report(rep: LatencyReport, queries: list) -> None:
    d = rep.to_dict()
    print(f"semantex latency benchmark — {rep.target}")
    print(f"  binary:        {rep.semantex_bin}")
    print(f"  queries:       {rep.query_count}")
    print(f"  runs/query:    {rep.runs_per_query}")
    print(f"  failed calls:  {len(rep.failed_calls)}")
    print()

    print("Per-query first-run elapsed (ms):")
    for q, t in zip(queries, rep.per_query_first_ms):
        t_str = f"{t:8.1f}" if t is not None else "  FAILED"
        print(f"  {t_str}  {q!r}")
    print()

    cold = d["cold"]
    warm = d["warm"]
    print("Cold (first call against fresh daemon):")
    print(f"  n={cold['n']}  p50={fmt_ms(cold['p50'])}  p95={fmt_ms(cold['p95'])}  p99={fmt_ms(cold['p99'])}  min={fmt_ms(cold['min'])}  max={fmt_ms(cold['max'])}")
    print()
    print("Warm (daemon already loaded):")
    print(f"  n={warm['n']}  p50={fmt_ms(warm['p50'])}  p95={fmt_ms(warm['p95'])}  p99={fmt_ms(warm['p99'])}  min={fmt_ms(warm['min'])}  max={fmt_ms(warm['max'])}")
    print()

    # Spec target check (honest — pass/fail without spin)
    targets = d["spec_targets"]
    warm_ok = warm["p50"] is not None and warm["p50"] <= targets["warm_p50_ms"]
    cold_ok = cold["p50"] is not None and cold["p50"] <= targets["cold_p50_ms"]
    print("Spec targets (v0.3 SOTA design Section 1):")
    print(f"  warm p50 ≤ {targets['warm_p50_ms']}ms : {'PASS' if warm_ok else 'FAIL'}  (measured {fmt_ms(warm['p50'])})")
    print(f"  cold p50 ≤ {targets['cold_p50_ms']}ms : {'PASS' if cold_ok else 'FAIL'}  (measured {fmt_ms(cold['p50'])})")

    if rep.failed_calls:
        print()
        print(f"Failed calls ({len(rep.failed_calls)}):")
        for f in rep.failed_calls[:5]:
            print(f"  rc={f['returncode']} q={f['query']!r}  err={f.get('error', '')[:80]}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Latency micro-bench for semantex (B4 of v0.3 SOTA bench suite)."
    )
    ap.add_argument(
        "--target",
        required=True,
        help="Path to an already-indexed repo (must contain a .semantex/ subdir).",
    )
    ap.add_argument(
        "--queries",
        type=int,
        default=30,
        help="Number of queries from the 30-query mini-set to run (default: 30).",
    )
    ap.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Timed runs per query (first counted as cold, rest as warm). Default: 5.",
    )
    ap.add_argument(
        "--cold-reset-each",
        action="store_true",
        help="Stop the daemon between queries to measure cold latency on every query. "
             "Slower but honest cold-start sample. Default: only the very first call is cold.",
    )
    ap.add_argument(
        "--bin",
        default=DEFAULT_BIN,
        help=f"Path to semantex binary (default: {DEFAULT_BIN}; or set SEMANTEX_BIN env).",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON report to stdout instead of text.",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Write JSON report to this path in addition to stdout text.",
    )
    args = ap.parse_args()

    target = os.path.abspath(args.target)
    if not Path(target).exists():
        print(f"ERROR: target path does not exist: {target}", file=sys.stderr)
        return 2

    semantex_index = Path(target) / ".semantex"
    if not semantex_index.exists():
        print(
            f"ERROR: {target} is not indexed. Run: {args.bin} index {target}",
            file=sys.stderr,
        )
        return 2

    if not Path(args.bin).exists() and not os.path.basename(args.bin) == args.bin:
        print(
            f"ERROR: semantex binary not found at {args.bin}. "
            f"Set SEMANTEX_BIN or run `cargo build --release -p semantex-cli`.",
            file=sys.stderr,
        )
        return 2

    queries = QUERIES_30[: max(1, min(args.queries, len(QUERIES_30)))]
    runs = max(1, args.runs)

    rep = run_bench(
        semantex_bin=args.bin,
        target=target,
        queries=queries,
        runs_per_query=runs,
        cold_reset_each=args.cold_reset_each,
    )

    if args.json:
        print(json.dumps(rep.to_dict(), indent=2))
    else:
        print_text_report(rep, queries)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rep.to_dict(), indent=2))
        if not args.json:
            print()
            print(f"JSON report written to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
