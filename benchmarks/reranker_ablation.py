#!/usr/bin/env python3
"""
Reranker ablation harness (W3 / E1)
====================================

Runs the owned 5-query suite WITH and WITHOUT the cross-encoder reranker
(`SEMANTEX_RERANKER=on` vs `off`) and reports per-query F1 deltas plus an
overall verdict. Spec reference: v0.3-sota-design Section 3 / E1, validation
clause "ablation on owned 5-query suite (must show >=+5pp F1) and zero
regression on Q21/Q25 hard failures".

Selection rationale (5-query subset of the 30-query grep_vs_semantex_bench):

1. Q9  exponential backoff       (semantic, lexically clear, reranker should help)
2. Q12 factory pattern           (semantic, NL>code vocabulary mismatch)
3. Q20 parallel failure handling (semantic, Q21 "hard failure" per RESEARCH doc)
4. Q25 connection lifecycle      (architectural, Q25 "hard failure" per RESEARCH doc)
5. Q29 BigQuery schema/querying  (architectural, multi-file cross-app trace)

Q20 and Q25 are the "hard failure" cases the spec explicitly forbids
regressing on. The remaining three span the two query categories where the
reranker is most likely to swing F1 either direction.

Usage
-----

    # Default: index pinned to platform monorepo (matches grep_vs_semantex_bench)
    python3 benchmarks/reranker_ablation.py --repo /path/to/repo

    # JSON output (for CI / scripted comparisons)
    python3 benchmarks/reranker_ablation.py --repo /path/to/repo --json

    # Use a custom semantex binary
    SEMANTEX_BIN=/path/to/semantex python3 benchmarks/reranker_ablation.py

Honest-reporting clause
-----------------------

If the with-reranker arm does NOT improve overall F1 by at least +5pp AND
maintain >= baseline F1 on the two Q21/Q25 hard-failure cases, the script
prints a "-F1: keep disabled" verdict and exits non-zero. The script does not
spin a story to hide a regression. This is a deliberate design choice from
the v0.3 spec's risk T7 mitigation.

Network / model download
------------------------

When SEMANTEX_RERANKER=on, the first invocation lazily downloads the
cross-encoder weights to ~/.fastembed_cache/ (typically a few hundred MB for
bge-reranker-v2-m3). The script does not perform the download itself; the
semantex daemon handles it via the fastembed crate. If the environment is
offline and no cached model exists, the with-reranker arm will fail and the
script will report it explicitly rather than silently degrading.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ------------------------------------------------------------------ config


REPO_DEFAULT = "/Users/tk/dev/platform"  # mirrors grep_vs_semantex_bench default
SEMANTEX_BIN_DEFAULT = (
    os.environ.get("SEMANTEX_BIN")
    or "/Users/tk/dev/qgrep/semantex/target/release/semantex"
)

NUM_RUNS = 3  # 1 warmup + 2 timed (median); reranker download dominates first call


# ------------------------------------------------------------------ queries

# Subset selected to (a) cover both Q21/Q25 hard failures the spec calls out,
# (b) span semantic and architectural categories, (c) keep total runtime under
# a couple minutes. The queries are lifted verbatim from
# benchmarks/grep_vs_semantex_bench.py so deltas can be cross-checked.


@dataclass
class Query:
    qid: str
    category: str
    description: str
    nl_query: str
    expected_files: list  # repo-relative paths (ground truth)


QUERIES: list[Query] = [
    Query(
        qid="Q9",
        category="semantic",
        description="retry with exponential backoff",
        nl_query="retry with exponential backoff and jitter on transient errors",
        expected_files=[
            "apps/medparse/api/utils/retry.js",
            "apps/medparse/tests/utils/retry.test.js",
        ],
    ),
    Query(
        qid="Q12",
        category="semantic",
        description="factory pattern for services",
        nl_query="factory returns service based on connection type switch",
        expected_files=[
            "apps/api/src/services/connections/connectionServiceFactory.ts",
            "apps/api/src/services/connections/connectionService.ts",
        ],
    ),
    Query(
        qid="Q20",
        category="semantic",
        description="parallel failure handling (hard failure Q21 in RESEARCH doc)",
        nl_query="parallel async API calls that handle individual failures gracefully",
        expected_files=[
            "apps/api/src/services/connections/vitalConnectionService.ts",
            "apps/epic/src/services/epicFhir/client.ts",
        ],
    ),
    Query(
        qid="Q25",
        category="architectural",
        description="connection lifecycle (hard failure Q25 in RESEARCH doc)",
        nl_query="connection lifecycle management",
        expected_files=[
            "apps/api/src/services/connections/connectionService.ts",
            "apps/api/src/services/connections/vitalConnectionService.ts",
            "apps/api/src/routes/v1/connections.ts",
        ],
    ),
    Query(
        qid="Q29",
        category="architectural",
        description="BigQuery schema discovery and dynamic querying",
        nl_query="how health data schema is discovered and dynamic BigQuery queries are constructed",
        expected_files=[
            "apps/api/src/repos/query/QueryBuilder.ts",
            "apps/api/src/repos/metadata/TableMetadataRegistry.ts",
            "apps/api/src/repos/healthRepositoryBQ.ts",
            "apps/api/src/schemas/healthQuerySchemas.ts",
        ],
    ),
]

# Queries we will NOT allow to regress (spec: "zero regression on Q21/Q25").
HARD_FAILURE_QIDS = {"Q20", "Q25"}

# Spec-mandated minimum overall F1 lift to flip the default to enabled.
F1_LIFT_THRESHOLD = 0.05  # 5 percentage points


# ------------------------------------------------------------------ results


@dataclass
class ArmResult:
    """One arm (on or off) of one query."""

    median_ms: float
    files: set
    precision: float
    recall: float
    f1: float
    raw_stderr: str = ""
    error: Optional[str] = None  # populated if the run failed


@dataclass
class QueryResult:
    query: Query
    off: ArmResult
    on: ArmResult

    @property
    def f1_delta(self) -> float:
        return self.on.f1 - self.off.f1


@dataclass
class Verdict:
    overall_off_f1: float
    overall_on_f1: float
    overall_delta: float
    meets_threshold: bool
    hard_failure_regressed: list = field(default_factory=list)
    enable_by_default: bool = False
    summary: str = ""


# ------------------------------------------------------------------ helpers


def normalize_path(file_path: str, repo: str) -> str:
    """Make a returned path relative to the repo root."""
    if os.path.isabs(file_path):
        try:
            return os.path.relpath(file_path, repo)
        except ValueError:
            return file_path
    return file_path


def compute_prf(found: set, truth: set) -> tuple:
    """Standard P/R/F1 at file level."""
    if not truth and not found:
        return 1.0, 1.0, 1.0
    tp = len(found & truth)
    fp = len(found - truth)
    fn = len(truth - found)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def run_one_arm(
    semantex_bin: str,
    repo: str,
    query: Query,
    reranker_on: bool,
) -> ArmResult:
    """Invoke the semantex CLI with the env var flipped to on or off.

    We use --json so we can parse files reliably, and pass --rerank so the
    reranker pipeline stage actually runs (the env var inside the daemon then
    decides whether to load the model or be an identity pass-through).
    """
    env = os.environ.copy()
    env["SEMANTEX_RERANKER"] = "on" if reranker_on else "off"
    # We deliberately do NOT set SEMANTEX_RERANKER_MODEL — let the default
    # (bge-reranker-v2-m3) be exercised, matching the spec's E1 default.

    # Note: `-p PATH` is required because semantex's positional args are queries,
    # not the project path. Without `-p` the binary defaults to cwd and the
    # daemon-port lookup fails, producing empty/wrong results.
    cmd = [semantex_bin, "--json", "--rerank", "-p", repo, query.nl_query]

    times = []
    last_stdout = b""
    last_stderr = b""
    err_msg: Optional[str] = None

    for i in range(NUM_RUNS):
        start = time.perf_counter()
        try:
            result = subprocess.run(
                cmd, capture_output=True, env=env, timeout=300
            )
        except subprocess.TimeoutExpired as e:
            err_msg = f"timeout: {e}"
            break
        except FileNotFoundError as e:
            err_msg = (
                f"semantex binary not found at {semantex_bin}: {e}. "
                f"Set SEMANTEX_BIN or run `cargo build --release -p semantex-cli`."
            )
            break

        elapsed = (time.perf_counter() - start) * 1000
        last_stdout = result.stdout
        last_stderr = result.stderr
        if result.returncode != 0:
            err_msg = (
                f"semantex exited with code {result.returncode}: "
                f"{last_stderr.decode('utf-8', errors='replace')[:500]}"
            )
            break
        if i > 0:  # skip warmup iteration
            times.append(elapsed)

    files: set = set()
    if not err_msg:
        try:
            results_json = json.loads(last_stdout)
            if isinstance(results_json, list):
                for r in results_json:
                    fpath = r.get("file") or r.get("file_path") or ""
                    if fpath:
                        files.add(normalize_path(fpath, repo))
        except (json.JSONDecodeError, TypeError) as e:
            err_msg = f"could not parse semantex JSON output: {e}"

    truth = set(query.expected_files)
    precision, recall, f1 = compute_prf(files, truth)
    median_ms = statistics.median(times) if times else 0.0

    return ArmResult(
        median_ms=median_ms,
        files=files,
        precision=precision,
        recall=recall,
        f1=f1,
        raw_stderr=last_stderr.decode("utf-8", errors="replace")[:500],
        error=err_msg,
    )


def run_ablation(semantex_bin: str, repo: str) -> list[QueryResult]:
    out: list[QueryResult] = []
    for q in QUERIES:
        print(
            f"[{q.qid}] {q.description} (category={q.category})",
            file=sys.stderr,
            flush=True,
        )
        print("  arm=off ...", file=sys.stderr, flush=True)
        off = run_one_arm(semantex_bin, repo, q, reranker_on=False)
        print(
            f"    F1={off.f1:.3f}  P={off.precision:.3f}  R={off.recall:.3f}  "
            f"ms={off.median_ms:.0f}"
            + (f"  ERR: {off.error}" if off.error else ""),
            file=sys.stderr,
            flush=True,
        )
        print("  arm=on  ...", file=sys.stderr, flush=True)
        on = run_one_arm(semantex_bin, repo, q, reranker_on=True)
        print(
            f"    F1={on.f1:.3f}  P={on.precision:.3f}  R={on.recall:.3f}  "
            f"ms={on.median_ms:.0f}"
            + (f"  ERR: {on.error}" if on.error else ""),
            file=sys.stderr,
            flush=True,
        )
        out.append(QueryResult(query=q, off=off, on=on))
    return out


def compute_verdict(results: list[QueryResult]) -> Verdict:
    valid = [r for r in results if not r.off.error and not r.on.error]
    if not valid:
        return Verdict(
            overall_off_f1=0.0,
            overall_on_f1=0.0,
            overall_delta=0.0,
            meets_threshold=False,
            hard_failure_regressed=[],
            enable_by_default=False,
            summary="ALL ARMS FAILED — see per-query errors above. Cannot compute verdict.",
        )

    overall_off = statistics.mean(r.off.f1 for r in valid)
    overall_on = statistics.mean(r.on.f1 for r in valid)
    delta = overall_on - overall_off
    meets = delta >= F1_LIFT_THRESHOLD

    regressed = [
        r.query.qid
        for r in valid
        if r.query.qid in HARD_FAILURE_QIDS and r.on.f1 < r.off.f1
    ]

    enable = meets and not regressed

    if enable:
        summary = (
            f"+F1: enable by default. Overall F1 lift +{delta:.3f} "
            f"({overall_off:.3f} -> {overall_on:.3f}), meets {F1_LIFT_THRESHOLD:+.2f} "
            f"threshold and no regression on hard-failure queries."
        )
    elif regressed:
        summary = (
            f"-F1: keep disabled. Regression on hard-failure queries: "
            f"{', '.join(regressed)}. Overall F1 delta = {delta:+.3f}."
        )
    elif not meets:
        summary = (
            f"-F1: keep disabled. Overall F1 lift +{delta:.3f} "
            f"({overall_off:.3f} -> {overall_on:.3f}) does not clear the "
            f"{F1_LIFT_THRESHOLD:+.2f} threshold."
        )
    else:
        summary = (
            f"Indeterminate: meets={meets}, regressed={regressed}. "
            f"Defaulting to keep-disabled per spec risk T7."
        )

    return Verdict(
        overall_off_f1=overall_off,
        overall_on_f1=overall_on,
        overall_delta=delta,
        meets_threshold=meets,
        hard_failure_regressed=regressed,
        enable_by_default=enable,
        summary=summary,
    )


# ------------------------------------------------------------------ output


def render_markdown(
    results: list[QueryResult], verdict: Verdict, repo: str, semantex_bin: str
) -> str:
    lines = []
    lines.append("# Reranker Ablation (E1 / W3)")
    lines.append("")
    lines.append(f"- Repo: `{repo}`")
    lines.append(f"- Binary: `{semantex_bin}`")
    lines.append(f"- Runs per arm: {NUM_RUNS} (first discarded as warmup, median of rest)")
    lines.append(
        f"- F1 lift threshold to enable by default: "
        f"{F1_LIFT_THRESHOLD:+.2f} (overall, all queries) AND zero "
        f"regression on hard-failure queries ({', '.join(sorted(HARD_FAILURE_QIDS))})"
    )
    lines.append("")
    lines.append("## Per-query results")
    lines.append("")
    lines.append(
        "| QID | Category | F1 (off) | F1 (on) | Delta | P off | P on | R off | R on | ms off | ms on | Notes |"
    )
    lines.append(
        "|-----|----------|---------:|--------:|------:|------:|-----:|------:|-----:|-------:|------:|-------|"
    )
    for r in results:
        note_bits = []
        if r.off.error:
            note_bits.append(f"off ERR: {r.off.error[:80]}")
        if r.on.error:
            note_bits.append(f"on ERR: {r.on.error[:80]}")
        if r.query.qid in HARD_FAILURE_QIDS and not r.off.error and not r.on.error:
            if r.on.f1 < r.off.f1:
                note_bits.append("HARD-FAILURE REGRESSION")
            else:
                note_bits.append("hard-failure check OK")
        notes = "; ".join(note_bits) if note_bits else ""
        lines.append(
            f"| {r.query.qid} | {r.query.category} | "
            f"{r.off.f1:.3f} | {r.on.f1:.3f} | {r.f1_delta:+.3f} | "
            f"{r.off.precision:.3f} | {r.on.precision:.3f} | "
            f"{r.off.recall:.3f} | {r.on.recall:.3f} | "
            f"{r.off.median_ms:.0f} | {r.on.median_ms:.0f} | {notes} |"
        )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- Overall F1 (off): **{verdict.overall_off_f1:.3f}**")
    lines.append(f"- Overall F1 (on): **{verdict.overall_on_f1:.3f}**")
    lines.append(f"- Overall delta: **{verdict.overall_delta:+.3f}**")
    lines.append(f"- Meets {F1_LIFT_THRESHOLD:+.2f} lift threshold: {verdict.meets_threshold}")
    lines.append(
        f"- Hard-failure regressions: "
        f"{', '.join(verdict.hard_failure_regressed) if verdict.hard_failure_regressed else 'none'}"
    )
    lines.append(f"- **Recommendation:** {verdict.summary}")
    lines.append("")
    return "\n".join(lines)


def render_json(
    results: list[QueryResult], verdict: Verdict, repo: str, semantex_bin: str
) -> str:
    payload = {
        "repo": repo,
        "semantex_bin": semantex_bin,
        "runs_per_arm": NUM_RUNS,
        "f1_lift_threshold": F1_LIFT_THRESHOLD,
        "hard_failure_qids": sorted(HARD_FAILURE_QIDS),
        "queries": [
            {
                "qid": r.query.qid,
                "category": r.query.category,
                "description": r.query.description,
                "nl_query": r.query.nl_query,
                "expected_files": r.query.expected_files,
                "off": {
                    "f1": r.off.f1,
                    "precision": r.off.precision,
                    "recall": r.off.recall,
                    "median_ms": r.off.median_ms,
                    "files": sorted(r.off.files),
                    "error": r.off.error,
                },
                "on": {
                    "f1": r.on.f1,
                    "precision": r.on.precision,
                    "recall": r.on.recall,
                    "median_ms": r.on.median_ms,
                    "files": sorted(r.on.files),
                    "error": r.on.error,
                },
                "delta": r.f1_delta,
            }
            for r in results
        ],
        "verdict": {
            "overall_off_f1": verdict.overall_off_f1,
            "overall_on_f1": verdict.overall_on_f1,
            "overall_delta": verdict.overall_delta,
            "meets_threshold": verdict.meets_threshold,
            "hard_failure_regressed": verdict.hard_failure_regressed,
            "enable_by_default": verdict.enable_by_default,
            "summary": verdict.summary,
        },
    }
    return json.dumps(payload, indent=2)


# ------------------------------------------------------------------ main


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo",
        default=REPO_DEFAULT,
        help=f"Path to the indexed repo (default: {REPO_DEFAULT})",
    )
    parser.add_argument(
        "--semantex-bin",
        default=SEMANTEX_BIN_DEFAULT,
        help=f"Path to semantex binary (default: {SEMANTEX_BIN_DEFAULT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write rendered output (markdown or JSON) to this file in addition to stdout",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of markdown (for CI/scripted comparison)",
    )
    parser.add_argument(
        "--exit-zero",
        action="store_true",
        help=(
            "Always exit 0, even when the verdict is 'keep disabled'. "
            "Without this flag, the script exits 1 to flag the ablation "
            "as not-yet-shipping-ready."
        ),
    )
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    if not os.path.isdir(repo):
        print(f"ERROR: repo dir does not exist: {repo}", file=sys.stderr)
        return 2

    if not os.path.isfile(args.semantex_bin):
        print(
            f"ERROR: semantex binary not found: {args.semantex_bin}\n"
            f"Hint: run `cargo build --release -p semantex-cli` or set "
            f"SEMANTEX_BIN.",
            file=sys.stderr,
        )
        return 2

    print(
        f"Running reranker ablation: {len(QUERIES)} queries, {NUM_RUNS} runs/arm "
        f"({NUM_RUNS * len(QUERIES) * 2} total semantex invocations)",
        file=sys.stderr,
    )
    print(f"  Repo: {repo}", file=sys.stderr)
    print(f"  Binary: {args.semantex_bin}", file=sys.stderr)
    print("", file=sys.stderr)

    results = run_ablation(args.semantex_bin, repo)
    verdict = compute_verdict(results)

    rendered = (
        render_json(results, verdict, repo, args.semantex_bin)
        if args.json
        else render_markdown(results, verdict, repo, args.semantex_bin)
    )

    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"\nWrote: {args.output}", file=sys.stderr)

    # Exit code policy: 0 if the reranker is ready to flip on by default,
    # 1 otherwise. CI invocations can pass --exit-zero to suppress.
    if args.exit_zero or verdict.enable_by_default:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
