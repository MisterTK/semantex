#!/usr/bin/env python3
"""
Public reproducer entry point — semantex v0.3 benchmark suite (B1–B5).

This is the single command the spec promises:

    python3 benchmarks/run_public.py --tool=all --benchmark=all --output=results/

It orchestrates per-sub-benchmark modules in `benchmarks/v0_3/` and writes
all artefacts under `--output`. Each sub-benchmark module is independently
testable; this orchestrator only wires arguments and aggregates summaries.

Design constraints (per W8 spec):
  * No tool gets defaults that hardcode local paths. Everything flows through
    `--repo`, `--semantex-bin`, or env vars.
  * If a competitor tool is not installed, SKIP with a stderr note rather
    than failing the whole sweep.
  * Output structure:
        <output>/
          run_metadata.json
          b1/...
          b2/...
          b3/...
          b4/...
          b5/...
          SUMMARY.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))  # let `from v0_3 import ...` work

from v0_3 import b1_public_retrieval  # noqa: E402
from v0_3 import b2_nl2code  # noqa: E402
from v0_3 import b3_agent_ccb  # noqa: E402
from v0_3 import b4_latency  # noqa: E402
from v0_3 import b5_head_to_head  # noqa: E402
from v0_3 import competitor_registry  # noqa: E402

BENCHMARKS = ("b1", "b2", "b3", "b4", "b5", "all")
DEFAULT_SEMANTEX_BIN = os.environ.get(
    "SEMANTEX_BIN",
    "/usr/local/bin/semantex",  # last-resort install location
)


def parse_args(argv):
    ap = argparse.ArgumentParser(
        description="Public reproducer for the semantex v0.3 benchmark suite.",
    )
    ap.add_argument(
        "--tool",
        default="semantex",
        help="Comma-separated tool slugs, or 'all'. Default: semantex. "
        "See COMPETITORS.md / `python3 benchmarks/v0_3/competitor_registry.py` "
        "for available slugs.",
    )
    ap.add_argument(
        "--benchmark",
        choices=BENCHMARKS,
        default="all",
        help="Which sub-benchmark to run. Default: all.",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output directory for results and intermediate artefacts.",
    )
    ap.add_argument(
        "--target-repo",
        default=None,
        help="Repo path used by B4 (latency) and B2 fallback resolver. "
        "If omitted, the script uses the current working directory.",
    )
    ap.add_argument(
        "--target-repos",
        nargs="*",
        default=None,
        help="Repos used by B3 (agent CCB). One per language ideally. "
        "Empty → B3 runs in dry-run mode.",
    )
    ap.add_argument(
        "--semantex-bin",
        default=DEFAULT_SEMANTEX_BIN,
        help=f"Path to semantex binary (default: {DEFAULT_SEMANTEX_BIN}; "
        f"or set SEMANTEX_BIN env).",
    )
    ap.add_argument(
        "--datasets-root",
        default=str(THIS_DIR / "datasets"),
        help="Root path containing B1 dataset subdirectories.",
    )
    ap.add_argument(
        "--dry-run-b3",
        action="store_true",
        help="Force B3 into dry-run mode regardless of --target-repos.",
    )
    return ap.parse_args(argv)


def resolve_tools(spec: str) -> list:
    if spec == "all":
        slugs = ["semantex"] + competitor_registry.list_ids()
    else:
        slugs = [s.strip() for s in spec.split(",") if s.strip()]
    return slugs


def detect_tool(slug: str, semantex_bin: str) -> tuple:
    """(available: bool, status_string)."""
    if slug == "semantex":
        if not Path(semantex_bin).exists():
            return False, f"semantex binary not found at {semantex_bin}"
        return True, semantex_bin
    spec = competitor_registry.by_id(slug)
    if spec is None:
        return False, f"unknown tool slug '{slug}'"
    return spec.detect()


def main(argv=None) -> int:
    args = parse_args(argv)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    target_repo = Path(args.target_repo or os.getcwd()).resolve()
    target_repos_b3 = [str(Path(r).resolve()) for r in (args.target_repos or [])]

    tools = resolve_tools(args.tool)
    benches = [args.benchmark] if args.benchmark != "all" else ["b1", "b2", "b3", "b4", "b5"]

    run_metadata = {
        "started_at": int(time.time()),
        "args": vars(args),
        "target_repo": str(target_repo),
        "target_repos_b3": target_repos_b3,
        "tools_requested": tools,
        "benchmarks": benches,
        "semantex_bin": args.semantex_bin,
    }
    (output / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))

    summary: dict = {"benchmarks": {}, "tools": {}}

    # Detect tools up-front so we skip cleanly.
    available: dict = {}
    for slug in tools:
        ok, msg = detect_tool(slug, args.semantex_bin)
        available[slug] = ok
        summary["tools"][slug] = {"available": ok, "status": msg}
        if not ok:
            print(f"  [skip] tool '{slug}' not available: {msg}", file=sys.stderr)

    # B1
    if "b1" in benches:
        b1_dir = output / "b1"
        for slug in tools:
            if not available[slug]:
                continue
            result = b1_public_retrieval.run(
                tool=slug,
                output_dir=b1_dir,
                datasets_root=Path(args.datasets_root).resolve(),
            )
            summary["benchmarks"].setdefault("b1", {})[slug] = {
                "ok": result.get("ok"),
                "reason": result.get("reason"),
            }
            print(
                f"[b1] {slug}: {'ok' if result.get('ok') else 'pending'} "
                f"— {result.get('reason') or ''}",
                file=sys.stderr,
            )

    # B2
    if "b2" in benches:
        b2_dir = output / "b2"
        for slug in tools:
            if not available[slug] or slug != "semantex":
                # B2 currently has runners only for semantex; other-tool wiring
                # is deferred to the human at sweep time.
                if slug != "semantex":
                    summary["benchmarks"].setdefault("b2", {})[slug] = {
                        "ok": False, "reason": "no runner for non-semantex tool yet",
                    }
                continue
            result = b2_nl2code.run(
                tool=slug,
                output_dir=b2_dir,
                semantex_bin=args.semantex_bin,
                repo_resolver=None,  # human plugs this in at sweep time
            )
            summary["benchmarks"].setdefault("b2", {})[slug] = {
                "ok": result.get("ok"),
                "n_evaluated": result.get("n_evaluated", 0),
                "n_needs_curation": result.get("n_needs_curation", 0),
                "mean_f1": result.get("mean_f1"),
            }
            print(
                f"[b2] {slug}: evaluated={result.get('n_evaluated', 0)} "
                f"needs_curation={result.get('n_needs_curation', 0)}",
                file=sys.stderr,
            )

    # B3 — heavy. Default to dry-run if no repos supplied or --dry-run-b3 set.
    if "b3" in benches:
        b3_dir = output / "b3"
        dry = args.dry_run_b3 or not target_repos_b3
        for slug in tools:
            if not available[slug]:
                continue
            result = b3_agent_ccb.run(
                tool=slug,
                output_dir=b3_dir,
                repos=target_repos_b3 or [str(target_repo)],
                dry_run=dry,
            )
            summary["benchmarks"].setdefault("b3", {})[slug] = {
                "ok": result.get("ok"),
                "dry_run": result.get("dry_run", False),
                "reason": result.get("reason"),
            }
            print(
                f"[b3] {slug}: {'dry-run' if result.get('dry_run') else 'real'} "
                f"ok={result.get('ok')}",
                file=sys.stderr,
            )

    # B4
    if "b4" in benches:
        b4_dir = output / "b4"
        for slug in tools:
            if not available[slug]:
                continue
            result = b4_latency.run(
                tool=slug,
                target_repo=str(target_repo),
                output_dir=b4_dir,
                semantex_bin=args.semantex_bin,
            )
            summary["benchmarks"].setdefault("b4", {})[slug] = {
                "ok": result.get("ok") if not result.get("skipped") else None,
                "skipped": result.get("skipped", False),
                "summary": result.get("summary"),
                "reason": result.get("reason"),
            }
            print(
                f"[b4] {slug}: {'skipped' if result.get('skipped') else result.get('ok')} "
                f"summary={result.get('summary')}",
                file=sys.stderr,
            )

    # B5 — aggregate-only, runs after B3
    if "b5" in benches:
        b5_dir = output / "b5"
        result = b5_head_to_head.run(
            output_dir=b5_dir,
            b3_results_dir=output / "b3",
        )
        summary["benchmarks"]["b5"] = result

    summary["finished_at"] = int(time.time())
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {output / 'SUMMARY.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
