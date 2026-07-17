#!/usr/bin/env python3
"""Combines Track A (benchmarks/relevance) + Track B (claude_bench.py) output
into one quality + query-latency report comparing the lateon-colbert and
coderank-137m dense-backend embedders. See
docs/superpowers/specs/2026-07-16-lateon-coderank-quality-headtohead-design.md
for the methodology this implements.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

DATASETS = ("coir", "csn")
EMBEDDERS = ("lateon-colbert", "coderank-137m")
SX_ARMS = ("sx-lateon", "sx-coderank")


def load_track_a_cell(run_dir: Path) -> list[dict]:
    """One dict per (dataset, ablation) metrics row under `run_dir`, merged
    with its query-latency instrumentation. scripts/run.py appends to `rows`
    and calls `_capture` in the same loop iteration for CSN's per-language
    runs (and once for coir-codetrans-dl), so pairing them positionally is
    safe -- there is no shared label to join on across all dataset code
    paths, so don't try to match by name instead."""
    report = json.loads((run_dir / "report.json").read_text())
    instr_files = sorted(run_dir.glob("instr-*.json"))
    instr = json.loads(instr_files[0].read_text()) if instr_files else []
    cells = []
    for i, row in enumerate(report["rows"]):
        cell = dict(row)
        if i < len(instr):
            cell["cold_latency_ms"] = instr[i].get("cold_latency_ms")
            cell["warm_latency_ms"] = instr[i].get("warm_latency_ms")
        else:
            cell["cold_latency_ms"] = None
            cell["warm_latency_ms"] = None
        cells.append(cell)
    return cells


def track_a_table(track_a_dir: Path) -> list[dict]:
    """Reads <track_a_dir>/<dataset>/<embedder>/ for every (dataset,
    embedder) pair in DATASETS x EMBEDDERS, skipping any not run yet."""
    rows = []
    for dataset in DATASETS:
        for embedder in EMBEDDERS:
            run_dir = track_a_dir / dataset / embedder
            if not (run_dir / "report.json").exists():
                continue
            for cell in load_track_a_cell(run_dir):
                rows.append({"embedder": embedder, **cell})
    return rows


def track_b_table(all_results: list[dict]) -> list[dict]:
    """Mean +/- stdev quality score per (repo, question_type, arm),
    restricted to the sx-lateon/sx-coderank arms and dropping
    unscored/errored cells."""
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for r in all_results:
        if r.get("arm") not in SX_ARMS or r.get("quality") is None:
            continue
        key = (Path(r["repo"]).name, r["question_type"], r["arm"])
        grouped.setdefault(key, []).append(r["quality"])
    rows = []
    for (repo, qtype, arm), scores in sorted(grouped.items()):
        rows.append({
            "repo": repo, "question_type": qtype, "arm": arm,
            "n": len(scores),
            "mean_quality": round(statistics.mean(scores), 2),
            "stdev": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0.0,
        })
    return rows


def flag_ambiguous_cells(track_b_rows: list[dict]) -> list[str]:
    """A (repo, question_type) cell is ambiguous if the two arms' mean +/-
    stdev bands overlap -- at that point 3 reps can't call a winner (spec's
    Track B synthesis requirement: flag it, don't force a verdict)."""
    by_cell: dict[tuple[str, str], dict[str, dict]] = {}
    for row in track_b_rows:
        by_cell.setdefault((row["repo"], row["question_type"]), {})[row["arm"]] = row
    flags = []
    for (repo, qtype), by_arm in sorted(by_cell.items()):
        if not all(arm in by_arm for arm in SX_ARMS):
            continue
        a, b = by_arm["sx-lateon"], by_arm["sx-coderank"]
        lo_a, hi_a = a["mean_quality"] - a["stdev"], a["mean_quality"] + a["stdev"]
        lo_b, hi_b = b["mean_quality"] - b["stdev"], b["mean_quality"] + b["stdev"]
        if lo_a <= hi_b and lo_b <= hi_a:
            flags.append(
                f"{repo}/{qtype}: ambiguous (sx-lateon {a['mean_quality']}±{a['stdev']} "
                f"vs sx-coderank {b['mean_quality']}±{b['stdev']}, n={a['n']})"
            )
    return flags


def render_report_md(track_a_rows: list[dict], track_b_rows: list[dict],
                      ambiguous: list[str]) -> str:
    lines = [
        "# next-plaid (lateon-colbert) vs coderank-hnsw (coderank-137m) "
        "— quality head-to-head\n",
        "## Track A — academic (CoIR + CSN)\n",
        "| embedder | dataset | n_queries | mrr@10 | ndcg@10 | recall@1 | "
        "recall@5 | recall@10 | map | cold ms | warm ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in track_a_rows:
        lines.append(
            f"| {r['embedder']} | {r['dataset']} | {r['n_queries']} | "
            f"{r['mrr_at_10']:.4f} | {r['ndcg_at_10']:.4f} | "
            f"{r['recall_at_1']:.4f} | {r['recall_at_5']:.4f} | "
            f"{r['recall_at_10']:.4f} | {r['map']:.4f} | "
            f"{r['cold_latency_ms']} | {r['warm_latency_ms']} |"
        )

    lines += [
        "\n## Track B — real-world (claude_bench.py, 1-5 Claude-judged quality)\n",
        "| repo | question_type | arm | n | mean quality | stdev |",
        "|---|---|---|---|---|---|",
    ]
    for r in track_b_rows:
        lines.append(
            f"| {r['repo']} | {r['question_type']} | {r['arm']} | {r['n']} | "
            f"{r['mean_quality']} | {r['stdev']} |"
        )

    lines.append(
        "\n## Ambiguous cells (mean ± stdev bands overlap between arms "
        "— 3 reps can't call a winner here)\n"
    )
    lines += [f"- {f}" for f in ambiguous] if ambiguous else ["- none"]

    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track-a-dir", required=True, type=Path,
                    help="dir containing <dataset>/<embedder>/report.json "
                         "subdirs, e.g. benchmarks/relevance/results/2026-07-16-headtohead")
    p.add_argument("--track-b-results", required=True, type=Path,
                    help="path to claude_bench.py's all_results.json (post-judge)")
    p.add_argument("--output", required=True, type=Path,
                    help="path to write the combined report.md to")
    args = p.parse_args()

    track_a_rows = track_a_table(args.track_a_dir)
    all_results = json.loads(args.track_b_results.read_text())
    track_b_rows = track_b_table(all_results)
    ambiguous = flag_ambiguous_cells(track_b_rows)

    report = render_report_md(track_a_rows, track_b_rows, ambiguous)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
