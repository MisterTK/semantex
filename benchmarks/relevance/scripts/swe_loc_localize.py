"""SWE-bench-Verified file-level localisation harness — the externally
reproducible retrieval benchmark this repo's other claims should be measured
against. Given an issue's title+body, rank the files the gold patch touched;
score with the same Acc@k / MRR protocol SweRank and LocAgent report, so the
numbers are directly comparable to published ones (see README.md).

Four arms per instance: `semantex search` (hybrid), `semantex search
--sparse-only` (BM25 baseline), `semantex agent` auto-routed search (no
forced route — the engine's own keyword classifier picks the mechanism), and
an external ripgrep keyword baseline (no semantex at all).

Fully offline after setup: instances must already be checked out + indexed
under $SWE_BENCH_REPO_CACHE (run `benchmarks/swe_bench/scripts/pre_index.py`
first), OR pass `--local-fixture` to run entirely against the tiny synthetic
fixture (no network, no real git history). No LLM calls either way — see the
swe_loc_runner module docstring. Deterministic: instances are always
processed in sorted instance_id order, so `--limit N` always picks the same
N instances.

Usage:
  # offline smoke run against the tiny synthetic fixture: point
  # SWE_BENCH_REPO_CACHE at a dir holding <instance_id>/ == a copy of
  # fixtures/tiny_corpus (see tests/test_swe_loc_localize.py for the exact
  # setup) so there's a real, tiny, local-only repo to index + search.
  export SWE_BENCH_REPO_CACHE=/tmp/tiny_swe_loc_cache
  mkdir -p $SWE_BENCH_REPO_CACHE/tiny__tiny-1
  cp fixtures/tiny_corpus/* $SWE_BENCH_REPO_CACHE/tiny__tiny-1/
  python -m scripts.swe_loc_localize --local-fixture fixtures/tiny_swe_loc_instance.json

  # real SWE-bench-Verified Phase-A instances (after pre_index.py --phase a)
  python -m scripts.swe_loc_localize --limit 3
  python -m scripts.swe_loc_localize   # full Phase-A set
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import click

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
# make the sibling swe_bench harness importable for SWE-loc reuse
SWE_BENCH_SRC = ROOT.parent / "swe_bench" / "src"
if SWE_BENCH_SRC.is_dir():
    sys.path.insert(0, str(SWE_BENCH_SRC))

from relevance_harness.datasets.swe_loc import load_swe_loc_queries
from relevance_harness.metrics import acc_at_k, mrr_at_k
from relevance_harness.report import (
    ReproStamp, current_git_rev, render_report_json, render_report_md,
)
from relevance_harness.swe_loc_runner import ARMS, relevance_vector, run_instance

RESULTS = ROOT / "results"


def _repo_cache_dir() -> Path:
    return Path(os.environ.get("SWE_BENCH_REPO_CACHE", Path.home() / ".swe_bench_repos"))


def _phase_a_ids() -> set[str] | None:
    f = ROOT.parent / "swe_bench" / "config" / "instances_phase_a.txt"
    if not f.is_file():
        return None
    return {ln.strip() for ln in f.read_text().splitlines() if ln.strip()}


def compute_arm_rows(
    relevances: dict[str, list[list[int]]],
    tokens: dict[str, list[int]],
    errors: dict[str, int],
) -> list[dict]:
    """One report row per arm: Acc@{1,5,10}, MRR@10, avg tokens-returned."""
    rows = []
    for arm in ARMS:
        rels = relevances[arm]
        n = len(rels)
        avg_tokens = (sum(tokens[arm]) / n) if n else 0.0
        rows.append({
            "dataset": "swe-loc-verified",
            "arm": arm,
            "n_queries": n,
            "n_errors": errors[arm],
            "acc_at_1": round(acc_at_k(rels, k=1), 4),
            "acc_at_5": round(acc_at_k(rels, k=5), 4),
            "acc_at_10": round(acc_at_k(rels, k=10), 4),
            "mrr_at_10": round(mrr_at_k(rels, k=10), 4),
            "avg_tokens_returned": round(avg_tokens, 1),
        })
    return rows


@click.command()
@click.option("--local-fixture", type=click.Path(path_type=Path), default=None,
              help="Local SWE-bench-Verified-shaped JSON (skips the HF download). "
                   "Use fixtures/sample_verified_subset.json for the offline demo.")
@click.option("--limit", type=int, default=None,
              help="Only run the first N instances, sorted by instance_id "
                   "(deterministic — the same N every run).")
@click.option("--k", default=10, type=int, show_default=True)
@click.option("--run-id", default="", help="reuse a results/<run-id> dir")
@click.option("--semantex-bin", default=os.environ.get("SEMANTEX_BINARY", "semantex"))
def main(local_fixture, limit, k, run_id, semantex_bin):
    instance_ids = _phase_a_ids() if local_fixture is None else None
    queries = load_swe_loc_queries(local_path=local_fixture, instance_ids=instance_ids)
    queries = sorted(queries, key=lambda q: q.query_id)
    if limit is not None:
        queries = queries[:limit]
    if not queries:
        click.echo("No queries to run (empty dataset / --limit 0).", err=True)
        raise SystemExit(2)

    if not run_id:
        tag = "fixture" if local_fixture else "verified"
        run_id = time.strftime(f"%Y%m%d-%H%M%S-swe-loc-{tag}")
    out_dir = RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    relevances: dict[str, list[list[int]]] = {a: [] for a in ARMS}
    tokens: dict[str, list[int]] = {a: [] for a in ARMS}
    errors: dict[str, int] = {a: 0 for a in ARMS}
    per_instance: list[dict] = []

    cache_dir = _repo_cache_dir()
    n_skipped = 0
    for q in queries:
        corpus_dir = cache_dir / q.query_id
        if not corpus_dir.is_dir():
            click.echo(
                f"skip {q.query_id}: repo not found at {corpus_dir} "
                f"(run benchmarks/swe_bench/scripts/pre_index.py first)",
                err=True,
            )
            n_skipped += 1
            continue

        gold = frozenset(q.gold_doc_ids)
        arm_results = run_instance(q, corpus_dir=corpus_dir, semantex_binary=semantex_bin, k=k)
        record = {"instance_id": q.query_id, "gold": sorted(gold), "arms": {}}
        summary = []
        for ar in arm_results:
            rels = relevance_vector(ar.ranked_files, gold)
            relevances[ar.arm].append(rels)
            tokens[ar.arm].append(ar.tokens_returned)
            if ar.error:
                errors[ar.arm] += 1
            record["arms"][ar.arm] = {
                "ranked_files": list(ar.ranked_files),
                "tokens_returned": ar.tokens_returned,
                "error": ar.error,
            }
            summary.append(f"{ar.arm}={'hit' if any(rels) else ('err' if ar.error else 'miss')}")
        per_instance.append(record)
        click.echo(f"[{len(per_instance)}/{len(queries)}] {q.query_id}: " + ", ".join(summary))

    rows = compute_arm_rows(relevances, tokens, errors)
    stamp = ReproStamp(
        git_rev=current_git_rev(),
        dense_backend=os.environ.get("SEMANTEX_EMBEDDER")
        or os.environ.get("SEMANTEX_DENSE_BACKEND", "default"),
        model_id="n/a-retrieval-only",
        k=k,
    )
    manifests = [{
        "dataset": "swe-loc-verified",
        "selected": len(per_instance),
        "total": len(per_instance) + n_skipped,
        "seed": "deterministic-sorted-instance_id",
        "dropped_ids": [],
    }] if n_skipped else []

    (out_dir / "report.json").write_text(
        render_report_json(rows=rows, stamp=stamp, manifests=manifests))
    (out_dir / "report.md").write_text(
        render_report_md(rows=rows, stamp=stamp, manifests=manifests))
    (out_dir / "per_instance.json").write_text(json.dumps(per_instance, indent=2))
    click.echo(f"\nReport: {out_dir / 'report.md'}")
    click.echo((out_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
