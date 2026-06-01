"""Acceptance gate: reproduce a published baseline within a stated tolerance.

Runs the configured ablation/dataset, computes the target metric, and exits
non-zero if |measured - expected| > tolerance. Proves the protocol is correct
before any tuning relies on the harness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
SWE_BENCH_SRC = ROOT.parent / "swe_bench" / "src"
if SWE_BENCH_SRC.is_dir():
    sys.path.insert(0, str(SWE_BENCH_SRC))

import os

from relevance_harness.datasets.csn import load_csn_corpus
from relevance_harness.metrics import mrr_at_k
from relevance_harness.runner import run_corpus

CONFIG = ROOT / "config"


@click.command()
@click.option("--baseline", default="csn_bm25", show_default=True)
@click.option("--semantex-bin", default=os.environ.get("SEMANTEX_BINARY", "semantex"))
def main(baseline: str, semantex_bin: str):
    baselines = yaml.safe_load((CONFIG / "baselines.yaml").read_text())
    b = baselines[baseline]
    csn_cfg = yaml.safe_load((CONFIG / "csn_subset.yaml").read_text())

    corpus = load_csn_corpus(
        language=b["language"],
        corpus_dir=ROOT / "results" / "_baseline" / f"csn_{b['language']}",
        dataset_id=csn_cfg["dataset_id"],
        corpus_size=csn_cfg["corpus_size"],
        query_size=csn_cfg["query_size"],
        seed=csn_cfg["seed"],
        trust_remote_code=csn_cfg.get("trust_remote_code", True),
    )
    out = run_corpus(corpus, ablation=b["ablation"], k=10, semantex_binary=semantex_bin,
                     match_mode="doc_id")
    measured = mrr_at_k(out.relevances, k=10)
    expected = float(b["expected_mrr_at_10"])
    tol = float(b["tolerance"])
    delta = abs(measured - expected)

    click.echo(f"baseline={baseline} measured_mrr@10={measured:.4f} "
               f"expected={expected:.4f} tol={tol:.4f} delta={delta:.4f}")
    click.echo(f"source: {b['source']}")
    if delta > tol:
        click.echo("FAIL: outside tolerance — protocol or wiring is wrong.", err=True)
        raise SystemExit(1)
    click.echo("PASS: within tolerance — protocol reproduces the baseline.")


if __name__ == "__main__":
    main()
