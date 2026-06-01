"""Main relevance entrypoint: load a dataset, run an ablation, write a report.

Examples:
  python -m scripts.run --dataset csn --ablation hybrid
  python -m scripts.run --dataset csn --ablation dense-only --embedder coderank-137m
  python -m scripts.run --dataset swe-loc --ablation hybrid
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import click
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
# make the sibling swe_bench harness importable for SWE-loc reuse
SWE_BENCH_SRC = ROOT.parent / "swe_bench" / "src"
if SWE_BENCH_SRC.is_dir():
    sys.path.insert(0, str(SWE_BENCH_SRC))

from relevance_harness.datasets.csn import load_csn_corpus
from relevance_harness.datasets.swe_loc import build_swe_loc_corpus, load_swe_loc_queries
from relevance_harness.report import (
    ReproStamp, compute_metrics_row, current_git_rev, render_report_json, render_report_md,
)
from relevance_harness.runner import run_corpus

CONFIG = ROOT / "config"
RESULTS = ROOT / "results"


def _phase_a_ids() -> set[str] | None:
    f = ROOT.parent / "swe_bench" / "config" / "instances_phase_a.txt"
    if not f.is_file():
        return None
    return {ln.strip() for ln in f.read_text().splitlines() if ln.strip()}


@click.command()
@click.option("--dataset", type=click.Choice(["csn", "coir", "swe-loc"]), required=True)
@click.option("--ablation", type=click.Choice(["sparse-only", "dense-only", "hybrid", "rerank"]),
              default="hybrid", show_default=True)
@click.option("--embedder", default="", help="lateon-colbert | coderank-137m "
              "(canonical SEMANTEX_EMBEDDER selector, integration §4 D-env-knob)")
@click.option("--k", default=10, type=int, show_default=True)
@click.option("--run-id", default="", help="reuse a results/<run-id> dir")
@click.option("--semantex-bin", default=os.environ.get("SEMANTEX_BINARY", "semantex"))
def main(dataset, ablation, embedder, k, run_id, semantex_bin):
    embedder = embedder or None
    if not run_id:
        run_id = time.strftime(f"%Y%m%d-%H%M%S-{dataset}-{ablation}")
    out_dir = RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_root = out_dir / "corpora"

    rows: list[dict] = []
    manifests: list[dict] = []

    if dataset == "csn":
        cfg = yaml.safe_load((CONFIG / "csn_subset.yaml").read_text())
        for lang in cfg["languages"]:
            corpus = load_csn_corpus(
                language=lang, corpus_dir=corpus_root / f"csn_{lang}",
                dataset_id=cfg["dataset_id"], corpus_size=cfg["corpus_size"],
                query_size=cfg["query_size"], seed=cfg["seed"],
                trust_remote_code=cfg.get("trust_remote_code", True),
            )
            out = run_corpus(corpus, ablation=ablation, k=k, semantex_binary=semantex_bin,
                             embedder=embedder, match_mode="doc_id")
            rows.append(compute_metrics_row(out, k=k))
            if corpus.manifest is not None:
                manifests.append(corpus.manifest.to_dict())
    elif dataset == "swe-loc":
        queries = load_swe_loc_queries(instance_ids=_phase_a_ids())
        agg_rel: list[list[int]] = []
        agg_nrel: list[int] = []
        for q in queries:
            try:
                corpus = build_swe_loc_corpus(
                    instance_id=q.query_id, query=q, semantex_binary=semantex_bin
                )
            except FileNotFoundError as e:
                click.echo(f"skip {q.query_id}: {e}", err=True)
                continue
            out = run_corpus(corpus, ablation=ablation, k=k, semantex_binary=semantex_bin,
                             embedder=embedder, match_mode="file")
            agg_rel.extend(out.relevances)
            agg_nrel.extend(out.n_relevant)
        from relevance_harness.runner import RunOutput
        rows.append(compute_metrics_row(
            RunOutput(corpus_name="swe-loc", ablation=ablation,
                      relevances=agg_rel, n_relevant=agg_nrel, per_query=[]),
            k=k,
        ))
    else:  # coir
        click.echo("CoIR run: fill config/coir_subset.yaml with real HF ids "
                   "(research notes Task 0.1 Step 4), then wire load_coir_subdataset here.",
                   err=True)
        raise SystemExit(2)

    stamp = ReproStamp(
        git_rev=current_git_rev(),
        # ReproStamp.dense_backend records the active embedder selector; the
        # canonical env var is SEMANTEX_EMBEDDER (integration §4 D-env-knob).
        dense_backend=embedder
        or os.environ.get("SEMANTEX_EMBEDDER")
        or os.environ.get("SEMANTEX_DENSE_BACKEND", "default"),
        model_id=os.environ.get("SEMANTEX_LLM_MODEL", "n/a-dense-path"),
        k=k,
    )
    (out_dir / "report.json").write_text(render_report_json(rows=rows, stamp=stamp, manifests=manifests))
    (out_dir / "report.md").write_text(render_report_md(rows=rows, stamp=stamp, manifests=manifests))
    click.echo(f"Report: {out_dir / 'report.md'}")
    click.echo((out_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
