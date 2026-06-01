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

from relevance_harness.datasets.coir import load_coir_subdataset
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
@click.option("--dataset", type=click.Choice(["csn", "coir", "coir-codetrans-dl", "swe-loc"]),
              required=True)
@click.option("--ablation", type=click.Choice(["sparse-only", "dense-only", "hybrid", "rerank"]),
              default="hybrid", show_default=True)
@click.option("--embedder", default="", help="lateon-colbert | coderank-137m "
              "(canonical SEMANTEX_EMBEDDER selector, integration §4 D-env-knob)")
@click.option("--k", default=10, type=int, show_default=True)
@click.option("--run-id", default="", help="reuse a results/<run-id> dir")
@click.option("--semantex-bin", default=os.environ.get("SEMANTEX_BINARY", "semantex"))
def main(dataset, ablation, embedder, k, run_id, semantex_bin):
    embedder = embedder or None
    # The embedder is authoritative at INDEX time and the dense backend is
    # persisted in the corpus's .semantex; share-one-dir would silently reuse the
    # first arm's index. Namespace the materialised corpus per embedder so each
    # arm builds + searches its OWN index over the (identical) corpus content.
    emb_tag = (embedder or "default").replace("/", "_")
    if not run_id:
        run_id = time.strftime(f"%Y%m%d-%H%M%S-{dataset}-{ablation}")
    out_dir = RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_root = out_dir / "corpora" / emb_tag

    rows: list[dict] = []
    manifests: list[dict] = []
    instr: list[dict] = []

    def _capture(out, label):
        instr.append({
            "cell": label,
            "embedder": out.embedder,
            "index_built": out.index_built,
            "index_secs": round(out.index_secs, 2),
            "index_peak_rss_mb": (round(out.index_peak_rss_mb, 1)
                                  if out.index_peak_rss_mb else None),
            "cold_latency_ms": (round(out.cold_latency_ms, 1)
                                if out.cold_latency_ms else None),
            "warm_latency_ms": (round(out.warm_latency_ms, 1)
                                if out.warm_latency_ms else None),
            "n_queries": len(out.latencies_ms),
        })

    if dataset == "csn":
        cfg = yaml.safe_load((CONFIG / "csn_subset.yaml").read_text())
        # Optional CPU-budget scope: RELEVANCE_CSN_LANGS=python[,javascript,go]
        # restricts the languages run (the seeded subset per language is unchanged).
        lang_override = os.environ.get("RELEVANCE_CSN_LANGS", "").strip()
        langs = [l.strip() for l in lang_override.split(",") if l.strip()] or cfg["languages"]
        for lang in langs:
            corpus = load_csn_corpus(
                language=lang, corpus_dir=corpus_root / f"csn_{lang}",
                dataset_id=cfg["dataset_id"], corpus_size=cfg["corpus_size"],
                query_size=cfg["query_size"], seed=cfg["seed"],
                trust_remote_code=cfg.get("trust_remote_code", True),
            )
            out = run_corpus(corpus, ablation=ablation, k=k, semantex_binary=semantex_bin,
                             embedder=embedder, match_mode="doc_id")
            rows.append(compute_metrics_row(out, k=k))
            _capture(out, f"csn/{lang}")
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
    elif dataset == "coir-codetrans-dl":
        # External-anchor CoIR sub-dataset (816 docs / 180 test queries, full
        # corpus, no subsetting). Same file-level matching + chunk-dedup protocol
        # as the acceptance gate (scripts/reproduce_baseline._run_external_coir),
        # but reported for ARBITRARY ablations + embedders (the D4 A/B), not the
        # sparse-only pass/fail gate.
        from scripts.reproduce_baseline import dedup_relevances_by_file, filewise_corpus
        bl = yaml.safe_load((CONFIG / "baselines.yaml").read_text())["coir_codetrans_dl"]
        corpus = load_coir_subdataset(
            name=bl["subdataset"],
            queries_corpus_id=bl["queries_corpus_id"],
            qrels_id=bl["qrels_id"],
            corpus_dir=corpus_root / f"coir_{bl['subdataset']}",
            corpus_size=bl.get("corpus_size"),
            query_size=bl.get("query_size"),
            seed=int(bl.get("seed", 0)),
            qrels_split=bl.get("qrels_split", "test"),
        )
        if corpus.manifest is not None:
            manifests.append(corpus.manifest.to_dict())
        fcorpus = filewise_corpus(corpus)
        retrieve_k = int(bl.get("retrieve_k", 50))
        raw = run_corpus(fcorpus, ablation=ablation, k=retrieve_k,
                         semantex_binary=semantex_bin, embedder=embedder, match_mode="file")
        out = dedup_relevances_by_file(raw, fcorpus)
        # dedup_relevances_by_file drops the instrumentation fields; re-attach so
        # the report can surface index build + latency for this cell.
        out.embedder = raw.embedder
        out.index_built = raw.index_built
        out.index_secs = raw.index_secs
        out.index_peak_rss_mb = raw.index_peak_rss_mb
        out.latencies_ms = raw.latencies_ms
        rows.append(compute_metrics_row(out, k=10))
        _capture(out, f"coir/{bl['subdataset']}")
    else:  # coir (multi-subdataset; not used by the D4 A/B)
        click.echo("Use --dataset coir-codetrans-dl for the D4 external anchor. The "
                   "generic multi-subdataset 'coir' path (cosqa/codesearchnet) is "
                   "not wired for the A/B.", err=True)
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
    import json as _json
    tag = f"{dataset}-{ablation}-{emb_tag}"
    (out_dir / f"report-{tag}.json").write_text(
        render_report_json(rows=rows, stamp=stamp, manifests=manifests))
    (out_dir / f"instr-{tag}.json").write_text(_json.dumps(instr, indent=2))
    (out_dir / "report.json").write_text(render_report_json(rows=rows, stamp=stamp, manifests=manifests))
    (out_dir / "report.md").write_text(render_report_md(rows=rows, stamp=stamp, manifests=manifests))
    click.echo(f"Report: {out_dir / 'report.md'}")
    click.echo((out_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
