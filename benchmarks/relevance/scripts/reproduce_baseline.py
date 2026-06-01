"""Acceptance gate: reproduce a baseline within a stated tolerance.

Two gate kinds (selected by the baseline entry's `type` in baselines.yaml):

  external_coir        — the ACCEPTANCE gate of record. Reproduces a PUBLISHED
                         external BM25 number (MTEB BM25 baseline) on a code-to-code
                         CoIR task (CodeTrans-DL). CoIR queries↔docs are both code
                         (no docstring-in-document leakage), so this validates the
                         ranking PROTOCOL against an independent source.

  internal_determinism — self-consistency / regression gate. Re-runs semantex's
                         own --sparse-only BM25 on the leaky CSN whole_func_string
                         subset across python+js+go and asserts each language's
                         MRR@10 is stable vs a recorded SELF-baseline. Catches
                         wiring breaks; NOT external protocol validation.

Exits non-zero if any language/metric falls outside tolerance.
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

from relevance_harness.datasets.coir import load_coir_subdataset
from relevance_harness.datasets.csn import load_csn_corpus
from relevance_harness.metrics import mrr_at_k, ndcg_at_k
from relevance_harness.runner import RunOutput, run_corpus
from relevance_harness.types import EvalCorpus, Query

CONFIG = ROOT / "config"


# --- pure protocol helpers (unit-tested; no network) -----------------------

def evaluate_metric(out: RunOutput, *, metric: str, k: int = 10) -> float:
    """Compute the gate's target metric from a RunOutput."""
    if metric == "ndcg_at_10":
        return ndcg_at_k(out.relevances, k=k, n_relevant=out.n_relevant)
    if metric == "mrr_at_10":
        return mrr_at_k(out.relevances, k=k)
    raise ValueError(f"unknown gate metric {metric!r}")


def within_tolerance(measured: float, expected: float, tolerance: float) -> bool:
    return abs(measured - expected) <= tolerance


def dedup_relevances_by_file(out: RunOutput, corpus: EvalCorpus) -> RunOutput:
    """Collapse each query's ranked chunks to one entry per file (best rank kept).

    semantex returns several chunks per multi-line doc; the published MTEB protocol
    scores at DOCUMENT granularity. We therefore keep only the first (highest-ranked)
    occurrence of each file before building the relevance vector, so a single gold
    document is counted once (never inflating nDCG above 1). `corpus.queries` carry
    file-path gold (see filewise_corpus); n_relevant stays the gold-file count.
    """
    gold_by_qid = {q.query_id: set(q.gold_doc_ids) for q in corpus.queries}
    relevances: list[list[int]] = []
    n_relevant: list[int] = []
    for rr in out.per_query:
        gold = gold_by_qid.get(rr.query_id, set())
        seen: set[str] = set()
        rels: list[int] = []
        for f in rr.ranked_files:
            if f in seen:
                continue
            seen.add(f)
            rels.append(1 if f in gold else 0)
        relevances.append(rels)
        n_relevant.append(len(gold))
    return RunOutput(corpus_name=out.corpus_name, ablation=out.ablation,
                     relevances=relevances, n_relevant=n_relevant, per_query=out.per_query)


def filewise_corpus(corpus: EvalCorpus) -> EvalCorpus:
    """Rewrite each query's gold doc-ids to gold FILE paths for file-level matching.

    Each CoIR/CSN doc is materialised as its own file, so the file path uniquely
    identifies the gold document. We match on file (not the "file:1-nlines" doc-id)
    because semantex chunks multi-line docs into several spans — the whole-file
    doc-id "file:1-N" never equals a returned chunk's "file:start-end", which would
    spuriously zero out recall. File-level matching is the correct doc identity here.
    """
    docid_to_file = {d.doc_id: d.file_path for d in corpus.documents}
    new_queries = tuple(
        Query(
            query_id=q.query_id,
            text=q.text,
            gold_doc_ids=tuple(
                docid_to_file.get(g, g) for g in q.gold_doc_ids
            ),
        )
        for q in corpus.queries
    )
    return EvalCorpus(
        name=corpus.name,
        documents=corpus.documents,
        queries=new_queries,
        corpus_dir=corpus.corpus_dir,
        manifest=corpus.manifest,
    )


# --- gate runners ----------------------------------------------------------

def _run_external_coir(b: dict, semantex_bin: str) -> int:
    """External-reproduction gate against a published BM25 CoIR number."""
    corpus = load_coir_subdataset(
        name=b["subdataset"],
        queries_corpus_id=b["queries_corpus_id"],
        qrels_id=b["qrels_id"],
        corpus_dir=ROOT / "results" / "_baseline" / f"coir_{b['subdataset']}",
        corpus_size=b.get("corpus_size"),
        query_size=b.get("query_size"),
        seed=int(b.get("seed", 0)),
        qrels_split=b.get("qrels_split", "test"),
    )
    if corpus.manifest is not None:
        m = corpus.manifest
        click.echo(f"subset manifest: dataset={m.dataset} kept={m.selected}/{m.total} "
                   f"dropped={len(m.dropped_ids)} seed={m.seed}")
    # File-level matching: each CoIR doc is its own file, and semantex chunks
    # multi-line docs, so the whole-file "file:1-N" doc-id never equals a returned
    # chunk's id. Match on file identity (see filewise_corpus docstring), then
    # collapse to one entry per file (document granularity, matching MTEB).
    fcorpus = filewise_corpus(corpus)
    # Retrieve deeper than 10 so that after collapsing chunks to unique files we
    # still have >=10 distinct documents to score nDCG@10 over.
    retrieve_k = int(b.get("retrieve_k", 50))
    raw = run_corpus(fcorpus, ablation=b["ablation"], k=retrieve_k,
                     semantex_binary=semantex_bin, match_mode="file")
    out = dedup_relevances_by_file(raw, fcorpus)
    metric = b["metric"]
    measured = evaluate_metric(out, metric=metric, k=10)

    # The gate makes TWO explicit assertions; BOTH must pass.
    #
    #  (1) TIGHT internal-determinism band on semantex's OWN measured CodeTrans-DL
    #      nDCG@10 (self_baseline_ndcg_at_10). This is the band that actually catches
    #      a subtle RANKING regression (e.g. 0.19 -> 0.12 would blow past it).
    #
    #  (2) LOOSE external sanity bound vs the published MTEB BM25 number
    #      (expected_ndcg_at_10). This proves end-to-end wiring against an
    #      independent source and fails on a GROSS protocol break (collapse to ~0).
    self_baseline = float(b["self_baseline_ndcg_at_10"])
    internal_tol = float(b["internal_tolerance"])
    published = float(b["expected_ndcg_at_10"])
    external_tol = float(b["tolerance"])

    internal_delta = abs(measured - self_baseline)
    external_delta = abs(measured - published)
    internal_ok = within_tolerance(measured, self_baseline, internal_tol)
    external_ok = within_tolerance(measured, published, external_tol)

    click.echo(f"[external_coir/{b['subdataset']}] measured_{metric}={measured:.4f} "
               f"(n_queries={len(out.relevances)})")
    click.echo(f"  (1) internal determinism: self_baseline={self_baseline:.4f} "
               f"tol={internal_tol:.4f} delta={internal_delta:.4f} "
               f"-> {'OK' if internal_ok else 'REGRESSED'}")
    click.echo(f"  (2) external sanity:      published={published:.4f} "
               f"tol={external_tol:.4f} delta={external_delta:.4f} "
               f"-> {'OK' if external_ok else 'OUT-OF-BOUND'}")
    click.echo(f"source: {b['source']}")
    if not internal_ok:
        click.echo("FAIL: internal-determinism band breached — a RANKING regression "
                   "in semantex's measured nDCG@10 (not a tolerance to widen; debug "
                   "the change that moved the number).", err=True)
        return 1
    if not external_ok:
        click.echo("FAIL: external sanity bound breached — gross protocol/wiring break "
                   "vs the published MTEB BM25 number; debug wiring.", err=True)
        return 1
    click.echo("PASS: (1) stable vs semantex's own measured baseline AND (2) within "
               "the external sanity bound of the published MTEB BM25 number.")
    return 0


def _run_internal_determinism(b: dict, semantex_bin: str) -> int:
    """Self-consistency gate: multi-language CSN MRR vs recorded self-baselines."""
    csn_cfg = yaml.safe_load((CONFIG / "csn_subset.yaml").read_text())
    tol = float(b["tolerance"])
    failures = []
    for language, self_baseline in b["per_language"].items():
        corpus = load_csn_corpus(
            language=language,
            corpus_dir=ROOT / "results" / "_baseline" / f"csn_{language}",
            dataset_id=csn_cfg["dataset_id"],
            corpus_size=csn_cfg["corpus_size"],
            query_size=csn_cfg["query_size"],
            seed=csn_cfg["seed"],
            trust_remote_code=csn_cfg.get("trust_remote_code", True),
        )
        out = run_corpus(corpus, ablation=b["ablation"], k=10, semantex_binary=semantex_bin,
                         match_mode="doc_id")
        measured = mrr_at_k(out.relevances, k=10)
        expected = float(self_baseline)
        delta = abs(measured - expected)
        ok = delta <= tol
        click.echo(f"[internal/csn/{language}] measured_mrr@10={measured:.4f} "
                   f"self_baseline={expected:.4f} tol={tol:.4f} delta={delta:.4f} "
                   f"{'OK' if ok else 'REGRESSED'}")
        if not ok:
            failures.append(language)
    click.echo(f"source: {b['source']}")
    if failures:
        click.echo(f"FAIL: per-language regression in {failures} — a wiring break, "
                   f"not external protocol validation.", err=True)
        return 1
    click.echo("PASS: all languages stable vs self-baseline (no wiring regression). "
               "NB: internal self-consistency only — see coir_codetrans_dl for the "
               "external protocol gate.")
    return 0


@click.command()
@click.option("--baseline", default="coir_codetrans_dl", show_default=True,
              help="coir_codetrans_dl (external) | csn_internal_determinism (internal)")
@click.option("--semantex-bin", default=os.environ.get("SEMANTEX_BINARY", "semantex"))
def main(baseline: str, semantex_bin: str):
    baselines = yaml.safe_load((CONFIG / "baselines.yaml").read_text())
    if baseline not in baselines:
        raise SystemExit(f"unknown baseline {baseline!r}; have {sorted(baselines)}")
    b = baselines[baseline]
    kind = b.get("type")
    if kind == "external_coir":
        rc = _run_external_coir(b, semantex_bin)
    elif kind == "internal_determinism":
        rc = _run_internal_determinism(b, semantex_bin)
    else:
        raise SystemExit(f"baseline {baseline!r} has unknown type {kind!r}")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
