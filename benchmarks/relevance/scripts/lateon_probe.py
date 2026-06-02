"""LateOn-Code (ColBERT late-interaction) as a code RETRIEVER, apples-to-apples
vs the coderank single-vector baseline on CoIR CodeTrans-DL.

WHY THIS IS APPLES-TO-APPLES
----------------------------
This probe reuses the EXACT harness machinery that produced the coderank
baseline numbers, so the queries / gold / metric are identical:

  * load the SAME 816-doc corpus + 180 test queries + qrels via the harness
    loader `relevance_harness.datasets.coir.load_coir_subdataset`, driven by the
    `coir_codetrans_dl` entry in config/baselines.yaml (subdataset, repo ids,
    qrels split, seed) — the same call reproduce_baseline / scripts.run make;
  * apply the SAME file-level matching (`filewise_corpus`) and per-file
    chunk-dedup (`dedup_relevances_by_file`) from scripts/reproduce_baseline.py;
  * score with the SAME pure metric functions (`ndcg_at_k`, `recall_at_k`,
    `mrr_at_k`) from `relevance_harness.metrics`.

The ONLY thing that differs from the baseline run is the retriever: instead of
semantex's coderank-hnsw single-vector channel, we encode corpus + queries with
a PyLate ColBERT multi-vector model and retrieve via late-interaction MaxSim
(PyLate Voyager/PLAID index). We then materialise the same per-query
`RankedResult(ranked_files=...)` structure the harness scores.

NOTE on doc identity: each CoIR doc is materialised by the loader to its own
file `doc_<idx>.txt` (see datasets/coir.build_corpus_from_splits). `filewise_corpus`
rewrites gold to those file paths. We therefore use the file path (relpath) as
the PyLate document id, so a retrieved id IS a file path and slots straight into
`ranked_files` — file-level granularity, matching the published MTEB protocol.

CPU-ONLY: semantex is a CPU-only local tool (hard product constraint). We force
device="cpu" everywhere (model + retrieve scoring).

SMOKE vs FULL: `--query-size 15` (default) runs a 15-query smoke to verify
wiring + sane numbers; `--query-size 0` runs the full seeded 180-query set.

OTHER MODELS: `--model lightonai/LateOn-Code` (130M) etc. — the probe is fully
parameterised on the PyLate model id, so the Run phase can A/B the bigger model.

Example:
  ORT_DYLIB_PATH=... SEMANTEX_QUIET_LIMITS=1 \
  .venv/bin/python scripts/lateon_probe.py --query-size 15
  .venv/bin/python scripts/lateon_probe.py --query-size 0   # full 180q
  .venv/bin/python scripts/lateon_probe.py --model lightonai/LateOn-Code --query-size 0
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

import click
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
# Make the `scripts` package importable whether this file is run directly
# (python scripts/lateon_probe.py) or as a module (python -m scripts.lateon_probe).
sys.path.insert(0, str(ROOT))

from relevance_harness.datasets.coir import load_coir_subdataset
from relevance_harness.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from relevance_harness.types import EvalCorpus

# Reuse the EXACT file-level matching helper the acceptance gate uses, so this
# probe is numerically comparable to the coderank baseline.
from scripts.reproduce_baseline import filewise_corpus  # noqa: E402

CONFIG = ROOT / "config"
RESULTS = ROOT / "results"


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024 or unit == "GB":
            return f"{f:.1f}{unit}"
        f /= 1024
    return f"{f:.1f}GB"


def _build_relevances(
    *,
    ranked_files_per_query: list[list[str]],
    gold_per_query: list[set[str]],
    k_dedup_cap: int | None = None,
) -> tuple[list[list[int]], list[int]]:
    """Collapse ranked file lists to one entry per file (best rank kept) and
    build the 0/1 relevance vectors + per-query n_relevant.

    Mirrors scripts.reproduce_baseline.dedup_relevances_by_file exactly: keep
    only the first occurrence of each file, then mark 1 if that file is gold.
    PyLate already returns at most one entry per document id (file), so the dedup
    is a no-op here, but we keep it for protocol parity / defensiveness.
    """
    relevances: list[list[int]] = []
    n_relevant: list[int] = []
    for ranked_files, gold in zip(ranked_files_per_query, gold_per_query):
        seen: set[str] = set()
        rels: list[int] = []
        for f in ranked_files:
            if f in seen:
                continue
            seen.add(f)
            rels.append(1 if f in gold else 0)
        relevances.append(rels)
        n_relevant.append(len(gold))
    return relevances, n_relevant


def _load_corpus(query_size: int) -> EvalCorpus:
    """Load the CoIR CodeTrans-DL corpus via the harness loader.

    Always materialises the FULL 816-doc corpus + the FULL seeded 180-query set
    (query_size=None in the loader so the seeded selection is identical to the
    baseline run). For a smoke we then pick a SEEDED RANDOM subset of N queries
    (not the sorted prefix): the corpus's queries are sorted by id, and the first
    handful happen to be near-duplicate variants of one d2l TF->Paddle snippet
    (query_ids 637..) that cluster pathologically and badly under-represent the
    task. A `random.Random(seed).sample` draw is representative AND reproducible
    — it mirrors how the harness's own `subset.select_queries` would subset.
    """
    bl = yaml.safe_load((CONFIG / "baselines.yaml").read_text())["coir_codetrans_dl"]
    corpus = load_coir_subdataset(
        name=bl["subdataset"],
        queries_corpus_id=bl["queries_corpus_id"],
        qrels_id=bl["qrels_id"],
        corpus_dir=RESULTS / "_baseline" / f"coir_{bl['subdataset']}",
        corpus_size=bl.get("corpus_size"),  # None => full 816
        query_size=bl.get("query_size"),  # None => full seeded 180
        seed=int(bl.get("seed", 0)),
        qrels_split=bl.get("qrels_split", "test"),
    )
    # File-level gold (rewrites gold_doc_ids -> file paths), matching the gate.
    fcorpus = filewise_corpus(corpus)
    if query_size and query_size > 0 and query_size < len(fcorpus.queries):
        import random
        seed = int(bl.get("seed", 0))
        # Seeded random sample, then restore id order for stable reporting.
        chosen = random.Random(seed).sample(list(fcorpus.queries), query_size)
        chosen.sort(key=lambda q: q.query_id)
        fcorpus = EvalCorpus(
            name=fcorpus.name,
            documents=fcorpus.documents,
            queries=tuple(chosen),
            corpus_dir=fcorpus.corpus_dir,
            manifest=fcorpus.manifest,
        )
    return fcorpus


def _load_model(model_id: str, query_length: int | None, document_length: int | None):
    from pylate import models

    kwargs: dict = {"model_name_or_path": model_id, "device": "cpu"}
    # Let the model card drive prefixes/lengths by default; only override if the
    # caller explicitly passes them (the model card already sets [Q]/[D] prefixes,
    # query_length 256, document_length 2048 for LateOn-Code-edge).
    if query_length:
        kwargs["query_length"] = query_length
    if document_length:
        kwargs["document_length"] = document_length
    model = models.ColBERT(**kwargs)
    return model


def _build_index(index_kind: str, index_folder: Path, embedding_size: int):
    from pylate import indexes

    index_folder.mkdir(parents=True, exist_ok=True)
    if index_kind == "voyager":
        # HNSW; robust on a tiny CPU corpus (no PLAID k-means cluster-count
        # constraints). embedding_size MUST match the model's dim (48 for LateOn).
        return indexes.Voyager(
            index_folder=str(index_folder),
            index_name="lateon",
            override=True,
            embedding_size=embedding_size,
        )
    if index_kind == "plaid":
        # FastPLAID (product-quantised). kmeans needs enough points; on 816 docs
        # we shrink max_points_per_centroid so the cluster count stays sane.
        return indexes.PLAID(
            index_folder=str(index_folder),
            index_name="lateon",
            override=True,
            device="cpu",
            kmeans_niters=4,
        )
    raise ValueError(f"unknown index kind {index_kind!r}")


@click.command()
@click.option("--model", "model_id", default="lightonai/LateOn-Code-edge",
              show_default=True, help="PyLate ColBERT model id (e.g. "
              "lightonai/LateOn-Code-edge | lightonai/LateOn-Code).")
@click.option("--query-size", default=15, type=int, show_default=True,
              help="Number of queries: 15 = smoke; 0 = full seeded 180.")
@click.option("--index", "index_kind", type=click.Choice(["voyager", "plaid"]),
              default="voyager", show_default=True,
              help="PyLate index backend (Voyager=HNSW, robust on tiny corpora).")
@click.option("--retrieve-k", default=50, type=int, show_default=True,
              help="Top-k to retrieve before scoring nDCG@10 over the dedup'd files.")
@click.option("--batch-size", default=32, type=int, show_default=True)
@click.option("--embedding-size", default=48, type=int, show_default=True,
              help="Multi-vector dim (48 for LateOn-Code-edge; needed by Voyager).")
@click.option("--query-length", default=0, type=int,
              help="Override model query_length (0 = use model card default 256).")
@click.option("--document-length", default=0, type=int,
              help="Override model document_length (0 = use model card default 2048).")
@click.option("--run-id", default="", help="results/<run-id> dir (default timestamped).")
def main(model_id, query_size, index_kind, retrieve_k, batch_size, embedding_size,
         query_length, document_length, run_id):
    t_start = time.monotonic()
    mode = "smoke" if (query_size and query_size > 0) else "full"
    emb_tag = model_id.replace("/", "_")
    if not run_id:
        run_id = time.strftime(f"%Y%m%d-%H%M%S-lateon-{mode}-{emb_tag}")
    out_dir = RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    index_folder = out_dir / "pylate_index"
    if index_folder.exists():
        shutil.rmtree(index_folder)

    click.echo(f"[lateon_probe] model={model_id} mode={mode} query_size={query_size} "
               f"index={index_kind} retrieve_k={retrieve_k} device=cpu")

    # --- 1. Load the SAME CoIR corpus + queries + gold as the baseline --------
    fcorpus = _load_corpus(query_size)
    docs = list(fcorpus.documents)
    doc_ids = [d.file_path for d in docs]          # file path == document identity
    doc_texts = [d.text for d in docs]
    queries = list(fcorpus.queries)
    gold_per_query = [set(q.gold_doc_ids) for q in queries]
    if fcorpus.manifest is not None:
        m = fcorpus.manifest
        click.echo(f"[lateon_probe] subset manifest: dataset={m.dataset} "
                   f"kept={m.selected}/{m.total} seed={m.seed}")
    click.echo(f"[lateon_probe] corpus_docs={len(docs)} queries={len(queries)} "
               f"(gold is file-level; each doc materialised as its own file)")

    # --- 2. Load PyLate ColBERT model (CPU) -----------------------------------
    t0 = time.monotonic()
    model = _load_model(model_id, query_length or None, document_length or None)
    model_load_s = time.monotonic() - t0
    # Surface the resolved tokenization config for provenance.
    cfg = {
        "query_prefix": getattr(model, "query_prefix", None),
        "document_prefix": getattr(model, "document_prefix", None),
        "query_length": getattr(model, "query_length", None),
        "document_length": getattr(model, "document_length", None),
    }
    click.echo(f"[lateon_probe] model loaded in {model_load_s:.1f}s; tok cfg={cfg}")

    # --- 3. Encode + index the corpus -----------------------------------------
    t0 = time.monotonic()
    doc_emb = model.encode(
        doc_texts, batch_size=batch_size, is_query=False,
        show_progress_bar=False, convert_to_numpy=True,
    )
    corpus_encode_s = time.monotonic() - t0
    index = _build_index(index_kind, index_folder, embedding_size)
    t0 = time.monotonic()
    index.add_documents(documents_ids=doc_ids, documents_embeddings=doc_emb)
    index_build_s = time.monotonic() - t0
    click.echo(f"[lateon_probe] corpus encoded in {corpus_encode_s:.1f}s; "
               f"index built in {index_build_s:.1f}s")

    from pylate import retrieve as _retrieve
    retriever = _retrieve.ColBERT(index=index)

    # --- 4. Per-query: encode query + MaxSim retrieve (timed on CPU) ----------
    ranked_files_per_query: list[list[str]] = []
    latencies_ms: list[float] = []
    for q in queries:
        t0 = time.monotonic()
        q_emb = model.encode(
            [q.text], batch_size=1, is_query=True,
            show_progress_bar=False, convert_to_numpy=True,
        )
        results = retriever.retrieve(
            queries_embeddings=q_emb, k=retrieve_k, device="cpu",
        )
        latencies_ms.append((time.monotonic() - t0) * 1000.0)
        # results is list[list[RerankResult]] — one inner list for our one query.
        ranked = [str(r["id"]) for r in results[0]] if results else []
        ranked_files_per_query.append(ranked)

    # --- 5. Score with the harness metrics (file-level, chunk-dedup'd) --------
    relevances, n_relevant = _build_relevances(
        ranked_files_per_query=ranked_files_per_query,
        gold_per_query=gold_per_query,
    )
    ndcg10 = ndcg_at_k(relevances, k=10, n_relevant=n_relevant)
    recall10 = recall_at_k(relevances, k=10, n_relevant=n_relevant)
    mrr10 = mrr_at_k(relevances, k=10)

    # --- 6. Index-size + latency instrumentation ------------------------------
    index_bytes = _dir_size_bytes(index_folder)
    per_query_lat = sorted(latencies_ms)
    lat_median = statistics.median(latencies_ms) if latencies_ms else 0.0
    lat_mean = statistics.fmean(latencies_ms) if latencies_ms else 0.0
    lat_p90 = per_query_lat[int(0.9 * (len(per_query_lat) - 1))] if per_query_lat else 0.0

    report = {
        "model": model_id,
        "mode": mode,
        "index_kind": index_kind,
        "device": "cpu",
        "n_corpus_docs": len(docs),
        "n_queries": len(queries),
        "retrieve_k": retrieve_k,
        "embedding_size": embedding_size,
        "tokenization_cfg": cfg,
        "metrics": {
            "ndcg_at_10": round(ndcg10, 4),
            "recall_at_10": round(recall10, 4),
            "mrr_at_10": round(mrr10, 4),
        },
        "baseline_note": {
            "coderank_137m_adaptive_off_dense": {"ndcg_at_10": 0.2805, "recall_at_10": 0.7333},
            "coderank_137m_adaptive_off_hybrid": {"ndcg_at_10": 0.2760, "recall_at_10": 0.7222},
            "single_vector_index_rss_mb": 438,
        },
        "latency_ms_per_query": {
            "median": round(lat_median, 1),
            "mean": round(lat_mean, 1),
            "p90": round(lat_p90, 1),
            "min": round(per_query_lat[0], 1) if per_query_lat else None,
            "max": round(per_query_lat[-1], 1) if per_query_lat else None,
        },
        "timings_s": {
            "model_load": round(model_load_s, 1),
            "corpus_encode": round(corpus_encode_s, 1),
            "index_build": round(index_build_s, 1),
            "total": round(time.monotonic() - t_start, 1),
        },
        "index_size": {
            "bytes": index_bytes,
            "human": _human(index_bytes),
            "note": f"multi-vector {index_kind} index for {len(docs)}-doc corpus; "
                    "extrapolate RSS vs the 438MB single-vector coderank baseline.",
        },
    }
    (out_dir / "lateon_report.json").write_text(json.dumps(report, indent=2))

    click.echo("")
    click.echo("=" * 72)
    click.echo(f"LateOn probe ({mode}) — {model_id}")
    click.echo("=" * 72)
    click.echo(f"  nDCG@10   = {ndcg10:.4f}   (baseline coderank dense 0.2805)")
    click.echo(f"  Recall@10 = {recall10:.4f}   (baseline coderank dense 0.7333)")
    click.echo(f"  MRR@10    = {mrr10:.4f}")
    click.echo(f"  per-query CPU latency: median={lat_median:.1f}ms "
               f"mean={lat_mean:.1f}ms p90={lat_p90:.1f}ms")
    click.echo(f"  index on disk: {_human(index_bytes)} ({len(docs)} docs, dim {embedding_size})")
    click.echo(f"  report: {out_dir / 'lateon_report.json'}")


if __name__ == "__main__":
    main()
