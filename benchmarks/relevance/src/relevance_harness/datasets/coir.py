"""CoIR loader → EvalCorpus.

CoIR sub-datasets ship as three logical splits: corpus (docs), queries, qrels
(TREC-style query-id/corpus-id/score rows). We materialise each corpus doc to a
file under corpus_dir, mint a doc id "<sourceid>__<relpath>:1-<nlines>", build
queries from the queries split, and attach gold doc ids via the qrels mapping.
Queries with no qrels entry are dropped (logged via the subset manifest size).

HF layout (research notes, Task 0.1 Step 4): the on-Hub repos are
`CoIR-Retrieval/<name>-queries-corpus` (one repo, splits `corpus` + `queries`,
cols `_id`/`text`) and `CoIR-Retrieval/<name>-qrels` (split `test`, cols
`query_id`/`corpus_id`/`score` — UNDERSCORES). `load_coir_subdataset` normalizes
the underscore qrels keys to the hyphen keys this module's injectable
`build_corpus_from_splits` expects (which keeps the unit tests stable).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..subset import select_queries
from ..types import Document, EvalCorpus, Query


def _qrels_map(qrel_rows: list[dict]) -> dict[str, list[str]]:
    """{query-id: [corpus-id, ...]} for rows with positive relevance."""
    out: dict[str, list[str]] = {}
    for r in qrel_rows:
        if int(r.get("score", 0)) <= 0:
            continue
        out.setdefault(r["query-id"], []).append(r["corpus-id"])
    return out


def build_corpus_from_splits(
    *,
    name: str,
    corpus_rows: list[dict],
    query_rows: list[dict],
    qrel_rows: list[dict],
    corpus_dir: Path,
    corpus_size: Optional[int],
    query_size: Optional[int],
    seed: int,
) -> EvalCorpus:
    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    rows = sorted(corpus_rows, key=lambda r: r["_id"])
    if corpus_size is not None:
        rows = rows[:corpus_size]

    documents: list[Document] = []
    source_to_docid: dict[str, str] = {}
    for idx, r in enumerate(rows):
        text = r["text"]
        relpath = f"doc_{idx}.txt"
        (corpus_dir / relpath).write_text(text)
        nlines = max(1, text.count("\n") + 1)
        doc_id = f"{r['_id']}__{relpath}:1-{nlines}"
        source_to_docid[r["_id"]] = doc_id
        documents.append(
            Document(doc_id=doc_id, text=text, file_path=relpath, start_line=1, end_line=nlines)
        )

    qmap = _qrels_map(qrel_rows)
    candidate_queries: list[dict] = []
    for r in query_rows:
        gold_source_ids = qmap.get(r["_id"], [])
        gold_doc_ids = [source_to_docid[g] for g in gold_source_ids if g in source_to_docid]
        if not gold_doc_ids:
            continue
        candidate_queries.append(
            {"query_id": r["_id"], "text": r["text"], "gold_doc_ids": tuple(gold_doc_ids)}
        )

    kept, manifest = select_queries(
        candidate_queries, n=query_size, seed=seed, dataset=f"coir/{name}"
    )
    queries = tuple(
        Query(query_id=q["query_id"], text=q["text"], gold_doc_ids=q["gold_doc_ids"])
        for q in kept
    )
    return EvalCorpus(
        name=f"coir/{name}",
        documents=tuple(documents),
        queries=queries,
        corpus_dir=corpus_dir,
        manifest=manifest,
    )


def _normalize_qrel_rows(qrel_rows: list[dict]) -> list[dict]:
    """Map the on-Hub underscore qrels keys to the hyphen keys this module uses.

    HF cols are `query_id` / `corpus_id`; build_corpus_from_splits + its unit
    tests use `query-id` / `corpus-id`. Pass-through if already hyphenated.
    """
    out: list[dict] = []
    for r in qrel_rows:
        out.append(
            {
                "query-id": r.get("query-id", r.get("query_id")),
                "corpus-id": r.get("corpus-id", r.get("corpus_id")),
                "score": r.get("score", 0),
            }
        )
    return out


def load_coir_subdataset(
    *,
    name: str,
    queries_corpus_id: str,
    qrels_id: str,
    corpus_dir: Path,
    corpus_size: Optional[int],
    query_size: Optional[int],
    seed: int,
    qrels_split: str = "test",
) -> EvalCorpus:
    """Load one CoIR sub-dataset from HuggingFace, materialise, subset.

    The combined `<name>-queries-corpus` repo exposes the `corpus` and `queries`
    splits; the `<name>-qrels` repo exposes train/test/valid. HF split/column
    names per research notes (Task 0.1 Step 4). If the org layout differs, adjust
    the load_dataset calls + the normalization here.
    """
    from datasets import load_dataset

    corpus_rows = list(load_dataset(queries_corpus_id, split="corpus"))
    query_rows = list(load_dataset(queries_corpus_id, split="queries"))
    qrel_rows = _normalize_qrel_rows(list(load_dataset(qrels_id, split=qrels_split)))
    return build_corpus_from_splits(
        name=name, corpus_rows=corpus_rows, query_rows=query_rows, qrel_rows=qrel_rows,
        corpus_dir=corpus_dir, corpus_size=corpus_size, query_size=query_size, seed=seed,
    )
