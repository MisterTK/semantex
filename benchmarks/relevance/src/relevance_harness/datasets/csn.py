"""CodeSearchNet loader → EvalCorpus.

Each function's code is written to a file under corpus_dir so semantex can index
it; the doc id is "<relpath>:1-<nlines>" to match SemantexClient's id minting.
Queries are the functions' docstrings (functions without a docstring are indexed
but not queried). Field names per research notes (Task 0.1 Step 3).

Dataset id: `code-search-net/code_search_net` (official parquet mirror; needs NO
trust_remote_code). Per-language config = python|java|javascript|go|php|ruby.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from ..subset import select_queries
from ..types import Document, EvalCorpus, Query


def _safe_relpath(func_code_url: str, fallback_path: str, idx: int) -> str:
    """A unique, filesystem-safe relative path for one function's file."""
    base = fallback_path or f"func_{idx}"
    base = re.sub(r"[^A-Za-z0-9_./-]", "_", base)
    # disambiguate by index so two funcs in the same source path don't collide
    stem, dot, ext = base.rpartition(".")
    if dot:
        return f"{stem}__{idx}.{ext}"
    return f"{base}__{idx}.txt"


def build_corpus_from_rows(
    *,
    rows: list[dict],
    language: str,
    corpus_dir: Path,
    query_size: Optional[int],
    seed: int,
) -> EvalCorpus:
    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    documents: list[Document] = []
    candidate_queries: list[dict] = []
    for idx, r in enumerate(rows):
        code = r["whole_func_string"]
        relpath = _safe_relpath(r.get("func_code_url", ""), r.get("func_path_in_repository", ""), idx)
        dest = corpus_dir / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(code)
        nlines = max(1, code.count("\n") + 1)
        doc_id = f"{relpath}:1-{nlines}"
        documents.append(
            Document(doc_id=doc_id, text=code, file_path=relpath, start_line=1, end_line=nlines)
        )
        doc = (r.get("func_documentation_string") or "").strip()
        if doc:
            candidate_queries.append(
                {"query_id": r["func_code_url"], "text": doc, "gold_doc_id": doc_id}
            )

    kept, _manifest = select_queries(
        candidate_queries, n=query_size, seed=seed, dataset=f"csn/{language}"
    )
    queries = tuple(
        Query(query_id=q["query_id"], text=q["text"], gold_doc_ids=(q["gold_doc_id"],))
        for q in kept
    )
    return EvalCorpus(
        name=f"csn/{language}",
        documents=tuple(documents),
        queries=queries,
        corpus_dir=corpus_dir,
    )


def load_csn_corpus(
    *,
    language: str,
    corpus_dir: Path,
    dataset_id: str,
    corpus_size: Optional[int],
    query_size: Optional[int],
    seed: int,
    trust_remote_code: bool = True,
) -> EvalCorpus:
    """Load CodeSearchNet for one language from HuggingFace, materialise, subset.

    `corpus_size` caps the number of indexed functions (None = all of the test
    split). Selection of the corpus slice is the first `corpus_size` rows after
    sorting by func_code_url (deterministic)."""
    import inspect

    from datasets import load_dataset

    # `trust_remote_code` was dropped from load_dataset's signature in recent
    # `datasets` releases (4.x). The official parquet mirror needs no remote code,
    # so only forward the kwarg when the installed version still accepts it.
    kwargs = {}
    if "trust_remote_code" in inspect.signature(load_dataset).parameters:
        kwargs["trust_remote_code"] = trust_remote_code
    ds = load_dataset(dataset_id, language, split="test", **kwargs)
    rows = sorted(list(ds), key=lambda r: r["func_code_url"])
    if corpus_size is not None:
        rows = rows[:corpus_size]
    return build_corpus_from_rows(
        rows=rows, language=language, corpus_dir=corpus_dir,
        query_size=query_size, seed=seed,
    )
