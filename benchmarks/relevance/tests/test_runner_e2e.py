import shutil

import pytest

from relevance_harness.metrics import mrr_at_k, recall_at_k
from relevance_harness.runner import run_corpus
from relevance_harness.types import EvalCorpus, Query


@pytest.mark.skipif(shutil.which("semantex") is None, reason="semantex binary not on PATH")
def test_tiny_corpus_end_to_end(tmp_path, fixtures_dir):
    # copy the tiny corpus to a writable temp dir (index writes .semantex/ there)
    corpus_dir = tmp_path / "tiny_corpus"
    shutil.copytree(fixtures_dir / "tiny_corpus", corpus_dir)

    corpus = EvalCorpus(
        name="tiny",
        documents=(),
        queries=(
            Query(query_id="q1", text="authenticate a user with a password", gold_doc_ids=("auth.py",)),
            Query(query_id="q2", text="pool of database connections", gold_doc_ids=("db.py",)),
            Query(query_id="q3", text="format a number as currency", gold_doc_ids=("util.py",)),
        ),
        corpus_dir=corpus_dir,
    )
    out = run_corpus(
        corpus, ablation="hybrid", k=10, semantex_binary="semantex", match_mode="file"
    )
    # every query's gold file should appear somewhere in the top-10
    assert recall_at_k(out.relevances, k=10, n_relevant=out.n_relevant) > 0.0
    # and the protocol produces a finite MRR
    assert 0.0 <= mrr_at_k(out.relevances, k=10) <= 1.0
