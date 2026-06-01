import pytest

from relevance_harness.types import (
    Document, EvalCorpus, Query, RankedResult,
)


def test_query_and_document_round_trip():
    q = Query(query_id="q1", text="how is auth handled", gold_doc_ids=("d1", "d3"))
    d = Document(doc_id="d1", text="def login(): ...", file_path="auth.py",
                 start_line=10, end_line=20)
    assert q.gold_doc_ids == ("d1", "d3")
    assert d.file_path == "auth.py"
    assert d.start_line == 10


def test_eval_corpus_qrels_lookup():
    corpus = EvalCorpus(
        name="tiny",
        documents=(
            Document(doc_id="d1", text="a", file_path="a.py", start_line=1, end_line=2),
            Document(doc_id="d2", text="b", file_path="b.py", start_line=1, end_line=2),
        ),
        queries=(
            Query(query_id="q1", text="find a", gold_doc_ids=("d1",)),
        ),
        corpus_dir=None,
    )
    assert corpus.name == "tiny"
    assert len(corpus.documents) == 2
    assert corpus.queries[0].gold_doc_ids == ("d1",)
    # qrels() returns {query_id: {doc_id: 1}}
    assert corpus.qrels() == {"q1": {"d1": 1}}


def test_ranked_result_holds_ordered_doc_ids():
    rr = RankedResult(query_id="q1", ranked_doc_ids=("d3", "d1", "d2"))
    assert rr.ranked_doc_ids[0] == "d3"
    assert rr.rank_of("d1") == 2          # 1-based rank
    assert rr.rank_of("missing") is None


def test_ranked_result_file_level_rank():
    # When matching by file (SWE-loc / CSN), ranked entries carry file paths.
    rr = RankedResult(
        query_id="q1",
        ranked_doc_ids=("x.py:1-5", "auth.py:10-20", "z.py:1-9"),
        ranked_files=("x.py", "auth.py", "z.py"),
    )
    assert rr.rank_of_file("auth.py") == 2
    assert rr.rank_of_file("absent.py") is None
