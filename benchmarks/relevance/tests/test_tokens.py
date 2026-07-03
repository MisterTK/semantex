from relevance_harness.tokens import estimate_tokens_returned
from relevance_harness.types import RankedResult


def test_estimate_tokens_returned_sums_content_chars():
    rr = RankedResult(
        query_id="q1",
        ranked_doc_ids=("a.py:1-2", "b.py:3-4"),
        ranked_files=("a.py", "b.py"),
        raw=({"content": "x" * 40}, {"content": "y" * 20}),
    )
    # 60 chars total / 4 chars-per-token = 15
    assert estimate_tokens_returned(rr) == 15


def test_estimate_tokens_returned_zero_when_no_content():
    rr = RankedResult(
        query_id="q1",
        ranked_doc_ids=("a.py:1-2",),
        ranked_files=("a.py",),
        raw=({"file": "a.py", "start_line": 1, "end_line": 2},),
    )
    assert estimate_tokens_returned(rr) == 0


def test_estimate_tokens_returned_zero_for_empty_raw():
    rr = RankedResult(query_id="q1", ranked_doc_ids=(), ranked_files=())
    assert estimate_tokens_returned(rr) == 0
