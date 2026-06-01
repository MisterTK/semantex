import json
from unittest.mock import patch

import pytest

from relevance_harness.indexing import ensure_index, index_is_complete


def _write_meta(corpus_dir, chunk_count):
    sx = corpus_dir / ".semantex"
    sx.mkdir(parents=True, exist_ok=True)
    (sx / "meta.json").write_text(json.dumps({"chunk_count": chunk_count}))


def test_index_is_complete_true_with_positive_chunk_count(tmp_path):
    _write_meta(tmp_path, 42)
    assert index_is_complete(tmp_path) is True


def test_index_is_complete_false_when_missing(tmp_path):
    assert index_is_complete(tmp_path) is False


def test_index_is_complete_false_with_zero_chunks(tmp_path):
    _write_meta(tmp_path, 0)
    assert index_is_complete(tmp_path) is False


def test_ensure_index_skips_when_already_complete(tmp_path):
    _write_meta(tmp_path, 10)
    with patch("relevance_harness.indexing.index_repo") as mr:
        ensure_index(corpus_dir=tmp_path, semantex_binary="semantex")
    mr.assert_not_called()


def test_ensure_index_builds_when_incomplete(tmp_path):
    from swe_bench_harness.indexer import IndexResult
    with patch("relevance_harness.indexing.index_repo") as mr:
        mr.return_value = IndexResult(ok=True, path=tmp_path / ".semantex")
        ensure_index(corpus_dir=tmp_path, semantex_binary="semantex")
    mr.assert_called_once()
    assert mr.call_args.kwargs["repo_path"] == tmp_path


def test_ensure_index_raises_on_failed_build(tmp_path):
    from swe_bench_harness.indexer import IndexResult
    with patch("relevance_harness.indexing.index_repo") as mr:
        mr.return_value = IndexResult(ok=False, path=tmp_path / ".semantex", error="kaboom")
        with pytest.raises(RuntimeError, match="kaboom"):
            ensure_index(corpus_dir=tmp_path, semantex_binary="semantex")
