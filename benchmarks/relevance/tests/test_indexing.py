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


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_ensure_index_skips_when_already_complete(tmp_path):
    _write_meta(tmp_path, 10)
    with patch("relevance_harness.indexing.subprocess.run") as mr:
        build = ensure_index(corpus_dir=tmp_path, semantex_binary="semantex")
    mr.assert_not_called()
    assert build.built is False


def test_ensure_index_builds_when_incomplete(tmp_path):
    with patch("relevance_harness.indexing.subprocess.run") as mr:
        mr.return_value = _Proc(returncode=0)
        build = ensure_index(corpus_dir=tmp_path, semantex_binary="semantex")
    mr.assert_called_once()
    assert mr.call_args.kwargs["cwd"] == tmp_path
    assert build.built is True


def test_ensure_index_threads_embedder_into_index_env(tmp_path):
    """The embedder MUST be set at INDEX time (backend is persisted in meta.json
    and authoritative at search time)."""
    with patch("relevance_harness.indexing.subprocess.run") as mr:
        mr.return_value = _Proc(returncode=0)
        ensure_index(corpus_dir=tmp_path, semantex_binary="semantex",
                     embedder="coderank-137m")
    assert mr.call_args.kwargs["env"]["SEMANTEX_EMBEDDER"] == "coderank-137m"


def test_index_env_locks_adaptive_sizing_off_by_default(tmp_path, monkeypatch):
    # The index subprocess inherits the same canonical A/B lock as search, so the
    # daemon it (or the first search) spawns is adaptive-OFF. Harmless at index
    # time; keeps the harness env uniform. See the rootcause doc §2.
    monkeypatch.delenv("SEMANTEX_ADAPTIVE_SIZING", raising=False)
    with patch("relevance_harness.indexing.subprocess.run") as mr:
        mr.return_value = _Proc(returncode=0)
        ensure_index(corpus_dir=tmp_path, semantex_binary="semantex")
    assert mr.call_args.kwargs["env"]["SEMANTEX_ADAPTIVE_SIZING"] == "0"


def test_index_env_respects_explicit_adaptive_sizing_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTEX_ADAPTIVE_SIZING", "1")
    with patch("relevance_harness.indexing.subprocess.run") as mr:
        mr.return_value = _Proc(returncode=0)
        ensure_index(corpus_dir=tmp_path, semantex_binary="semantex")
    assert mr.call_args.kwargs["env"]["SEMANTEX_ADAPTIVE_SIZING"] == "1"


def test_ensure_index_raises_on_failed_build(tmp_path):
    with patch("relevance_harness.indexing.subprocess.run") as mr:
        mr.return_value = _Proc(returncode=1, stderr="kaboom")
        with pytest.raises(RuntimeError, match="kaboom"):
            ensure_index(corpus_dir=tmp_path, semantex_binary="semantex")
