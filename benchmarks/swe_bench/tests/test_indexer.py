import subprocess
from unittest.mock import patch

import pytest

from swe_bench_harness.indexer import IndexResult, index_repo


def test_index_invokes_semantex_cli(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "main.py").write_text("def f(): pass\n")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        # also simulate the .semantex/ dir being created
        (repo / ".semantex").mkdir()
        result = index_repo(repo_path=repo, semantex_binary="semantex", timeout_secs=60)
    assert result.ok
    assert result.path == repo / ".semantex"
    assert mock_run.call_args.kwargs["timeout"] == 60
    args = mock_run.call_args.args[0]
    assert args[0] == "semantex"
    assert args[1] == "index"


def test_index_timeout_marks_failure(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("semantex", 1)):
        result = index_repo(repo_path=repo, semantex_binary="semantex", timeout_secs=1)
    assert not result.ok
    assert "timeout" in result.error.lower()


def test_index_nonzero_exit_marks_failure(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=2, stdout="", stderr="boom"
        )
        result = index_repo(repo_path=repo, semantex_binary="semantex", timeout_secs=60)
    assert not result.ok
    assert "boom" in result.error
