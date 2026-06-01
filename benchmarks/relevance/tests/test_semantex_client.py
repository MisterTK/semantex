import json
import subprocess
from unittest.mock import patch

from relevance_harness.semantex_client import SemantexClient, parse_results


SAMPLE_JSON = json.dumps([
    {"file": "auth.py", "start_line": 10, "end_line": 20, "score": 0.42,
     "source": "Hybrid", "chunk_type": "AstNode", "name": "login", "language": "python"},
    {"file": "db.py", "start_line": 1, "end_line": 9, "score": 0.20,
     "source": "Sparse", "chunk_type": "TextWindow"},
])


def test_parse_results_extracts_files_and_doc_ids():
    rr = parse_results("q1", SAMPLE_JSON)
    assert rr.query_id == "q1"
    assert rr.ranked_files == ("auth.py", "db.py")
    # doc id is "file:start-end" — stable, matches how loaders mint ids
    assert rr.ranked_doc_ids == ("auth.py:10-20", "db.py:1-9")


def test_parse_results_empty_array():
    rr = parse_results("q1", "[]")
    assert rr.ranked_doc_ids == ()
    assert rr.ranked_files == ()


def test_search_builds_hybrid_command_and_env():
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/corpus")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(args=[], returncode=0,
                                                      stdout=SAMPLE_JSON, stderr="")
        rr = client.search("q1", "auth handler", ablation="hybrid", k=10)
    args = mr.call_args.args[0]
    assert args[0] == "semantex"
    assert "auth handler" in args
    assert "--json" in args and "--no-content" in args
    assert "-m" in args and "10" in args
    # hybrid uses NEITHER dense-only nor sparse-only
    assert "--dense-only" not in args and "--sparse-only" not in args
    assert "--rerank" not in args
    assert mr.call_args.kwargs["cwd"] == "/tmp/corpus"
    assert rr.ranked_files == ("auth.py", "db.py")


def test_search_sparse_only_flag():
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/c")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="[]", stderr="")
        client.search("q1", "x", ablation="sparse-only", k=5)
    assert "--sparse-only" in mr.call_args.args[0]


def test_search_dense_only_flag():
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/c")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="[]", stderr="")
        client.search("q1", "x", ablation="dense-only", k=5)
    assert "--dense-only" in mr.call_args.args[0]


def test_search_rerank_adds_flag_on_hybrid():
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/c")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="[]", stderr="")
        client.search("q1", "x", ablation="rerank", k=5)
    args = mr.call_args.args[0]
    assert "--rerank" in args
    assert "--dense-only" not in args and "--sparse-only" not in args


def test_embedder_sets_env():
    client = SemantexClient(
        semantex_binary="semantex", corpus_dir="/tmp/c", embedder="coderank-137m"
    )
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="[]", stderr="")
        client.search("q1", "x", ablation="hybrid", k=5)
    env = mr.call_args.kwargs["env"]
    # SEMANTEX_EMBEDDER is canonical (integration §4 D-env-knob).
    assert env["SEMANTEX_EMBEDDER"] == "coderank-137m"


def test_failed_search_raises_with_stderr():
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/c")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(args=[], returncode=3, stdout="", stderr="boom")
        try:
            client.search("q1", "x", ablation="hybrid", k=5)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "boom" in str(e)
