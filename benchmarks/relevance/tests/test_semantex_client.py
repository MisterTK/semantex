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
    # the query is passed AFTER a `--` end-of-options separator, and is the last
    # arg, so a query that starts with dashes is never parsed as a CLI flag.
    assert "--" in args
    assert args[-1] == "auth handler"
    assert args.index("--") < args.index("auth handler")


def test_search_dash_leading_query_not_parsed_as_flag():
    # real CSN docstrings can start with dashes (e.g. "---- utils ----"); they
    # must reach semantex as the positional query, guarded by `--`.
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/c")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="[]", stderr="")
        client.search("q1", "---- utils ----", ablation="hybrid", k=5)
    args = mr.call_args.args[0]
    assert args[-1] == "---- utils ----"
    assert args[args.index("--") + 1] == "---- utils ----"


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


def test_search_env_locks_adaptive_sizing_off_by_default(monkeypatch):
    # Canonical A/B measurement config: adaptive result sizing is OFF for
    # relevance A/Bs because it clips ~45% of recoverable recall (confidence
    # threshold + per-file dedup) before any feature runs, invalidating the
    # comparison. It stays ON in the product (the -18% agent-CCB feature). See
    # docs/superpowers/plans/2026-06-01-why-no-feature-uplift-rootcause.md §2.
    monkeypatch.delenv("SEMANTEX_ADAPTIVE_SIZING", raising=False)
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/c")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="[]", stderr="")
        client.search("q1", "x", ablation="hybrid", k=5)
    assert mr.call_args.kwargs["env"]["SEMANTEX_ADAPTIVE_SIZING"] == "0"


def test_search_env_respects_explicit_adaptive_sizing_override(monkeypatch):
    # An explicit export wins over the lock, so the harness can still measure the
    # shipped adaptive-ON behaviour (e.g. to reproduce the OFF-vs-ON delta).
    monkeypatch.setenv("SEMANTEX_ADAPTIVE_SIZING", "1")
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/c")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="[]", stderr="")
        client.search("q1", "x", ablation="hybrid", k=5)
    assert mr.call_args.kwargs["env"]["SEMANTEX_ADAPTIVE_SIZING"] == "1"


def test_reset_daemon_runs_stop_in_corpus_dir(monkeypatch):
    # The daemon caches adaptive_sizing at spawn time and lives 30 min idle, so a
    # stale adaptive-ON daemon from a prior run would silently serve A/B queries.
    # reset_daemon stops it so the next search spawns a fresh one under the lock.
    monkeypatch.delenv("SEMANTEX_ADAPTIVE_SIZING", raising=False)
    client = SemantexClient(semantex_binary="/abs/semantex", corpus_dir="/tmp/c")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        client.reset_daemon()
    assert mr.call_args.args[0] == ["/abs/semantex", "stop", "."]
    assert mr.call_args.kwargs["cwd"] == "/tmp/c"
    # never raises even when no daemon is running (stop is best-effort)
    assert mr.call_args.kwargs["check"] is False
    # carries the locked env so a respawn inherits the canonical A/B config
    assert mr.call_args.kwargs["env"]["SEMANTEX_ADAPTIVE_SIZING"] == "0"


def test_failed_search_raises_with_stderr():
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/c")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(args=[], returncode=3, stdout="", stderr="boom")
        try:
            client.search("q1", "x", ablation="hybrid", k=5)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "boom" in str(e)


# --- route-stress: forced-route measurement seam -----------------------------

# The agent `--json-hits` path returns the SAME JSON array shape as
# `search --json`, so parse_results / the file-level gold matcher are unchanged.
ROUTE_HITS_JSON = json.dumps([
    {"file": "routergroup.go", "start_line": 30, "end_line": 45, "score": 0.09,
     "source": "Hybrid", "chunk_type": "AstNode", "name": "Group", "language": "go"},
    {"file": "ginS/gins.go", "start_line": 1, "end_line": 9, "score": 0.06,
     "source": "Hybrid", "chunk_type": "AstNode", "name": "Group"},
])


def test_route_forces_agent_json_hits_command():
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/c")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=ROUTE_HITS_JSON, stderr=""
        )
        rr = client.search("q1", "router group", ablation="hybrid", k=10, route="semantic")
    args = mr.call_args.args[0]
    # Forced-route path uses `agent --route <name> --json-hits`, NOT `search`.
    assert args[:2] == ["semantex", "agent"]
    assert "--route" in args and args[args.index("--route") + 1] == "semantic"
    assert "--json-hits" in args
    # ablation flags are NOT passed in route mode (route owns mechanism choice).
    assert "--dense-only" not in args and "--sparse-only" not in args
    # query is the LAST arg, behind a `--` separator (dash-leading-safe).
    assert args[-1] == "router group"
    assert args[args.index("--") + 1] == "router group"
    # the result shape is identical to the search path — files are repo-relative.
    assert rr.ranked_files == ("routergroup.go", "ginS/gins.go")
    assert rr.ranked_doc_ids == ("routergroup.go:30-45", "ginS/gins.go:1-9")


def test_route_none_keeps_search_path_unchanged():
    # Regression guard: the routeless default must still hit `search --json`.
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/c")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="[]", stderr=""
        )
        client.search("q1", "x", ablation="hybrid", k=5)
    args = mr.call_args.args[0]
    assert args[0] == "semantex" and "agent" not in args
    assert "--json" in args and "--json-hits" not in args


def test_route_alias_files_is_accepted():
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/c")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="[]", stderr=""
        )
        client.search("q1", "**/*_test.go", ablation="hybrid", k=5, route="files")
    args = mr.call_args.args[0]
    assert args[args.index("--route") + 1] == "files"


def test_unknown_route_raises():
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/c")
    try:
        client.search("q1", "x", ablation="hybrid", k=5, route="deep")
        assert False, "synthesis route 'deep' must be rejected (prose-only)"
    except ValueError as e:
        assert "deep" in str(e)


def test_search_with_content_omits_no_content_flag():
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/c")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=SAMPLE_JSON, stderr="")
        client.search("q1", "x", ablation="hybrid", k=5, with_content=True)
    assert "--no-content" not in mr.call_args.args[0]


def test_search_default_still_omits_content():
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/c")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=SAMPLE_JSON, stderr="")
        client.search("q1", "x", ablation="hybrid", k=5)
    assert "--no-content" in mr.call_args.args[0]


def test_parse_results_stashes_raw_dicts():
    rr = parse_results("q1", SAMPLE_JSON)
    assert len(rr.raw) == 2
    assert rr.raw[0]["file"] == "auth.py"


def test_search_agent_auto_builds_no_route_command():
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/c")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=SAMPLE_JSON, stderr="")
        rr = client.search_agent_auto("q1", "auth handler")
    args = mr.call_args.args[0]
    assert args == ["semantex", "agent", "--json-hits", "--", "auth handler"]
    assert "--route" not in args
    assert rr.ranked_files == ("auth.py", "db.py")


def test_route_failure_raises_with_stderr():
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/c")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="daemon not running"
        )
        try:
            client.search("q1", "x", ablation="hybrid", k=5, route="semantic")
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "daemon not running" in str(e)
            assert "json-hits" in str(e)
