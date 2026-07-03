"""Orchestration tests for swe_loc_runner.run_instance, fully mocked (no
semantex binary, no ripgrep, no daemon, no network) so they run in any
CI-less environment. See test_runner_e2e.py / the tiny_corpus fixture for the
real end-to-end (skipif semantex/rg missing) smoke coverage.
"""
from unittest.mock import MagicMock, patch

from relevance_harness.swe_loc_runner import ARMS, relevance_vector, run_instance
from relevance_harness.types import Query, RankedResult


def _rr(files):
    return RankedResult(query_id="q", ranked_doc_ids=tuple(f"{f}:1-2" for f in files),
                         ranked_files=tuple(files))


def test_relevance_vector_marks_gold_hits():
    assert relevance_vector(("a.py", "b.py", "c.py"), frozenset({"b.py"})) == [0, 1, 0]


def test_run_instance_runs_all_four_arms(tmp_path):
    query = Query(query_id="demo__demo-1", text="fix the login bug", gold_doc_ids=("auth.py",))

    with patch("relevance_harness.swe_loc_runner.ensure_index") as m_index, \
         patch("relevance_harness.swe_loc_runner.SemantexClient") as m_client_cls, \
         patch("relevance_harness.swe_loc_runner.reset_daemon") as m_reset, \
         patch("relevance_harness.swe_loc_runner.start_daemon") as m_start, \
         patch("relevance_harness.swe_loc_runner.rank_files_by_keyword_hits") as m_rg:
        client = MagicMock()
        client.search.side_effect = [_rr(["auth.py", "util.py"]), _rr(["db.py"])]
        client.search_agent_auto.return_value = _rr(["auth.py"])
        m_client_cls.return_value = client
        m_start.return_value = MagicMock(poll=MagicMock(return_value=None))
        m_rg.return_value = ["auth.py", "db.py"]

        results = run_instance(query, corpus_dir=tmp_path, semantex_binary="semantex", k=10)

    m_index.assert_called_once()
    assert {r.arm for r in results} == set(ARMS)
    by_arm = {r.arm: r for r in results}
    assert by_arm["hybrid"].ranked_files == ("auth.py", "util.py")
    assert by_arm["sparse-only"].ranked_files == ("db.py",)
    assert by_arm["agent-routed"].ranked_files == ("auth.py",)
    assert by_arm["ripgrep"].ranked_files == ("auth.py", "db.py")
    assert all(r.error is None for r in results)
    # agent-routed needs its own daemon lifecycle: reset (x2: pre-emptive +
    # post-run cleanup) then a fresh start.
    assert m_reset.call_count >= 1
    m_start.assert_called_once()


def test_run_instance_records_per_arm_errors_without_aborting(tmp_path):
    query = Query(query_id="demo__demo-2", text="connection pool leaks", gold_doc_ids=("db.py",))

    with patch("relevance_harness.swe_loc_runner.ensure_index"), \
         patch("relevance_harness.swe_loc_runner.SemantexClient") as m_client_cls, \
         patch("relevance_harness.swe_loc_runner.reset_daemon"), \
         patch("relevance_harness.swe_loc_runner.start_daemon", side_effect=RuntimeError("no daemon")), \
         patch("relevance_harness.swe_loc_runner.rank_files_by_keyword_hits",
               side_effect=FileNotFoundError("rg missing")):
        client = MagicMock()
        client.search.side_effect = [_rr(["db.py"]), RuntimeError("index missing")]
        m_client_cls.return_value = client

        results = run_instance(query, corpus_dir=tmp_path, semantex_binary="semantex", k=10)

    by_arm = {r.arm: r for r in results}
    assert by_arm["hybrid"].error is None
    assert by_arm["hybrid"].ranked_files == ("db.py",)
    assert by_arm["sparse-only"].error == "index missing"
    assert by_arm["sparse-only"].ranked_files == ()
    assert by_arm["agent-routed"].error == "no daemon"
    assert by_arm["ripgrep"].error == "rg missing"
    # every arm is still represented -- a failure never drops a row
    assert {r.arm for r in results} == set(ARMS)
