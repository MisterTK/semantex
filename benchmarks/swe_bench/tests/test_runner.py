from pathlib import Path
from unittest.mock import MagicMock, patch

from swe_bench_harness.conditions import Condition
from swe_bench_harness.dataset import Instance
from swe_bench_harness.runner import RunResult, run_one


def _make_condition(**overrides) -> Condition:
    defaults = dict(
        id="c1_baseline",
        label="x",
        agent_model="claude-sonnet-4-6",
        semantex_enabled=False,
        semantex_features="",
        semantex_llm_features=False,
    )
    defaults.update(overrides)
    return Condition(**defaults)


def _make_instance() -> Instance:
    return Instance(
        instance_id="astropy__astropy-12907",
        repo="astropy/astropy",
        base_commit="d16bfe05a744909de4b27f5875fe0d4ed41ce607",
        problem_statement="...",
    )


def test_run_one_returns_patch_and_metrics(tmp_path):
    inst = _make_instance()
    cond = _make_condition()
    fake_openhands_result = MagicMock(
        patch="diff --git a/foo.py b/foo.py\n...",
        turns=[
            {"input_tokens": 1000, "output_tokens": 50,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
             "tool_calls": ["terminal"]},
            {"input_tokens": 200, "output_tokens": 80,
             "cache_creation_input_tokens": 900, "cache_read_input_tokens": 0,
             "tool_calls": ["file_editor"]},
        ],
    )
    # Make sure the repo dir exists so run_one doesn't bail early on path check
    (tmp_path / inst.instance_id).mkdir(parents=True)
    with patch("swe_bench_harness.runner._invoke_openhands", return_value=fake_openhands_result):
        result = run_one(
            instance=inst, condition=cond, replicate=0,
            repo_cache_root=tmp_path, max_turns=50,
        )
    assert isinstance(result, RunResult)
    assert result.instance_id == inst.instance_id
    assert result.condition_id == cond.id
    assert result.replicate == 0
    assert result.patch.startswith("diff --git")
    assert len(result.turns) == 2
    assert result.turns[0]["input_tokens"] == 1000
    assert result.turns[1]["tool_calls"] == ["file_editor"]
    assert result.wall_clock_secs > 0
    assert result.error == ""


def test_run_one_records_failure_without_raising(tmp_path):
    inst = _make_instance()
    cond = _make_condition()
    (tmp_path / inst.instance_id).mkdir(parents=True)
    with patch(
        "swe_bench_harness.runner._invoke_openhands",
        side_effect=RuntimeError("boom"),
    ):
        result = run_one(
            instance=inst, condition=cond, replicate=0,
            repo_cache_root=tmp_path, max_turns=50,
        )
    assert result.error == "boom"
    assert result.patch == ""
    assert result.turns == []


def test_run_one_includes_condition_id_in_result(tmp_path):
    inst = _make_instance()
    cond = _make_condition(id="c3_semantex_with_llm")
    (tmp_path / inst.instance_id).mkdir(parents=True)
    fake = MagicMock(patch="", turns=[])
    with patch("swe_bench_harness.runner._invoke_openhands", return_value=fake):
        result = run_one(
            instance=inst, condition=cond, replicate=2,
            repo_cache_root=tmp_path, max_turns=10,
        )
    assert result.condition_id == "c3_semantex_with_llm"
    assert result.replicate == 2
