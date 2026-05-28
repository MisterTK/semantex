from pathlib import Path
from unittest.mock import patch

from swe_bench_harness.conditions import Condition
from swe_bench_harness.dataset import Instance
from swe_bench_harness.orchestrator import iter_units, run_all
from swe_bench_harness.runner import RunResult


def _inst(i: int) -> Instance:
    return Instance(
        instance_id=f"r{i}", repo="o/r", base_commit="0"*40, problem_statement="x",
    )


def _cond(name: str) -> Condition:
    return Condition(
        id=name, label=name, agent_model="claude-sonnet-4-6",
        semantex_enabled=False, semantex_features="", semantex_llm_features=False,
    )


def test_iter_units_is_paired_and_ordered():
    insts = [_inst(0), _inst(1)]
    conds = [_cond("c1"), _cond("c2")]
    units = list(iter_units(insts, conds, replicates=2))
    assert len(units) == 8  # 2 inst * 2 cond * 2 rep
    triples = {(u.instance.instance_id, u.condition.id, u.replicate) for u in units}
    assert len(triples) == 8


def test_run_all_writes_one_json_per_unit(tmp_path):
    insts = [_inst(0)]
    conds = [_cond("c1"), _cond("c2")]
    out_dir = tmp_path / "run_x"
    with patch(
        "swe_bench_harness.orchestrator.run_one",
        side_effect=lambda **kw: RunResult(
            instance_id=kw["instance"].instance_id,
            condition_id=kw["condition"].id,
            replicate=kw["replicate"],
            patch="diff",
        ),
    ):
        run_all(
            instances=insts, conditions=conds, replicates=2,
            out_dir=out_dir, repo_cache_root=tmp_path,
            workers=1, max_turns=10,
        )
    files = sorted(out_dir.glob("*.json"))
    assert len(files) == 4


def test_run_all_skips_existing_outputs(tmp_path):
    insts = [_inst(0)]
    conds = [_cond("c1")]
    out_dir = tmp_path / "run_x"
    out_dir.mkdir()
    # pre-create the result file for (r0, c1, 0)
    (out_dir / "r0__c1__0.json").write_text("{}")
    with patch("swe_bench_harness.orchestrator.run_one") as mr:
        run_all(
            instances=insts, conditions=conds, replicates=1,
            out_dir=out_dir, repo_cache_root=tmp_path,
            workers=1, max_turns=10,
        )
    mr.assert_not_called()
