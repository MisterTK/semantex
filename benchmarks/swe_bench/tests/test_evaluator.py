import json
from pathlib import Path
from unittest.mock import patch

from swe_bench_harness.evaluator import (
    EvalResult, build_predictions_file, parse_eval_report,
)


def test_build_predictions_writes_jsonl(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (runs_dir / "r0__c1__0.json").write_text(json.dumps({
        "instance_id": "r0", "condition_id": "c1", "replicate": 0,
        "patch": "diff --git a/x b/x\n",
    }))
    (runs_dir / "r1__c1__0.json").write_text(json.dumps({
        "instance_id": "r1", "condition_id": "c1", "replicate": 0,
        "patch": "diff --git a/y b/y\n",
    }))
    out = build_predictions_file(
        runs_dir=runs_dir, condition_id="c1", replicate=0, out_path=tmp_path / "preds.jsonl",
        model_name="claude-sonnet-4-6+c1+rep0",
    )
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(ln) for ln in lines]
    assert {p["instance_id"] for p in parsed} == {"r0", "r1"}
    assert all(p["model_name_or_path"] == "claude-sonnet-4-6+c1+rep0" for p in parsed)


def test_build_predictions_skips_errored_runs(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    # one good unit, one errored unit — only good should appear in preds
    (runs_dir / "r0__c1__0.json").write_text(json.dumps({
        "instance_id": "r0", "condition_id": "c1", "replicate": 0,
        "patch": "diff --git a/x b/x\n", "error": "",
    }))
    (runs_dir / "r1__c1__0.json").write_text(json.dumps({
        "instance_id": "r1", "condition_id": "c1", "replicate": 0,
        "patch": "", "error": "ANTHROPIC_API_KEY missing",
    }))
    out = build_predictions_file(
        runs_dir=runs_dir, condition_id="c1", replicate=0, out_path=tmp_path / "preds.jsonl",
        model_name="m",
    )
    parsed = [json.loads(ln) for ln in out.read_text().strip().splitlines()]
    assert {p["instance_id"] for p in parsed} == {"r0"}


def test_parse_eval_report_extracts_resolution_rate(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "resolved_ids": ["r0", "r2"],
        "unresolved_ids": ["r1"],
        "error_ids": [],
        "submitted_ids": ["r0", "r1", "r2"],
    }))
    result = parse_eval_report(report)
    assert isinstance(result, EvalResult)
    assert result.resolved == {"r0", "r2"}
    assert result.unresolved == {"r1"}
    assert abs(result.resolution_rate - 2 / 3) < 1e-9
