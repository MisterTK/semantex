"""Wrap the official SWE-bench harness for patch evaluation.

The harness consumes JSONL of {instance_id, model_name_or_path, model_patch}
and runs each instance's tests in a Docker container at the patched state.

IMPORTANT (research finding F): swebench's --report_dir is unreliable in
normal eval mode. We invoke with cwd= set to our chosen output dir, then
locate the aggregate report by glob in that dir."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalResult:
    resolved: frozenset[str]
    unresolved: frozenset[str]
    errored: frozenset[str]
    submitted: frozenset[str]

    @property
    def resolution_rate(self) -> float:
        if not self.submitted:
            return 0.0
        return len(self.resolved) / len(self.submitted)


def build_predictions_file(
    *,
    runs_dir: Path,
    condition_id: str,
    replicate: int,
    out_path: Path,
    model_name: str,
) -> Path:
    """Aggregate per-unit JSON files into the swebench-expected JSONL.

    Skips units with non-empty `error` (no patch to evaluate)."""
    rows = []
    for jf in sorted(runs_dir.glob(f"*__{condition_id}__{replicate}.json")):
        data = json.loads(jf.read_text())
        if data.get("error"):
            continue
        rows.append({
            "instance_id": data["instance_id"],
            "model_name_or_path": model_name,
            "model_patch": data.get("patch", ""),
        })
    out_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return out_path


def run_eval(
    *,
    predictions_path: Path,
    run_id: str,
    eval_cwd: Path,
    max_workers: int = 4,
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
) -> Path:
    """Invoke swebench's evaluator with cwd=eval_cwd. Returns the aggregate report path.

    Requires Docker. Per research finding F, --report_dir is ignored; we use cwd
    and then locate the aggregate report by glob."""
    eval_cwd.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "-m", "swebench.harness.run_evaluation",
        "--predictions_path", str(predictions_path),
        "--max_workers", str(max_workers),
        "--run_id", run_id,
        "--dataset_name", dataset_name,
    ]
    subprocess.run(cmd, check=True, cwd=eval_cwd)
    # Aggregate report lands at eval_cwd/{model}.{run_id}.json
    matches = list(eval_cwd.glob(f"*.{run_id}.json"))
    if not matches:
        raise FileNotFoundError(
            f"swebench report for run_id={run_id} not found in {eval_cwd}"
        )
    return matches[0]


def parse_eval_report(report_path: Path) -> EvalResult:
    data = json.loads(report_path.read_text())
    submitted = set(data.get("submitted_ids", []))
    resolved = set(data.get("resolved_ids", []))
    unresolved = set(data.get("unresolved_ids", []))
    errored = set(data.get("error_ids", []))
    # If submitted_ids missing, infer from union
    if not submitted:
        submitted = resolved | unresolved | errored
    return EvalResult(
        resolved=frozenset(resolved),
        unresolved=frozenset(unresolved),
        errored=frozenset(errored),
        submitted=frozenset(submitted),
    )
