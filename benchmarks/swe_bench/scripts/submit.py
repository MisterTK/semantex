"""Full post-run pipeline: build predictions per (condition, replicate),
invoke swebench evaluator, join with run records, write report + leaderboard JSON.

Usage:
  python -m scripts.submit --run-id 20260528-120000-phase_a
  python -m scripts.submit --run-id <id> --baseline c1_baseline --system-name 'my-system'
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import yaml

from swe_bench_harness.analyze import build_unit_table
from swe_bench_harness.conditions import load_conditions
from swe_bench_harness.evaluator import build_predictions_file, run_eval
from swe_bench_harness.report import (
    leaderboard_submission_dict, render_markdown_report, summarize_by_condition,
)
from swe_bench_harness.stats import mcnemar_paired_test


CONFIG = Path(__file__).parent.parent / "config"
RESULTS = Path(__file__).parent.parent / "results"


@click.command()
@click.option("--run-id", required=True, help="run_id directory name under results/")
@click.option("--baseline", default="c1_baseline", show_default=True,
              help="Condition treated as the baseline for paired comparisons")
@click.option("--system-name", default="semantex+OpenHands+Sonnet-4.6", show_default=True,
              help="Identifier embedded in predictions file model_name_or_path")
def main(run_id: str, baseline: str, system_name: str):
    run_dir = RESULTS / run_id
    runs_dir = run_dir / "runs"
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True, parents=True)

    # 1. determine (condition_id, replicate) pairs present in runs_dir.
    # Filename format: {instance_id}__{condition_id}__{replicate}.json — instance_ids
    # themselves contain '__' (e.g. astropy__astropy-13033), so rsplit from the right.
    pairs = sorted({
        (jf.stem.rsplit("__", 2)[1], int(jf.stem.rsplit("__", 2)[2]))
        for jf in runs_dir.glob("*.json")
    })

    # 2. for each pair: build preds, run swebench eval (Docker), copy report
    eval_reports: dict[str, dict] = {}
    for cond_id, rep in pairs:
        preds = build_predictions_file(
            runs_dir=runs_dir, condition_id=cond_id, replicate=rep,
            out_path=eval_dir / f"preds_{cond_id}_{rep}.jsonl",
            model_name=f"{system_name}+{cond_id}+rep{rep}",
        )
        report_path = run_eval(
            predictions_path=preds,
            run_id=f"{run_id}_{cond_id}_{rep}",
            eval_cwd=eval_dir,
        )
        dest = eval_dir / f"report_{cond_id}_{rep}.json"
        dest.write_text(report_path.read_text())
        eval_reports[f"{cond_id}__{rep}"] = json.loads(dest.read_text())

    # 3. tidy DataFrame
    pricing = yaml.safe_load((CONFIG / "models.yaml").read_text())
    df = build_unit_table(
        runs_dir=runs_dir, eval_reports=eval_reports,
        agent_model="claude-sonnet-4-6", pricing=pricing,
    )
    df.to_csv(run_dir / "units.csv", index=False)

    # 4. per-condition summary
    summary = summarize_by_condition(df)
    summary.to_csv(run_dir / "summary.csv", index=False)

    # 5. paired McNemar tests against baseline
    all_instance_ids = sorted(df.instance_id.unique().tolist())
    baseline_resolved = set(df[(df.condition_id == baseline) & df.resolved].instance_id)
    paired_tests = []
    for c in sorted(df.condition_id.unique()):
        if c == baseline:
            continue
        treatment_resolved = set(df[(df.condition_id == c) & df.resolved].instance_id)
        t = mcnemar_paired_test(
            baseline_resolved=baseline_resolved,
            treatment_resolved=treatment_resolved,
            all_instances=all_instance_ids,
        )
        t["treatment"] = c
        t["baseline"] = baseline
        paired_tests.append(t)
    (run_dir / "paired_tests.json").write_text(json.dumps(paired_tests, indent=2))

    # 6. markdown report
    md = render_markdown_report(df=df, summary=summary, paired_tests=paired_tests)
    (run_dir / "report.md").write_text(md)

    # 7. leaderboard submission JSON per (condition, replicate)
    leaderboard_dir = run_dir / "leaderboard"
    leaderboard_dir.mkdir(exist_ok=True)
    for c, r in pairs:
        out = leaderboard_submission_dict(
            df=df, condition_id=c, replicate=r,
            system_name=f"{system_name}+{c}+rep{r}",
        )
        (leaderboard_dir / f"{c}_rep{r}.json").write_text(json.dumps(out, indent=2))

    print(f"Done. Report: {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
