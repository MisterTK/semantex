"""Compare each non-baseline condition against the baseline on a run.

Emits a markdown table per (treatment, baseline) pair covering the 7
semantex-impact questions: resolution-rate lift (McNemar), semantex tool
usage, turn / CCB / cost paired deltas, and a hard-instance subset cut.

Outputs to stdout AND writes `comparison.md` next to `report.md`.
Run on the same `--run-id` that `submit.py` evaluated.

Usage:
  python -m scripts.compare_conditions --run-id 20260529-054239-phase_a_c1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import yaml

from swe_bench_harness.analyze import build_unit_table
from swe_bench_harness.stats import mcnemar_paired_test, paired_bootstrap_diff


# The MCP tools exposed by `semantex mcp` (per crates/semantex-mcp).
SEMANTEX_TOOLS = frozenset({
    "semantex",
    "semantex_agent",
    "semantex_search",
    "semantex_deep",
    "semantex_index",
    "semantex_status",
    "semantex_health",
    "semantex_validate",
})

CONFIG_DIR = Path(__file__).parent.parent / "config"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def _load_eval_reports(eval_dir: Path) -> dict[str, dict]:
    """Load `report_<cond_id>_<rep>.json` keyed by `<cond_id>__<rep>`."""
    reports: dict[str, dict] = {}
    for f in eval_dir.glob("report_*.json"):
        key = f.stem[len("report_"):]  # e.g. "c2_semantex_no_llm_0"
        cond_id, rep = key.rsplit("_", 1)
        reports[f"{cond_id}__{rep}"] = json.loads(f.read_text())
    return reports


def _semantex_calls(tool_distribution: dict) -> int:
    return sum(n for tool, n in (tool_distribution or {}).items() if tool in SEMANTEX_TOOLS)


def _used_semantex(tool_distribution: dict) -> bool:
    return _semantex_calls(tool_distribution) > 0


def compare(df: pd.DataFrame, baseline: str, treatment: str) -> dict:
    """Compute the 7 deltas between treatment and baseline, paired by instance_id."""
    base = df[df.condition_id == baseline].set_index("instance_id")
    treat = df[df.condition_id == treatment].set_index("instance_id")
    common = sorted(base.index.intersection(treat.index))
    if not common:
        return {"treatment": treatment, "baseline": baseline, "n_paired_instances": 0}

    base_p = base.loc[common]
    treat_p = treat.loc[common]

    # 1. Resolution rate (McNemar exact)
    base_resolved = set(base_p[base_p.resolved].index)
    treat_resolved = set(treat_p[treat_p.resolved].index)
    mcn = mcnemar_paired_test(
        baseline_resolved=base_resolved,
        treatment_resolved=treat_resolved,
        all_instances=common,
    )

    # 2. Did the agent use semantex when offered?
    used = treat_p.tool_distribution.apply(_used_semantex)
    calls = treat_p.tool_distribution.apply(_semantex_calls)

    # 3-5. Paired bootstrap on turns / CCB / cost
    turns_d = paired_bootstrap_diff(
        a=treat_p.num_turns.to_numpy(dtype=float),
        b=base_p.num_turns.to_numpy(dtype=float),
        n_resamples=5000, seed=42,
    )
    ccb_d = paired_bootstrap_diff(
        a=treat_p.ccb.to_numpy(dtype=float),
        b=base_p.ccb.to_numpy(dtype=float),
        n_resamples=5000, seed=42,
    )
    cost_d = paired_bootstrap_diff(
        a=treat_p.cost_usd.to_numpy(dtype=float),
        b=base_p.cost_usd.to_numpy(dtype=float),
        n_resamples=5000, seed=42,
    )

    # 6. Resolved per dollar (cost-effectiveness)
    base_rpd = base_p.resolved.sum() / max(base_p.cost_usd.sum(), 1e-9)
    treat_rpd = treat_p.resolved.sum() / max(treat_p.cost_usd.sum(), 1e-9)

    # 7. Hard subset: baseline needed >20 turns OR baseline failed.
    hard_mask = (base_p.num_turns > 20) | (~base_p.resolved)
    hard_ids = base_p[hard_mask].index.tolist()
    hard_mcn = (
        mcnemar_paired_test(
            baseline_resolved={i for i in base.loc[hard_ids].index if base.loc[i].resolved},
            treatment_resolved={i for i in treat.loc[hard_ids].index if treat.loc[i].resolved},
            all_instances=hard_ids,
        )
        if hard_ids
        else {"b": 0, "c": 0, "p_value": 1.0, "treatment_lift_pp": 0.0}
    )

    return {
        "treatment": treatment,
        "baseline": baseline,
        "n_paired_instances": len(common),

        # 1. Resolution rate
        "rr_baseline": float(base_p.resolved.mean()),
        "rr_treatment": float(treat_p.resolved.mean()),
        "rr_lift_pp": float((treat_p.resolved.mean() - base_p.resolved.mean()) * 100),
        "mcnemar_p": mcn["p_value"],
        "mcnemar_b": mcn["b"],
        "mcnemar_c": mcn["c"],

        # 2. semantex usage
        "semantex_use_rate": float(used.mean()),
        "mean_semantex_calls": float(calls.mean()),
        "max_semantex_calls": int(calls.max() if len(calls) else 0),

        # 3. Turns
        "turns_baseline": float(base_p.num_turns.mean()),
        "turns_treatment": float(treat_p.num_turns.mean()),
        "turns_delta_mean": turns_d["mean_a_minus_b"],
        "turns_delta_ci": (turns_d["ci_low"], turns_d["ci_high"]),

        # 4. CCB
        "ccb_baseline": float(base_p.ccb.mean()),
        "ccb_treatment": float(treat_p.ccb.mean()),
        "ccb_delta_pct": float((treat_p.ccb.mean() - base_p.ccb.mean()) / base_p.ccb.mean() * 100)
                         if base_p.ccb.mean() > 0 else 0.0,
        "ccb_delta_ci": (ccb_d["ci_low"], ccb_d["ci_high"]),

        # 5. Cost
        "cost_baseline": float(base_p.cost_usd.mean()),
        "cost_treatment": float(treat_p.cost_usd.mean()),
        "cost_delta_ci": (cost_d["ci_low"], cost_d["ci_high"]),

        # 6. Resolved per $
        "rpd_baseline": float(base_rpd),
        "rpd_treatment": float(treat_rpd),

        # 7. Hard subset
        "hard_n": len(hard_ids),
        "hard_rr_baseline": float(base.loc[hard_ids].resolved.mean()) if hard_ids else 0.0,
        "hard_rr_treatment": float(treat.loc[hard_ids].resolved.mean()) if hard_ids else 0.0,
        "hard_lift_pp": (float(treat.loc[hard_ids].resolved.mean() - base.loc[hard_ids].resolved.mean()) * 100) if hard_ids else 0.0,
        "hard_mcnemar_p": hard_mcn["p_value"],
    }


def render(rows: list[dict], run_id: str) -> str:
    lines: list[str] = [f"# Condition comparison — `{run_id}`\n"]
    if not rows:
        lines.append("_No non-baseline conditions found in this run._\n")
        return "".join(lines)

    for r in rows:
        if r["n_paired_instances"] == 0:
            lines.append(f"## {r['treatment']} vs {r['baseline']}\n_No instances appear in both conditions._\n\n")
            continue

        b, t = r["baseline"], r["treatment"]
        lines.append(f"## {t} vs {b}\n")
        lines.append(f"Paired instances: **{r['n_paired_instances']}**\n\n")
        lines.append(f"| metric | {b} | {t} | delta |\n")
        lines.append("|---|---:|---:|---:|\n")
        lines.append(
            f"| Resolution rate | {r['rr_baseline']:.1%} | {r['rr_treatment']:.1%} | "
            f"**{r['rr_lift_pp']:+.1f}pp** "
            f"(McNemar p={r['mcnemar_p']:.3f}, b={r['mcnemar_b']}, c={r['mcnemar_c']}) |\n"
        )
        lines.append(
            f"| Mean turns | {r['turns_baseline']:.1f} | {r['turns_treatment']:.1f} | "
            f"{r['turns_delta_mean']:+.2f} "
            f"(CI95 [{r['turns_delta_ci'][0]:+.2f}, {r['turns_delta_ci'][1]:+.2f}]) |\n"
        )
        lines.append(
            f"| Mean CCB (tok) | {r['ccb_baseline']:,.0f} | {r['ccb_treatment']:,.0f} | "
            f"{r['ccb_delta_pct']:+.1f}% "
            f"(CI95 [{r['ccb_delta_ci'][0]:+,.0f}, {r['ccb_delta_ci'][1]:+,.0f}]) |\n"
        )
        lines.append(
            f"| Mean cost ($) | {r['cost_baseline']:.3f} | {r['cost_treatment']:.3f} | "
            f"CI95 [{r['cost_delta_ci'][0]:+.3f}, {r['cost_delta_ci'][1]:+.3f}] |\n"
        )
        lines.append(
            f"| Resolved per $ | {r['rpd_baseline']:.3f} | {r['rpd_treatment']:.3f} | "
            f"{(r['rpd_treatment'] - r['rpd_baseline']):+.3f} |\n"
        )

        lines.append(f"\n### semantex tool usage in {t}\n")
        lines.append(f"- **{r['semantex_use_rate']:.0%}** of paired units invoked a `semantex_*` tool at least once\n")
        lines.append(f"- Mean **{r['mean_semantex_calls']:.1f}** semantex calls per unit (max {r['max_semantex_calls']})\n")

        lines.append(
            f"\n### Hard subset ({b} needed >20 turns OR failed; n={r['hard_n']})\n"
        )
        if r["hard_n"]:
            lines.append(
                f"- {b} resolved: {r['hard_rr_baseline']:.1%}\n"
                f"- {t} resolved: {r['hard_rr_treatment']:.1%}\n"
                f"- **Lift on hard instances: {r['hard_lift_pp']:+.1f}pp** "
                f"(McNemar p={r['hard_mcnemar_p']:.3f})\n"
            )
        else:
            lines.append("- _No instances qualified as hard._\n")

        lines.append("\n---\n\n")

    return "".join(lines)


@click.command()
@click.option("--run-id", required=True)
@click.option("--baseline", default="c1_baseline", show_default=True)
@click.option("--replicate", default=0, type=int, show_default=True,
              help="Which replicate to compare (default rep 0). Aggregate across replicates not yet supported.")
def main(run_id: str, baseline: str, replicate: int):
    run_dir = RESULTS_DIR / run_id
    runs_dir = run_dir / "runs"
    eval_dir = run_dir / "eval"

    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} does not exist. Run scripts.run first.", file=sys.stderr)
        sys.exit(1)
    if not eval_dir.exists():
        print(f"ERROR: {eval_dir} does not exist. Run scripts.submit first.", file=sys.stderr)
        sys.exit(1)

    eval_reports = _load_eval_reports(eval_dir)
    pricing = yaml.safe_load((CONFIG_DIR / "models.yaml").read_text())

    df = build_unit_table(
        runs_dir=runs_dir, eval_reports=eval_reports,
        agent_model="claude-sonnet-4-6", pricing=pricing,
    )
    df = df[df.replicate == replicate]

    treatments = sorted(c for c in df.condition_id.unique() if c != baseline)
    rows = [compare(df, baseline, t) for t in treatments]

    md = render(rows, run_id)
    out = run_dir / "comparison.md"
    out.write_text(md)
    print(md)
    print(f"\n(Written to {out})", file=sys.stderr)


if __name__ == "__main__":
    main()
