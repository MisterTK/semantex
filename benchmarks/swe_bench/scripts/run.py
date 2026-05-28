"""Main entrypoint: orchestrate runs for a phase.

Examples:
  python -m scripts.run --phase a --replicates 2 --workers 4
  python -m scripts.run --phase a --conditions c1_baseline,c2_semantex_no_llm
  python -m scripts.run --phase b --replicates 3 --workers 10
  python -m scripts.run --phase a --run-id 20260528-phase_a  # resume an existing run
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import click

from swe_bench_harness.conditions import load_conditions
from swe_bench_harness.dataset import load_verified
from swe_bench_harness.orchestrator import run_all


@click.command()
@click.option("--phase", type=click.Choice(["a", "b"]), required=True)
@click.option("--replicates", default=2, type=int, show_default=True)
@click.option("--workers", default=4, type=int, show_default=True)
@click.option("--conditions", default="",
              help="Comma-separated condition keys; empty = all three")
@click.option("--max-turns", default=75, type=int, show_default=True)
@click.option("--run-id", default="", help="Reuse an existing run_id for resumability")
def main(phase: str, replicates: int, workers: int, conditions: str,
         max_turns: int, run_id: str) -> None:
    config = Path(__file__).parent.parent / "config"
    repo_cache = Path(os.environ.get("SWE_BENCH_REPO_CACHE", Path.home() / ".swe_bench_repos"))

    ids_file = config / f"instances_phase_{phase}.txt"
    wanted = {ln.strip() for ln in ids_file.read_text().splitlines() if ln.strip()}
    insts = [i for i in load_verified() if i.instance_id in wanted]

    all_conds = load_conditions(config / "conditions.yaml")
    keys = conditions.split(",") if conditions else list(all_conds.keys())
    chosen = [all_conds[k] for k in keys]

    if not run_id:
        run_id = time.strftime(f"%Y%m%d-%H%M%S-phase_{phase}")
    out_dir = Path(__file__).parent.parent / "results" / run_id / "runs"
    print(f"run_id = {run_id}")
    print(f"out_dir = {out_dir}")
    print(f"instances = {len(insts)}, conditions = {[c.id for c in chosen]}, replicates = {replicates}")

    run_all(
        instances=insts, conditions=chosen, replicates=replicates,
        out_dir=out_dir, repo_cache_root=repo_cache,
        workers=workers, max_turns=max_turns,
    )


if __name__ == "__main__":
    main()
