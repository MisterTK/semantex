"""Aggregate RunOutputs into a metrics table + report.md / report.json.

Every report carries a reproducibility stamp (git rev, dense backend, model id,
k) and the subset manifest(s), so a number is never reported without provenance.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass

import pandas as pd

from .metrics import mean_average_precision, mrr_at_k, ndcg_at_k, recall_at_k
from .runner import RunOutput


@dataclass(frozen=True)
class ReproStamp:
    git_rev: str
    dense_backend: str
    model_id: str
    k: int


def current_git_rev() -> str:
    """Short git rev of the semantex repo, or 'unknown' if not a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def compute_metrics_row(run: RunOutput, *, k: int) -> dict:
    rels = run.relevances
    nrel = run.n_relevant
    return {
        "dataset": run.corpus_name,
        "ablation": run.ablation,
        "n_queries": len(rels),
        "mrr_at_10": mrr_at_k(rels, k=10),
        "ndcg_at_10": ndcg_at_k(rels, k=10, n_relevant=nrel),
        "recall_at_1": recall_at_k(rels, k=1, n_relevant=nrel),
        "recall_at_5": recall_at_k(rels, k=5, n_relevant=nrel),
        "recall_at_10": recall_at_k(rels, k=10, n_relevant=nrel),
        "map": mean_average_precision(rels, n_relevant=nrel),
    }


def render_report_json(*, rows: list[dict], stamp: ReproStamp, manifests: list[dict]) -> str:
    return json.dumps(
        {"stamp": asdict(stamp), "rows": rows, "manifests": manifests}, indent=2
    )


def render_report_md(*, rows: list[dict], stamp: ReproStamp, manifests: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# semantex Relevance Report\n\n")
    lines.append(
        f"- **git rev:** `{stamp.git_rev}`\n"
        f"- **dense backend:** `{stamp.dense_backend}`\n"
        f"- **model:** `{stamp.model_id}`\n"
        f"- **cutoff k:** {stamp.k}\n\n"
    )
    lines.append("## Metrics\n\n")
    df = pd.DataFrame(rows)
    lines.append(df.to_markdown(index=False, floatfmt=".4f") + "\n\n")
    if manifests:
        lines.append("## Subset manifest\n\n")
        for m in manifests:
            lines.append(
                f"- **{m['dataset']}**: kept {m['selected']}/{m['total']} "
                f"(seed {m['seed']}, dropped {len(m.get('dropped_ids', []))})\n"
            )
    return "".join(lines)
