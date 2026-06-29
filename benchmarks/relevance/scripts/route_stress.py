"""Oracle-regret route-router evaluation entrypoint.

Loads the route_stress corpora, runs every retrieval route over every query,
records the keyword router's choice per query, and emits the decision table:
oracle-best route, per-route oracle-win counts, regret (overall / per-mechanism /
per-repo), router accuracy + confusion matrix, and per-mechanism × per-route nDCG.

Examples:
  python -m scripts.route_stress                     # all 3 repos
  python -m scripts.route_stress --repo gin          # one repo (fast partial)
  python -m scripts.route_stress --repo gin --verbose

Outputs (committed) land in results/route_stress/:
  report-<repo>.json   machine-readable per-repo result
  report.json          combined machine-readable result
  report.md            combined human-readable decision table
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import click

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
SWE_BENCH_SRC = ROOT.parent / "swe_bench" / "src"
if SWE_BENCH_SRC.is_dir():
    sys.path.insert(0, str(SWE_BENCH_SRC))

from relevance_harness.report import current_git_rev  # noqa: E402
from relevance_harness.route_eval import (  # noqa: E402
    RETRIEVAL_ROUTES,
    SYNTHESIS_ROUTES,
    RouteEvalResult,
    evaluate_repo,
    result_to_dict,
)

FIXTURES = ROOT / "fixtures" / "route_stress"
RESULTS = ROOT / "results" / "route_stress"

# repo -> (corpus fixture, on-disk repo dir). Repo dirs match the route_stress
# `repo_path_at_gen` fields; override with SEMANTEX_ROUTESTRESS_<REPO>_DIR env.
REPOS: dict[str, tuple[str, str]] = {
    "gin": ("gin_route_stress.json", "/path/to/gin"),
    "flask": ("flask_route_stress.json", "/path/to/flask"),
    "platform": ("platform_route_stress.json", "/path/to/platform"),
}


def _repo_dir(repo: str, default: str) -> str:
    return os.environ.get(f"SEMANTEX_ROUTESTRESS_{repo.upper()}_DIR", default)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _render_repo_md(res: RouteEvalResult) -> str:
    lines: list[str] = []
    lines.append(f"## {res.repo}")
    lines.append("")
    lines.append(f"- total queries: **{res.total_queries}**")
    lines.append(f"- overall regret (oracle nDCG@10 − router-picked nDCG@10): "
                 f"**{res.overall_regret:.4f}**")
    lines.append(f"- router accuracy (router choice == an oracle-best route): "
                 f"**{res.overall_router_accuracy:.1%}** "
                 f"({res.router_matches_oracle_count}/{res.total_queries})")
    lines.append(f"- router chose a SYNTHESIS route (no file hits, scored 0): "
                 f"**{res.synthesis_count}**")
    lines.append("")

    # Per-route oracle-win counts (the falsifiability signal).
    lines.append("### Per-route oracle-win counts")
    lines.append("(a route 'wins' a query when it is — or ties — the best-nDCG route; "
                 "~0 wins ⇒ deletion candidate)")
    lines.append("")
    lines.append(_md_table(
        ["route", "oracle-wins", "mean nDCG@10"],
        [[r, res.oracle_win_counts.get(r, 0),
          f"{res.per_route_mean_ndcg.get(r, 0.0):.4f}"] for r in RETRIEVAL_ROUTES],
    ))
    lines.append("")

    # Per-mechanism × per-route mean nDCG.
    lines.append("### Per-mechanism × per-route mean nDCG@10")
    lines.append("(does each route actually win on its intended mechanism? does one "
                 "route dominate everything?)")
    lines.append("")
    mechs = sorted(res.per_mechanism_route_ndcg.keys())
    rows = []
    for mech in mechs:
        rmap = res.per_mechanism_route_ndcg[mech]
        best_route = max(RETRIEVAL_ROUTES, key=lambda r: rmap.get(r, -1.0))
        row = [mech]
        for r in RETRIEVAL_ROUTES:
            v = rmap.get(r, 0.0)
            cell = f"**{v:.3f}**" if r == best_route and v > 0 else f"{v:.3f}"
            row.append(cell)
        rows.append(row)
    lines.append(_md_table(["mechanism", *RETRIEVAL_ROUTES], rows))
    lines.append("")

    # Regret per mechanism.
    lines.append("### Regret per mechanism")
    lines.append("")
    lines.append(_md_table(
        ["mechanism", "mean regret", "router accuracy"],
        [[mech, f"{res.per_mechanism_regret.get(mech, 0.0):.4f}",
          f"{res.per_mechanism_router_accuracy.get(mech, 0.0):.1%}"]
         for mech in mechs],
    ))
    lines.append("")

    # Confusion matrix: rows = intended_mechanism, cols = router choice.
    lines.append("### Confusion matrix (rows = intended_mechanism, cols = router choice)")
    lines.append("")
    chosen_cols: list[str] = []
    for mech in mechs:
        for col in res.confusion_matrix.get(mech, {}):
            if col not in chosen_cols:
                chosen_cols.append(col)
    # Order columns: retrieval routes first, then synthesis, then anything else.
    def _col_key(c: str) -> tuple[int, str]:
        if c in RETRIEVAL_ROUTES:
            return (0, str(RETRIEVAL_ROUTES.index(c)))
        if c in SYNTHESIS_ROUTES:
            return (1, c)
        return (2, c)
    chosen_cols.sort(key=_col_key)
    rows = []
    for mech in mechs:
        cm = res.confusion_matrix.get(mech, {})
        rows.append([mech, *[cm.get(c, 0) for c in chosen_cols]])
    lines.append(_md_table(["mechanism ↓ / choice →", *chosen_cols], rows))
    lines.append("")
    return "\n".join(lines)


def _render_combined_md(results: list[RouteEvalResult], git_rev: str) -> str:
    lines: list[str] = []
    lines.append("# Route-stress oracle-regret evaluation")
    lines.append("")
    lines.append(f"- git rev: `{git_rev}`")
    lines.append(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- config: `SEMANTEX_ADAPTIVE_SIZING=0` (canonical A/B lock)")
    lines.append(f"- retrieval routes scored: {', '.join(RETRIEVAL_ROUTES)}")
    lines.append(f"- synthesis routes (scored 0 for file-gold): "
                 f"{', '.join(sorted(SYNTHESIS_ROUTES))}")
    lines.append("")

    # Cross-repo summary.
    lines.append("## Summary (per repo)")
    lines.append("")
    lines.append(_md_table(
        ["repo", "queries", "overall regret", "router acc", "synthesis picks"],
        [[r.repo, r.total_queries, f"{r.overall_regret:.4f}",
          f"{r.overall_router_accuracy:.1%}", r.synthesis_count] for r in results],
    ))
    lines.append("")

    # Pooled per-route oracle wins across all repos.
    pooled_wins = {route: 0 for route in RETRIEVAL_ROUTES}
    total_q = 0
    for r in results:
        total_q += r.total_queries
        for route in RETRIEVAL_ROUTES:
            pooled_wins[route] += r.oracle_win_counts.get(route, 0)
    lines.append("## Pooled per-route oracle-win counts (all repos)")
    lines.append(f"(total queries: {total_q})")
    lines.append("")
    lines.append(_md_table(
        ["route", "oracle-wins"],
        [[route, pooled_wins[route]] for route in RETRIEVAL_ROUTES],
    ))
    lines.append("")

    for r in results:
        lines.append(_render_repo_md(r))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--repo", type=click.Choice(["gin", "flask", "platform", "all"]),
              default="all", show_default=True,
              help="Which repo(s) to evaluate. 'all' runs gin+flask+platform.")
@click.option("--k", default=10, type=int, show_default=True)
@click.option("--embedder", default="", help="lateon-colbert | coderank-137m "
              "(SEMANTEX_EMBEDDER selector); blank = repo's on-disk backend")
@click.option("--semantex-bin", default=os.environ.get("SEMANTEX_BINARY", "semantex"))
@click.option("--verbose", is_flag=True, help="Per-query / per-route progress")
def main(repo, k, embedder, semantex_bin, verbose):
    embedder = embedder or None
    repos = ["gin", "flask", "platform"] if repo == "all" else [repo]

    RESULTS.mkdir(parents=True, exist_ok=True)
    git_rev = current_git_rev()

    results: list[RouteEvalResult] = []
    for name in repos:
        fixture, default_dir = REPOS[name]
        corpus_path = FIXTURES / fixture
        repo_dir = _repo_dir(name, default_dir)
        if not Path(repo_dir).is_dir():
            click.echo(f"SKIP {name}: repo dir not found: {repo_dir}", err=True)
            continue
        click.echo(f"=== {name} ({repo_dir}) ===", err=True)
        t0 = time.monotonic()
        res = evaluate_repo(
            corpus_path,
            repo_dir,
            semantex_binary=semantex_bin,
            k=k,
            embedder=embedder,
            verbose=verbose,
        )
        click.echo(f"    done in {time.monotonic() - t0:.1f}s "
                   f"(regret={res.overall_regret:.4f}, "
                   f"router_acc={res.overall_router_accuracy:.1%})", err=True)
        results.append(res)
        # Write per-repo JSON immediately (partial-run safety).
        (RESULTS / f"report-{name}.json").write_text(
            json.dumps(result_to_dict(res), indent=2))

    if not results:
        click.echo("No repos evaluated.", err=True)
        raise SystemExit(1)

    # Combined machine-readable + markdown report.
    combined = {
        "git_rev": git_rev,
        "config": {"adaptive_sizing": "0", "k": k,
                   "embedder": embedder or "on-disk-backend"},
        "retrieval_routes": list(RETRIEVAL_ROUTES),
        "synthesis_routes": sorted(SYNTHESIS_ROUTES),
        "repos": [result_to_dict(r) for r in results],
    }
    (RESULTS / "report.json").write_text(json.dumps(combined, indent=2))
    md = _render_combined_md(results, git_rev)
    (RESULTS / "report.md").write_text(md)
    click.echo(f"\nReport: {RESULTS / 'report.md'}")
    click.echo(md)


if __name__ == "__main__":
    main()
