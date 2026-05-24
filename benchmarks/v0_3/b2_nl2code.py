#!/usr/bin/env python3
"""
B2 — Owned NL→code 50-query benchmark.

Iterates the curated query slots in `benchmarks/datasets/nl2code-v1/MANIFEST.json`,
runs semantex (and any other detected tool) against each `target_repo_url@sha`,
and computes file-level Precision/Recall/F1 vs the `expected_files` ground truth.

Status today: MANIFEST contains 10 seeded example queries (status="active") and
40 slots marked status="needs_curation". This script runs only the active rows
and writes a status summary covering both.

Per spec §4.7: any benchmark with SD > 30% of mean across replications is
flagged "noise-dominated, no claim". B2 here defaults to one replication
because the metrics are deterministic — recall variance comes from re-curating
the manifest, not re-running the same query.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent.parent / "datasets" / "nl2code-v1" / "MANIFEST.json"
)


@dataclass
class B2Result:
    qid: str
    language: str
    question: str
    target_repo: str
    expected_files: list
    found_files: list
    precision: float
    recall: float
    f1: float
    elapsed_ms: float
    error: Optional[str] = None
    meets_recall_target: Optional[bool] = None


def _normalize_path(path: str, repo_root: str) -> str:
    if os.path.isabs(path):
        try:
            return os.path.relpath(path, repo_root)
        except ValueError:
            return path
    return path


def _run_semantex(semantex_bin: str, repo: str, query: str, top_k: int = 10) -> tuple:
    cmd = [
        semantex_bin,
        "--json", "--max-count", str(top_k),
        "-p", repo, query,
    ]
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, timeout=60)
    elapsed = (time.perf_counter() - start) * 1000.0
    if proc.returncode != 0:
        return [], elapsed, f"semantex rc={proc.returncode}: {proc.stderr.decode(errors='replace')[:200]}"
    try:
        results = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return [], elapsed, f"json decode: {e}"
    files = []
    if isinstance(results, list):
        for r in results:
            fp = r.get("file") or r.get("file_path") or ""
            if fp:
                files.append(_normalize_path(fp, repo))
    return files, elapsed, None


def _prf(found_set: set, truth_set: set) -> tuple:
    if not truth_set and not found_set:
        return 1.0, 1.0, 1.0
    tp = len(found_set & truth_set)
    fp = len(found_set - truth_set)
    fn = len(truth_set - found_set)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def run(*, tool: str, output_dir: Path, semantex_bin: str,
        manifest_path: Optional[Path] = None, repo_resolver=None) -> dict:
    """`repo_resolver(slot) -> Optional[str]` returns a local checkout path
    for the slot's `target_repo_url@target_sha`. If None, the row is skipped
    with a "repo not checked out" note.

    Without a resolver, this scaffold can't run anything — but it still emits
    a structured status file so the orchestrator can include B2 in the summary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_p = manifest_path or DEFAULT_MANIFEST
    if not manifest_p.exists():
        out = {"tool": tool, "ok": False, "reason": f"missing manifest: {manifest_p}"}
        (output_dir / "b2_status.json").write_text(json.dumps(out, indent=2))
        return out

    manifest = json.loads(manifest_p.read_text())
    queries = manifest.get("queries", [])
    active_queries = [q for q in queries if q.get("status") == "active"]
    needs_curation = [q for q in queries if q.get("status") == "needs_curation"]

    results: list = []
    failures: list = []

    for q in active_queries:
        repo_path = repo_resolver(q) if repo_resolver else None
        if repo_path is None or not Path(repo_path).exists():
            failures.append({
                "qid": q["qid"],
                "reason": f"repo not available locally for {q.get('target_repo_url')}@{q.get('target_sha')}",
            })
            continue
        if tool == "semantex":
            found, elapsed, err = _run_semantex(semantex_bin, repo_path, q["question"])
        else:
            failures.append({"qid": q["qid"], "reason": f"tool {tool} not wired"})
            continue
        if err is not None:
            failures.append({"qid": q["qid"], "reason": err, "elapsed_ms": elapsed})
            continue
        truth = set(q.get("expected_files", []))
        p, r, f = _prf(set(found), truth)
        rt = q.get("min_recall_target", 0.5)
        results.append(asdict(B2Result(
            qid=q["qid"], language=q.get("language", "?"),
            question=q["question"], target_repo=str(repo_path),
            expected_files=sorted(truth), found_files=sorted(set(found)),
            precision=p, recall=r, f1=f,
            elapsed_ms=elapsed, meets_recall_target=(r >= rt),
        )))

    summary = {
        "tool": tool,
        "ok": True if results else False,
        "manifest": str(manifest_p),
        "n_active": len(active_queries),
        "n_needs_curation": len(needs_curation),
        "n_total_slots": len(queries),
        "n_evaluated": len(results),
        "n_failures": len(failures),
    }
    if results:
        summary["mean_f1"] = statistics.mean(r["f1"] for r in results)
        summary["mean_recall"] = statistics.mean(r["recall"] for r in results)
        summary["mean_precision"] = statistics.mean(r["precision"] for r in results)
    summary["results"] = results
    summary["failures"] = failures
    (output_dir / f"b2_{tool}.json").write_text(json.dumps(summary, indent=2))
    return summary
