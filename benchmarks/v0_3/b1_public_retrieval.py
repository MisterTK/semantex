#!/usr/bin/env python3
"""
B1 — Public retrieval sub-benchmark (CSN-adv, CoSQA, CodeXGLUE, SWE-bench-Lite).

Scaffolding only. Spec §4.1 requires running each tool against four published
datasets and reporting MRR@10, NDCG@10, Recall@{1,10,100}, F1, P@5.

This module sketches the contract `run_public.py` will call once the human
wires the datasets. We do NOT pretend to run a full B1 sweep — instead we
emit a `b1_status.json` recording which datasets are present and which are
TODO. Downstream aggregation can render this honestly.

See `benchmarks/BENCHMARK-v0.3-PLAN.md` for dataset URLs and prep steps.
"""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_DATASETS = {
    "codesearchnet-adv": "CodeSearchNet adversarial split",
    "cosqa": "CoSQA",
    "codexglue": "CodeXGLUE code-search task",
    "swe-bench-lite-retrieval": "SWE-bench-Lite retrieval split",
}


def run(*, tool: str, output_dir: Path, datasets_root: Path) -> dict:
    """Probe dataset availability under `datasets_root`. Always returns a
    structured dict, never crashes — so run_public.py can summarize honestly."""
    output_dir.mkdir(parents=True, exist_ok=True)
    status = {}
    for slug, name in REQUIRED_DATASETS.items():
        ds = datasets_root / slug
        manifest = ds / "MANIFEST.json"
        status[slug] = {
            "display": name,
            "present": ds.exists(),
            "manifest_present": manifest.exists(),
            "path": str(ds),
        }
    out = {
        "tool": tool,
        "ok": False,  # B1 never reports OK from this scaffold
        "reason": "B1 requires curated public datasets — see benchmarks/BENCHMARK-v0.3-PLAN.md",
        "datasets": status,
    }
    (output_dir / "b1_status.json").write_text(json.dumps(out, indent=2))
    return out
