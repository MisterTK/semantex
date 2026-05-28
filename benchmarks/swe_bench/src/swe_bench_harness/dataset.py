"""SWE-bench Verified dataset loader and deterministic subset selection."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Instance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str

    @property
    def repo_org(self) -> str:
        return self.repo.split("/", 1)[0]

    @property
    def repo_name(self) -> str:
        return self.repo.split("/", 1)[1]


def load_verified(*, local_path: Path | None = None) -> list[Instance]:
    """Load SWE-bench Verified. If local_path given, read JSON file instead of HF."""
    if local_path is not None:
        with local_path.open() as f:
            rows = json.load(f)
    else:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        rows = list(ds)
    return [
        Instance(
            instance_id=r["instance_id"],
            repo=r["repo"],
            base_commit=r["base_commit"],
            problem_statement=r["problem_statement"],
        )
        for r in rows
    ]


def select_subset(
    instances: Iterable[Instance], *, n: int, seed: int
) -> list[Instance]:
    """Deterministic random subset selection. Stable across runs for same seed."""
    pool = sorted(instances, key=lambda i: i.instance_id)  # canonical order
    if n >= len(pool):
        return pool
    rng = random.Random(seed)
    return sorted(rng.sample(pool, n), key=lambda i: i.instance_id)
