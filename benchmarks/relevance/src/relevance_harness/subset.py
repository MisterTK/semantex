"""Deterministic, seeded, *logged* subsetting. Never silently truncate."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SubsetManifest:
    """Exactly what a subset kept and dropped — recorded in every report."""
    dataset: str
    total: int
    selected: int
    seed: int
    kept_ids: list[str]
    dropped_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "total": self.total,
            "selected": self.selected,
            "seed": self.seed,
            "kept_ids": self.kept_ids,
            "dropped_ids": self.dropped_ids,
        }


def select_queries(
    queries: list[dict],
    *,
    n: Optional[int],
    seed: int,
    dataset: str,
    id_key: str = "query_id",
) -> tuple[list[dict], SubsetManifest]:
    """Pick a deterministic seeded subset of `queries` and record a manifest.

    `queries` is a list of dicts each carrying `id_key`. Canonical order is by
    id (stable across runs). If `n` is None or >= the pool size, keep ALL and
    record an empty dropped list. Selection is reproducible for a fixed seed.
    """
    pool = sorted(queries, key=lambda q: q[id_key])
    total = len(pool)
    if n is None or n >= total:
        kept = pool
    else:
        rng = random.Random(seed)
        kept = sorted(rng.sample(pool, n), key=lambda q: q[id_key])
    kept_ids = [q[id_key] for q in kept]
    kept_set = set(kept_ids)
    dropped_ids = [q[id_key] for q in pool if q[id_key] not in kept_set]
    manifest = SubsetManifest(
        dataset=dataset,
        total=total,
        selected=len(kept),
        seed=seed,
        kept_ids=kept_ids,
        dropped_ids=dropped_ids,
    )
    return kept, manifest
