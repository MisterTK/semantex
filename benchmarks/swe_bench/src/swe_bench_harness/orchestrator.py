"""Fan out (instance × condition × replicate) units across workers,
checkpoint each result to disk, skip-existing for resumability."""
from __future__ import annotations

import concurrent.futures as cf
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .conditions import Condition
from .dataset import Instance
from .runner import RunResult, run_one


@dataclass(frozen=True)
class Unit:
    instance: Instance
    condition: Condition
    replicate: int

    @property
    def key(self) -> str:
        return f"{self.instance.instance_id}__{self.condition.id}__{self.replicate}"


def iter_units(
    instances: Iterable[Instance],
    conditions: Iterable[Condition],
    *,
    replicates: int,
) -> Iterator[Unit]:
    for inst in instances:
        for cond in conditions:
            for rep in range(replicates):
                yield Unit(instance=inst, condition=cond, replicate=rep)


def run_all(
    *,
    instances: list[Instance],
    conditions: list[Condition],
    replicates: int,
    out_dir: Path,
    repo_cache_root: Path,
    workers: int,
    max_turns: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    units = list(iter_units(instances, conditions, replicates=replicates))

    pending = [u for u in units if not (out_dir / f"{u.key}.json").exists()]
    print(f"{len(units)} units total; {len(pending)} pending after checkpoint.")

    def _do(u: Unit) -> Unit:
        result = run_one(
            instance=u.instance,
            condition=u.condition,
            replicate=u.replicate,
            repo_cache_root=repo_cache_root,
            max_turns=max_turns,
        )
        (out_dir / f"{u.key}.json").write_text(json.dumps(asdict(result), indent=2))
        return u

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for i, u in enumerate(pool.map(_do, pending), 1):
            print(f"  [{i:>5}/{len(pending)}] {u.key}")
