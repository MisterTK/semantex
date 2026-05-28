"""Pre-index every instance's repo at its base SHA.

Idempotent: skips instances whose .semantex/updated_at marker exists.
Parallel: uses --workers (default 8). Each worker isolates one repo per dir.

Repos are stored under $SWE_BENCH_REPO_CACHE/<instance_id>/.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

import click

from swe_bench_harness.dataset import Instance, load_verified
from swe_bench_harness.indexer import index_repo
from swe_bench_harness.repo_checkout import checkout


def _cache_dir() -> Path:
    return Path(os.environ.get("SWE_BENCH_REPO_CACHE", Path.home() / ".swe_bench_repos"))


def _process_one(inst: Instance, semantex_bin: str) -> dict:
    dest = _cache_dir() / inst.instance_id
    marker = dest / ".semantex" / "updated_at"
    if marker.exists():
        return {"instance_id": inst.instance_id, "status": "cached"}
    t0 = time.monotonic()
    try:
        checkout(
            repo_url=f"https://github.com/{inst.repo}.git",
            sha=inst.base_commit,
            dest=dest,
        )
    except Exception as e:
        return {
            "instance_id": inst.instance_id,
            "status": "checkout_failed",
            "error": str(e),
        }
    res = index_repo(repo_path=dest, semantex_binary=semantex_bin, timeout_secs=900)
    return {
        "instance_id": inst.instance_id,
        "status": "indexed" if res.ok else "index_failed",
        "duration_secs": time.monotonic() - t0,
        "error": res.error if not res.ok else "",
    }


@click.command()
@click.option("--phase", type=click.Choice(["a", "b"]), required=True)
@click.option("--workers", default=8, type=int)
@click.option("--semantex-bin", default="semantex", show_default=True)
def main(phase: str, workers: int, semantex_bin: str) -> None:
    config = Path(__file__).parent.parent / "config"
    ids_file = config / f"instances_phase_{phase}.txt"
    wanted = {ln.strip() for ln in ids_file.read_text().splitlines() if ln.strip()}

    all_insts = load_verified()
    insts = [i for i in all_insts if i.instance_id in wanted]
    assert len(insts) == len(wanted), (
        f"mismatch: want {len(wanted)}, found {len(insts)}"
    )

    print(f"Pre-indexing {len(insts)} instances with {workers} workers...")
    results = []
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, i, semantex_bin): i for i in insts}
        for fut in cf.as_completed(futures):
            r = fut.result()
            results.append(r)
            print(f"  [{len(results):>4}/{len(insts)}] {r['instance_id']:<50} {r['status']}")

    out = _cache_dir() / f"pre_index_phase_{phase}_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    n_ok = sum(1 for r in results if r["status"] in ("indexed", "cached"))
    print(f"\nDone. {n_ok}/{len(results)} ready. Report: {out}")
    if n_ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
