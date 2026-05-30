"""Pre-index every instance's repo at its base SHA.

Idempotent: skips instances whose .semantex/updated_at marker exists.
Parallel: uses --workers (default 8). Each worker isolates one repo per dir.

Repos are stored under $SWE_BENCH_REPO_CACHE/<instance_id>/.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import shutil
import sys
import time
from pathlib import Path

import click

from swe_bench_harness.dataset import Instance, load_verified
from swe_bench_harness.indexer import index_repo
from swe_bench_harness.repo_checkout import checkout


def _cache_dir() -> Path:
    return Path(os.environ.get("SWE_BENCH_REPO_CACHE", Path.home() / ".swe_bench_repos"))


def _index_is_complete(semantex_dir: Path) -> bool:
    """True iff the .semantex dir holds a finished index.

    semantex writes `.semantex/meta.json` LAST — after sparse, graph, and the
    PLAID build — so its presence marks a completed `semantex index` run. There
    is NO separate `updated_at` marker file; `updated_at` is a FIELD *inside*
    meta.json. (The earlier harness looked for a nonexistent `updated_at` file,
    so it never recognized a finished index: it re-deleted and re-indexed every
    repo on every pass and reported 0 done forever.) We additionally require a
    positive chunk_count so a torn/empty meta.json is treated as incomplete.
    """
    meta = semantex_dir / "meta.json"
    if not meta.exists():
        return False
    try:
        data = json.loads(meta.read_text())
    except (ValueError, OSError):
        return False
    return int(data.get("chunk_count", 0)) > 0


def _process_one(inst: Instance, semantex_bin: str) -> dict:
    dest = _cache_dir() / inst.instance_id
    semantex_dir = dest / ".semantex"
    if _index_is_complete(semantex_dir):
        return {"instance_id": inst.instance_id, "status": "cached"}
    # A .semantex dir without a valid meta.json is a partial/aborted index
    # (e.g. sparse built but dense PLAID never finished). `semantex index` would
    # treat its chunk manifest as up-to-date and skip every file, producing 0
    # chunks and never completing. Clear it so we always do a clean full index.
    if semantex_dir.exists():
        shutil.rmtree(semantex_dir, ignore_errors=True)
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
    # Large repos (e.g. django/sympy) embed tens of thousands of chunks on CPU
    # and can take 30-60 min; the old 900s default timed them out. Override via
    # SWE_BENCH_INDEX_TIMEOUT.
    timeout = int(os.environ.get("SWE_BENCH_INDEX_TIMEOUT", "7200"))
    res = index_repo(repo_path=dest, semantex_binary=semantex_bin, timeout_secs=timeout)
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
