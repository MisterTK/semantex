import json

import pytest

from swe_bench_harness.dataset import Instance, load_verified, select_subset


def test_instance_round_trip():
    inst = Instance(
        instance_id="astropy__astropy-12907",
        repo="astropy/astropy",
        base_commit="d16bfe05a744909de4b27f5875fe0d4ed41ce607",
        problem_statement="...",
    )
    assert inst.repo_org == "astropy"
    assert inst.repo_name == "astropy"


def test_load_from_fixture(fixtures_dir):
    insts = load_verified(local_path=fixtures_dir / "tiny_verified_subset.json")
    assert len(insts) == 5
    assert insts[0].instance_id == "astropy__astropy-12907"


def test_select_subset_is_deterministic(fixtures_dir):
    insts = load_verified(local_path=fixtures_dir / "tiny_verified_subset.json")
    a = select_subset(insts, n=3, seed=42)
    b = select_subset(insts, n=3, seed=42)
    assert [i.instance_id for i in a] == [i.instance_id for i in b]


def test_select_subset_different_seeds_differ(fixtures_dir):
    insts = load_verified(local_path=fixtures_dir / "tiny_verified_subset.json")
    a = select_subset(insts, n=3, seed=1)
    b = select_subset(insts, n=3, seed=2)
    assert {i.instance_id for i in a} != {i.instance_id for i in b}


def test_select_subset_n_larger_than_pool_returns_all(fixtures_dir):
    insts = load_verified(local_path=fixtures_dir / "tiny_verified_subset.json")
    out = select_subset(insts, n=999, seed=0)
    assert len(out) == 5
