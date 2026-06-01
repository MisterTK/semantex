from relevance_harness.subset import SubsetManifest, select_queries


def _queries(n: int) -> list[dict]:
    return [{"query_id": f"q{i}", "text": f"t{i}"} for i in range(n)]


def test_select_is_deterministic_for_same_seed():
    qs = _queries(20)
    a, ma = select_queries(qs, n=5, seed=42, dataset="csn")
    b, mb = select_queries(qs, n=5, seed=42, dataset="csn")
    assert [q["query_id"] for q in a] == [q["query_id"] for q in b]
    assert ma.kept_ids == mb.kept_ids


def test_different_seeds_differ():
    qs = _queries(20)
    a, _ = select_queries(qs, n=5, seed=1, dataset="csn")
    b, _ = select_queries(qs, n=5, seed=2, dataset="csn")
    assert {q["query_id"] for q in a} != {q["query_id"] for q in b}


def test_manifest_records_kept_and_dropped():
    qs = _queries(10)
    selected, manifest = select_queries(qs, n=3, seed=0, dataset="csn")
    assert isinstance(manifest, SubsetManifest)
    assert manifest.total == 10
    assert manifest.selected == 3
    assert len(manifest.kept_ids) == 3
    assert len(manifest.dropped_ids) == 7
    assert set(manifest.kept_ids) | set(manifest.dropped_ids) == {q["query_id"] for q in qs}
    assert manifest.seed == 0
    assert manifest.dataset == "csn"


def test_n_none_or_larger_than_pool_keeps_all_and_logs_no_drop():
    qs = _queries(4)
    selected, manifest = select_queries(qs, n=None, seed=0, dataset="csn")
    assert len(selected) == 4
    assert manifest.selected == 4
    assert manifest.dropped_ids == []


def test_select_sorts_by_id_for_canonical_order():
    qs = [{"query_id": f"q{i}", "text": "x"} for i in (5, 1, 3, 2, 4)]
    selected, _ = select_queries(qs, n=None, seed=0, dataset="csn")
    assert [q["query_id"] for q in selected] == ["q1", "q2", "q3", "q4", "q5"]
