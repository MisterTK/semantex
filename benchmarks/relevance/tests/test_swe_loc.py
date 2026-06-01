import json
from pathlib import Path

from relevance_harness.datasets.swe_loc import (
    changed_files_from_patch, load_swe_loc_queries,
)


def test_changed_files_from_single_file_patch():
    patch = (
        "diff --git a/src/auth.py b/src/auth.py\n"
        "--- a/src/auth.py\n+++ b/src/auth.py\n"
        "@@ -10,7 +10,7 @@ def login(username, password):\n"
        "-    return check(password)\n"
        "+    return check(password.encode('utf-8'))\n"
    )
    assert changed_files_from_patch(patch) == ["src/auth.py"]


def test_changed_files_from_multi_file_patch():
    patch = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        "diff --git a/pkg/b.py b/pkg/b.py\n--- a/pkg/b.py\n+++ b/pkg/b.py\n@@ -1 +1 @@\n-1\n+2\n"
    )
    assert changed_files_from_patch(patch) == ["a.py", "pkg/b.py"]


def test_changed_files_ignores_dev_null_for_new_files():
    # a newly-added file: the b/ side is the gold path, a/ side is /dev/null
    patch = (
        "diff --git a/new.py b/new.py\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x = 1\n"
    )
    assert changed_files_from_patch(patch) == ["new.py"]


def test_load_swe_loc_queries_from_local_fixture(fixtures_dir):
    queries = load_swe_loc_queries(local_path=fixtures_dir / "sample_verified_subset.json")
    assert len(queries) == 3
    q1 = next(q for q in queries if q.query_id == "demo__demo-1")
    assert "unicode" in q1.text.lower()
    assert q1.gold_doc_ids == ("src/auth.py",)
    q3 = next(q for q in queries if q.query_id == "demo__demo-3")
    assert set(q3.gold_doc_ids) == {"a.py", "pkg/b.py"}
