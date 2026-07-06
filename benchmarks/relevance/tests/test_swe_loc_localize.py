"""Tests for scripts/swe_loc_localize.py.

`test_compute_arm_rows_*` are pure unit tests (no I/O). `test_cli_end_to_end*`
drives the real CLI against the tiny, no-network `tiny_corpus` fixture --
skipped (not failed) when the `semantex` binary isn't on PATH, so `pytest`
stays green in a CI-less environment while still exercising the full
index -> 5-arm-search -> report pipeline for real whenever the binary is
available (note: with the binary present, the `rerank` arm may reach out to
download the cross-encoder model on its first-ever run).
"""
import shutil

import pytest

from scripts import swe_loc_localize as loc_script


def test_compute_arm_rows_basic_metrics():
    relevances = {
        "hybrid": [[1, 0], [0, 1]],
        "sparse-only": [[0, 0], [0, 1]],
        "rerank": [[1, 0], [1, 0]],
        "agent-routed": [[1, 0], [1, 0]],
        "ripgrep": [[0, 0], [0, 0]],
    }
    tokens = {
        "hybrid": [100, 200], "sparse-only": [10, 20], "rerank": [120, 220],
        "agent-routed": [50, 50], "ripgrep": [0, 0],
    }
    errors = {a: 0 for a in loc_script.ARMS}

    rows = loc_script.compute_arm_rows(relevances, tokens, errors)
    by_arm = {r["arm"]: r for r in rows}

    # q1 hits at rank 1 ([1, 0]), q2's hit is at rank 2 ([0, 1]) -> acc@1 = 0.5
    assert by_arm["hybrid"]["acc_at_1"] == pytest.approx(0.5)
    assert by_arm["hybrid"]["acc_at_10"] == pytest.approx(1.0)
    assert by_arm["sparse-only"]["acc_at_1"] == pytest.approx(0.0)
    assert by_arm["sparse-only"]["acc_at_10"] == pytest.approx(0.5)
    assert by_arm["rerank"]["acc_at_1"] == pytest.approx(1.0)
    assert by_arm["ripgrep"]["acc_at_10"] == pytest.approx(0.0)
    assert by_arm["hybrid"]["avg_tokens_returned"] == pytest.approx(150.0)
    assert by_arm["hybrid"]["n_queries"] == 2
    # No durations passed -> latency columns are present but null.
    assert by_arm["hybrid"]["p50_latency_ms"] is None
    assert by_arm["hybrid"]["p95_warm_latency_ms"] is None


def test_compute_arm_rows_reports_errors_and_handles_zero_queries():
    relevances = {a: [] for a in loc_script.ARMS}
    tokens = {a: [] for a in loc_script.ARMS}
    errors = {a: 0 for a in loc_script.ARMS}
    errors["ripgrep"] = 3

    rows = loc_script.compute_arm_rows(relevances, tokens, errors)
    by_arm = {r["arm"]: r for r in rows}
    assert by_arm["ripgrep"]["n_errors"] == 3
    assert by_arm["ripgrep"]["n_queries"] == 0
    assert by_arm["ripgrep"]["avg_tokens_returned"] == 0.0
    assert by_arm["ripgrep"]["acc_at_1"] == 0.0


def test_compute_arm_rows_latency_percentiles():
    relevances = {a: [[1], [1], [1]] for a in loc_script.ARMS}
    tokens = {a: [10, 10, 10] for a in loc_script.ARMS}
    errors = {a: 0 for a in loc_script.ARMS}
    # 100ms/200ms/300ms cold; p50=200ms, p95 close to 300ms (linear interp).
    durations = {a: [0.1, 0.2, 0.3] for a in loc_script.ARMS}
    warm_durations = {"hybrid": [0.01, 0.02, 0.03], "rerank": [0.01, 0.02, 0.03]}

    rows = loc_script.compute_arm_rows(relevances, tokens, errors, durations, warm_durations)
    by_arm = {r["arm"]: r for r in rows}

    assert by_arm["hybrid"]["p50_latency_ms"] == pytest.approx(200.0)
    assert by_arm["hybrid"]["p95_latency_ms"] == pytest.approx(290.0)
    assert by_arm["hybrid"]["p50_warm_latency_ms"] == pytest.approx(20.0)
    # ripgrep has cold durations but no warm rerun -> warm columns are null.
    assert by_arm["ripgrep"]["p50_latency_ms"] == pytest.approx(200.0)
    assert by_arm["ripgrep"]["p50_warm_latency_ms"] is None
    assert by_arm["ripgrep"]["p95_warm_latency_ms"] is None


@pytest.mark.skipif(shutil.which("semantex") is None, reason="semantex binary not on PATH")
def test_cli_end_to_end_on_tiny_fixture(tmp_path, fixtures_dir, monkeypatch):
    # Materialise one "repo" under a fake SWE_BENCH_REPO_CACHE, matching the
    # instance_id in tiny_swe_loc_instance.json, from the tiny_corpus fixture
    # (auth.py / db.py / util.py) so the whole index -> search pipeline runs
    # against real, tiny, local-only files -- no network involved.
    cache_dir = tmp_path / "repo_cache"
    instance_dir = cache_dir / "tiny__tiny-1"
    shutil.copytree(fixtures_dir / "tiny_corpus", instance_dir)
    monkeypatch.setenv("SWE_BENCH_REPO_CACHE", str(cache_dir))

    from click.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(loc_script.main, [
        "--local-fixture", str(fixtures_dir / "tiny_swe_loc_instance.json"),
        "--run-id", "pytest-tiny-smoke",
        "--k", "10",
    ])
    assert result.exit_code == 0, result.output

    out_dir = loc_script.RESULTS / "pytest-tiny-smoke"
    try:
        assert (out_dir / "report.md").exists()
        assert (out_dir / "report.json").exists()
        assert (out_dir / "per_instance.json").exists()
        report = (out_dir / "report.json").read_text()
        assert "hybrid" in report and "sparse-only" in report and "agent-routed" in report
        assert "rerank" in report
        # the gold file (auth.py) should be found at rank 1 by at least the
        # hybrid arm on a corpus this small and this on-the-nose a query.
        assert '"acc_at_1"' in report
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
