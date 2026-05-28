import numpy as np

from swe_bench_harness.stats import (
    mcnemar_paired_test, paired_bootstrap_diff,
)


def test_mcnemar_returns_pvalue_and_effect():
    # treatment resolves {r1, r2, r3}; baseline resolves {r1, r4}
    # paired contingency:
    #   r1: both resolved
    #   r2: treatment only (c=1)
    #   r3: treatment only (c=2)
    #   r4: baseline only (b=1)
    baseline_resolved = {"r1", "r4"}
    treatment_resolved = {"r1", "r2", "r3"}
    all_instances = ["r1", "r2", "r3", "r4"]
    result = mcnemar_paired_test(
        baseline_resolved=baseline_resolved,
        treatment_resolved=treatment_resolved,
        all_instances=all_instances,
    )
    assert "p_value" in result
    assert result["b"] == 1
    assert result["c"] == 2
    assert result["treatment_lift_pp"] == ((3 - 2) / 4) * 100


def test_bootstrap_diff_returns_ci():
    rng = np.random.default_rng(42)
    a = rng.normal(loc=10, scale=2, size=100)
    b = rng.normal(loc=9, scale=2, size=100)
    out = paired_bootstrap_diff(a=a, b=b, n_resamples=1000, seed=42)
    assert out["mean_a_minus_b"] > 0  # a is larger
    assert out["ci_low"] < out["mean_a_minus_b"] < out["ci_high"]


def test_bootstrap_diff_zero_when_identical():
    a = np.array([1.0, 2.0, 3.0])
    out = paired_bootstrap_diff(a=a, b=a, n_resamples=200, seed=0)
    assert abs(out["mean_a_minus_b"]) < 1e-9
