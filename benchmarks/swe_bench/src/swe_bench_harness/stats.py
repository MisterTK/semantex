"""Statistical comparisons for paired benchmark designs."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def mcnemar_paired_test(
    *,
    baseline_resolved: set[str],
    treatment_resolved: set[str],
    all_instances: list[str],
) -> dict:
    """Exact McNemar test on resolution status across paired instances.

    For each instance, observe baseline-resolved and treatment-resolved.
    b = #instances baseline-resolved but treatment-not
    c = #instances treatment-resolved but baseline-not
    Test H0: b == c (no difference in discordant pairs)."""
    b = c = 0
    for inst in all_instances:
        in_base = inst in baseline_resolved
        in_treat = inst in treatment_resolved
        if in_base and not in_treat:
            b += 1
        elif in_treat and not in_base:
            c += 1
    n = b + c
    if n == 0:
        p = 1.0
    else:
        p = binomtest(c, n, p=0.5).pvalue
    return {
        "b": b,
        "c": c,
        "p_value": p,
        "treatment_lift_pp": ((len(treatment_resolved) - len(baseline_resolved))
                              / len(all_instances)) * 100,
    }


def paired_bootstrap_diff(
    *,
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """Paired bootstrap CI for mean(a - b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, "a and b must be paired"
    diffs = a - b
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = diffs[idx].mean()
    lo = np.quantile(means, (1 - ci) / 2)
    hi = np.quantile(means, 1 - (1 - ci) / 2)
    return {
        "mean_a_minus_b": float(diffs.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_pairs": n,
    }
