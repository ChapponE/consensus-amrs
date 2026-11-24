"""Inference on paired per-population gains (Appendix B.2 of the paper).

The unit is the population: ``delta[k]`` is the score of the consensus minus the
score of a baseline on population ``k``.

- a two-sided paired **permutation** (sign-flip) test on the mean gain: exact
  enumeration up to ``EXACT_MAX_N`` populations, ``N_PERMUTATIONS`` random sign
  vectors beyond, with the ``(b + 1) / (B + 1)`` estimator;
- a 95% **percentile bootstrap** interval of the mean gain (``N_BOOTSTRAP``
  resamples of the populations).

All randomness is seeded, so the tables are reproducible bit for bit.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

EXACT_MAX_N = 20
N_PERMUTATIONS = 200_000
N_BOOTSTRAP = 20_000
SEED = 0


def _sign_matrix(n: int, rng: np.random.Generator, n_resamples: int):
    """Yield ``(signs, exact)`` chunks: all 2^n sign vectors, or random ones."""
    if n <= EXACT_MAX_N:
        codes = np.arange(2 ** n, dtype=np.int64)
        for start in range(0, len(codes), 1 << 16):
            bits = (codes[start:start + (1 << 16), None] >> np.arange(n)) & 1
            yield 1.0 - 2.0 * bits, True
        return
    remaining = n_resamples
    while remaining > 0:
        size = min(remaining, 1 << 14)
        yield rng.choice((-1.0, 1.0), size=(size, n)), False
        remaining -= size


def permutation_test(delta, n_resamples: int = N_PERMUTATIONS, seed: int = SEED) -> float:
    """Two-sided sign-flip p-value of the mean of ``delta``; NaN on an empty or all-zero sample."""
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if len(delta) == 0 or not np.any(delta):
        return float("nan")
    observed = delta.mean()
    # Resampled means equal to the observed one up to rounding count as extreme.
    tolerance = 1e-12 * max(1.0, float(np.abs(delta).sum()))
    rng = np.random.default_rng(seed)
    extreme, total, exact = 0, 0, True
    for signs, exact in _sign_matrix(len(delta), rng, n_resamples):
        means = signs @ delta / len(delta)
        extreme += int((np.abs(means) >= abs(observed) - tolerance).sum())
        total += len(means)
    return extreme / total if exact else (extreme + 1) / (total + 1)


def bootstrap_ci(delta, level: float = 0.95, n_resamples: int = N_BOOTSTRAP,
                 seed: int = SEED) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean gain."""
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if len(delta) < 2 or np.ptp(delta) == 0:
        value = float(delta.mean()) if len(delta) else float("nan")
        return value, value
    result = stats.bootstrap((delta,), np.mean, confidence_level=level,
                             n_resamples=n_resamples, method="percentile",
                             rng=np.random.default_rng(seed))
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def paired_summary(ours, theirs) -> dict:
    """Interval and p-value of the mean gain ``ours - theirs``, keyed as in the result tables."""
    delta = np.asarray(ours, dtype=float) - np.asarray(theirs, dtype=float)
    low, high = bootstrap_ci(delta)
    return dict(ci95_low=low, ci95_high=high, p_perm=permutation_test(delta))


__all__ = ["permutation_test", "bootstrap_ci", "paired_summary"]
