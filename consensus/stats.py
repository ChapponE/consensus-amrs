"""Distribution-free inference on paired per-population gains.

The unit is the population: ``delta[k]`` is the score of the consensus minus the
score of a baseline on population ``k``. None of these tests assumes normality.

- the paired **permutation** (sign-flip) test on the mean gain: under the null
  hypothesis that the two methods are exchangeable on each population, every
  gain is as likely to carry either sign. Exact enumeration up to
  ``EXACT_MAX_N`` populations, Monte Carlo beyond, with the ``(b + 1) / (B + 1)``
  estimator that never returns zero;
- the **Wilcoxon** signed-rank test, which only uses the ranks of the gains;
- the **sign** test, which only uses their signs;
- a **percentile bootstrap** confidence interval for the mean gain: the populations are
  resampled with replacement and the interval is read between the 2.5 and 97.5
  percentiles of the resampled means.

All randomness is seeded, so a table is reproducible bit for bit.
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


def permutation_test(delta, alternative: str = "two-sided",
                     n_resamples: int = N_PERMUTATIONS, seed: int = SEED) -> float:
    """Sign-flip p-value of the mean of ``delta``; NaN on an empty or all-zero sample."""
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
        if alternative == "two-sided":
            extreme += int((np.abs(means) >= abs(observed) - tolerance).sum())
        elif alternative == "greater":
            extreme += int((means >= observed - tolerance).sum())
        elif alternative == "less":
            extreme += int((means <= observed + tolerance).sum())
        else:
            raise ValueError(f"Unknown alternative {alternative!r}.")
        total += len(means)
    return extreme / total if exact else (extreme + 1) / (total + 1)


def wilcoxon(delta, alternative: str = "two-sided") -> float:
    """Wilcoxon signed-rank p-value; zero gains are discarded (Wilcoxon's convention)."""
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if not np.any(delta):
        return float("nan")
    return float(stats.wilcoxon(delta, alternative=alternative, zero_method="wilcox").pvalue)


def sign_test(delta, alternative: str = "two-sided") -> float:
    """Binomial test on the number of positive gains among the non-zero ones."""
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta) & (delta != 0)]
    if len(delta) == 0:
        return float("nan")
    return float(stats.binomtest(int((delta > 0).sum()), len(delta), 0.5,
                                 alternative=alternative).pvalue)


def bootstrap_ci(delta, level: float = 0.95, n_resamples: int = N_BOOTSTRAP,
                 seed: int = SEED, method: str = "percentile") -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean gain."""
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if len(delta) < 2 or np.ptp(delta) == 0:
        value = float(delta.mean()) if len(delta) else float("nan")
        return value, value
    result = stats.bootstrap((delta,), np.mean, confidence_level=level,
                             n_resamples=n_resamples, method=method,
                             rng=np.random.default_rng(seed))
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def lag1_autocorrelation(values, n_resamples: int = N_PERMUTATIONS // 10,
                         seed: int = SEED) -> tuple[float, float]:
    """Lag-1 autocorrelation of a series and its two-sided permutation p-value.

    Shuffling the order destroys any serial dependence while keeping the values,
    which is the null of independent, identically distributed populations.
    """
    x = np.asarray(values, dtype=float)
    x = x - x.mean()
    denominator = float((x ** 2).sum())
    if len(x) < 3 or denominator == 0.0:
        return float("nan"), float("nan")
    observed = float((x[:-1] * x[1:]).sum() / denominator)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_resamples):
        y = rng.permutation(x)
        extreme += abs(float((y[:-1] * y[1:]).sum() / denominator)) >= abs(observed) - 1e-12
    return observed, (extreme + 1) / (n_resamples + 1)


def paired_summary(ours, theirs) -> dict:
    """Every statistic reported for one comparison, keyed as in the result tables."""
    ours, theirs = np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float)
    delta = ours - theirs
    low, high = bootstrap_ci(delta)
    with np.errstate(all="ignore"):
        p_t2 = float(stats.ttest_rel(ours, theirs).pvalue)
    return dict(
        median_gain=float(np.median(delta)),
        ci95_low=low, ci95_high=high,
        n_worse=int((delta < 0).sum()), n_tied=int((delta == 0).sum()),
        p_perm=permutation_test(delta, "two-sided"),
        p_perm_greater=permutation_test(delta, "greater"),
        p_wilcoxon=wilcoxon(delta, "two-sided"),
        p_wilcoxon_greater=wilcoxon(delta, "greater"),
        p_sign_2s=sign_test(delta, "two-sided"),
        p_ttest_2s=p_t2,
    )


__all__ = ["permutation_test", "wilcoxon", "sign_test", "bootstrap_ci",
           "lag1_autocorrelation", "paired_summary"]
