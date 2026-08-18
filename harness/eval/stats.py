"""Interval estimates and reliability metrics for evaluation results.

Two questions need different answers, and conflating them is the most common
way a small robotics leaderboard misleads:

- "How reliable is this model on this task?"  -> an interval on one success
  rate, plus pass^k for consistency across repeats.
- "Is model A better than model B?"          -> paired comparison over a task
  set; interval overlap is NOT a significance test.

Every function is pure and takes plain counts, so results can be recomputed
from a flat per-episode log without re-running anything.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

_Z95 = 1.959963984540054  # normal quantile for a two-sided 95% interval


def _has_scipy() -> bool:
    try:
        import scipy.stats  # noqa: F401

        return True
    except ImportError:
        return False


def beta_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Bayesian credible interval on a success rate, as RoboLab computes it.

    Uniform Beta(1,1) prior -> Beta(k+1, n-k+1) posterior. Preferred over a
    Wald standard error because it stays inside [0, 1] and is correctly
    asymmetric near the boundaries -- a Wald interval is simply wrong at 10/10.

    Falls back to a Wilson score interval when scipy is absent; the two agree
    closely, and Wilson shares the properties that matter here.

    At the boundaries the interval is made **one-sided**, which is the standard
    convention and not a cosmetic choice. An equal-tailed interval at k=0 puts
    2.5% of the posterior mass below the observed rate, so it returns something
    like [0.1%, 11.2%] for a model that scored 0/30 -- an interval that excludes
    its own point estimate. Reported next to a bar of length zero that reads as
    a bug in the harness, and a reader cannot tell "scored zero" from "we are
    unsure it is zero". Spending the whole alpha on the side where the
    uncertainty actually lives gives [0, 9.3%]: the claim we can support.
    """
    if n <= 0:
        return 0.0, 1.0
    k = max(0, min(int(k), int(n)))
    alpha = 1.0 - confidence
    if not _has_scipy():
        return wilson_ci(k, n, confidence=confidence)
    from scipy.stats import beta as _beta

    if k == 0:  # one-sided: all of alpha goes to the upper tail
        return 0.0, float(_beta.ppf(confidence, k + 1, n - k + 1))
    if k == n:
        return float(_beta.ppf(alpha, k + 1, n - k + 1)), 1.0
    lo, hi = _beta.ppf([alpha / 2, 1 - alpha / 2], k + 1, n - k + 1)
    return float(lo), float(hi)


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval -- the dependency-free fallback."""
    if n <= 0:
        return 0.0, 1.0
    z = _Z95 if abs(confidence - 0.95) < 1e-9 else _z_for(confidence)
    k = max(0, min(int(k), int(n)))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    # Snap the boundaries exactly: rounding leaves hi at 0.9999999999999999 for
    # k == n, which puts the observed 1.0 outside its own interval.
    return (0.0 if k == 0 else lo), (1.0 if k == n else hi)


def _z_for(confidence: float) -> float:
    """Inverse normal CDF via bisection -- avoids a scipy dependency."""
    target = 1.0 - (1.0 - confidence) / 2.0
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2)))
        if cdf < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def pass_at_k(successes: int, trials: int, k: int) -> Optional[float]:
    """P(at least one of k sampled trials succeeds) -- the discovery metric.

    Unbiased hypergeometric estimator: 1 - C(n-c, k)/C(n, k). Returns None
    when there are fewer than k trials to sample from.
    """
    n, c = int(trials), int(successes)
    if k <= 0 or n < k:
        return None
    if n - c < k:
        return 1.0
    miss = 1.0
    for i in range(k):
        miss *= (n - c - i) / (n - i)
    return 1.0 - miss


def pass_hat_k(successes: int, trials: int, k: int) -> Optional[float]:
    """P(ALL k sampled trials succeed) -- the reliability metric.

    C(c, k)/C(n, k). Monotonically non-increasing in k, and 0 as soon as
    c < k. This is the number that matters for deployability: a manipulation
    policy succeeding 4 times in 5 is not shippable, and pass^1 alone hides
    that completely.
    """
    n, c = int(trials), int(successes)
    if k <= 0 or n < k:
        return None
    if c < k:
        return 0.0
    return math.comb(c, k) / math.comb(n, k)


def aggregate_over_tasks(
    per_task: Iterable[tuple[int, int]],
    k: int,
    metric: str = "pass_hat_k",
) -> Optional[float]:
    """Average pass@k / pass^k over tasks, skipping tasks with < k trials.

    ``per_task`` yields (successes, trials) pairs. Averaging per task rather
    than pooling episodes keeps a task with many rollouts from dominating.
    """
    fn = pass_hat_k if metric == "pass_hat_k" else pass_at_k
    vals = [v for s, n in per_task if (v := fn(s, n, k)) is not None]
    return sum(vals) / len(vals) if vals else None


def resolution_ratio(trials_per_arm: int, mde_pp: float, baseline: float = 0.5) -> float:
    """q = N / N*, where N* resolves a difference of ``mde_pp`` percentage points.

    A pair with q < 1 is an unresolved comparison and should be rendered as a
    tie rather than a rank. Two-proportion sample size at 95% / 80% power.
    """
    if mde_pp <= 0:
        return float("inf")
    delta = mde_pp / 100.0
    p1 = min(max(baseline, 1e-6), 1 - 1e-6)
    p2 = min(max(p1 + delta, 1e-6), 1 - 1e-6)
    z_a, z_b = _Z95, 0.8416212335729143  # 80% power
    n_star = ((z_a + z_b) ** 2) * (p1 * (1 - p1) + p2 * (1 - p2)) / (delta ** 2)
    return trials_per_arm / n_star if n_star > 0 else float("inf")


def mcnemar(both: int, only_a: int, only_b: int, neither: int = 0) -> dict:
    """Paired binary comparison -- the right test for two models on shared tasks.

    Uses only the discordant pairs, which is exactly why pairing buys power:
    tasks both models get right (or wrong) carry no information about which
    is better. Returns the chi-square statistic with continuity correction
    and a two-sided p-value.
    """
    b, c = int(only_a), int(only_b)
    n_disc = b + c
    if n_disc == 0:
        return {"discordant": 0, "statistic": 0.0, "p_value": 1.0}
    stat = (abs(b - c) - 1) ** 2 / n_disc  # Edwards continuity correction
    stat = max(stat, 0.0)
    # two-sided p from the chi-square(1) survival function == erfc(sqrt(x/2))
    p = math.erfc(math.sqrt(stat / 2.0))
    return {"discordant": n_disc, "statistic": stat, "p_value": min(1.0, max(0.0, p))}


def rank_interval(
    scores: Sequence[float], intervals: Sequence[tuple[float, float]]
) -> list[tuple[int, int]]:
    """Rank ranges from non-overlapping intervals, so ties read as ties.

    A model's best possible rank is 1 + the number of models whose interval
    lies strictly above its own; its worst is M - the number strictly below.
    Printing 1, 2, 3, 4 for four statistically indistinguishable models
    invents an ordering the data does not support.
    """
    m = len(scores)
    out: list[tuple[int, int]] = []
    for i in range(m):
        lo_i, hi_i = intervals[i]
        better = sum(1 for j in range(m) if j != i and intervals[j][0] > hi_i)
        worse = sum(1 for j in range(m) if j != i and intervals[j][1] < lo_i)
        out.append((1 + better, m - worse))
    return out
