"""Statistical utilities for algorithm selection."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

import numpy as np


def _rankdata(row: np.ndarray) -> np.ndarray:
    order = np.argsort(row)
    ranks = np.empty_like(order, dtype=float)
    sorted_row = row[order]
    i = 0
    n = len(row)
    while i < n:
        j = i
        while j + 1 < n and sorted_row[j + 1] == sorted_row[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def friedman_nemenyi(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Input to friedman_nemenyi must be 2-D")
    n, k = matrix.shape
    if n < 2 or k < 2:
        raise ValueError(
            "Friedman test requires at least two algorithms and two instances"
        )

    ranks = np.apply_along_axis(_rankdata, 1, matrix)
    avg_ranks = ranks.mean(axis=0)
    q_denom = math.sqrt(k * (k + 1) / (6.0 * n))
    if q_denom == 0:
        q_denom = 1.0

    p_matrix = np.full((k, k), np.nan, dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            diff = avg_ranks[i] - avg_ranks[j]
            q = diff / q_denom
            # The Nemenyi statistic uses the Studentized-range distribution
            # divided by sqrt(2), not an unadjusted pairwise normal tail.  The
            # latter materially understates p-values when comparing >2 methods.
            p = _studentized_range_sf_infinite(abs(q) * math.sqrt(2.0), k)
            p_matrix[i, j] = p_matrix[j, i] = p
    return avg_ranks, p_matrix


def irace_friedman_race(
    matrix: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply irace's Friedman/Wilcoxon elimination rule.

    With more than two candidates this mirrors ``aux2_friedman`` in irace:
    a tie-corrected Friedman omnibus test gates Conover's comparison against
    the best rank sum.  With two candidates it uses a paired Wilcoxon
    signed-rank test and, unlike irace's observed-dominance shortcut, requires
    the configured significance threshold for every elimination.  The
    returned Boolean mask marks statistically surviving candidates.
    """

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("A race requires a 2-D matrix with at least two candidates")
    if values.shape[0] < 1 or not np.all(np.isfinite(values)):
        raise ValueError("A race requires at least one finite result block")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")

    n, k = values.shape
    ranks = np.apply_along_axis(_rankdata, 1, values)
    rank_sums = ranks.sum(axis=0)
    survivors = np.ones(k, dtype=bool)

    if k == 2:
        diffs = values[:, 0] - values[:, 1]
        # Every elimination must satisfy the configured significance level.
        # In particular, do not treat one-sided observed dominance as a
        # zero-p-value shortcut: with only a few blocks it is not significant,
        # and an all-tied race must keep both candidates.
        if np.all(diffs == 0):
            return rank_sums, survivors, 1.0
        p_value, pseudo_median = _paired_wilcoxon(diffs)
        if p_value < alpha:
            survivors[1 if pseudo_median <= 0 else 0] = False
        return rank_sums, survivors, p_value

    # irace does not run the post-hoc comparison until at least two blocks are
    # available because its variance estimate contains (n - 1).
    if n < 2:
        return rank_sums, survivors, 1.0

    tie_correction = 0.0
    for row in ranks:
        _, counts = np.unique(row, return_counts=True)
        tie_correction += float(np.sum(counts**3 - counts))
    denominator = n * k * (k + 1) - tie_correction / (k - 1)
    if denominator <= np.finfo(float).eps:
        return rank_sums, survivors, 1.0
    centered = rank_sums - n * (k + 1) / 2.0
    statistic = 12.0 * float(np.sum(centered**2)) / denominator
    p_value = _chi_square_sf(statistic, k - 1)
    if not np.isfinite(p_value) or p_value >= alpha:
        return rank_sums, survivors, p_value

    sum_squared_ranks = float(np.sum(ranks**2))
    variance = (
        2.0
        * (n * sum_squared_ranks - float(np.sum(rank_sums**2)))
        / ((n - 1) * (k - 1))
    )
    threshold = _student_t_ppf(1.0 - alpha / 2.0, (n - 1) * (k - 1)) * math.sqrt(
        max(0.0, variance)
    )
    order = np.argsort(rank_sums, kind="stable")
    best_sum = float(rank_sums[order[0]])
    keep = []
    for index in order:
        if abs(float(rank_sums[index]) - best_sum) > threshold:
            break
        keep.append(int(index))
    survivors[:] = False
    survivors[keep] = True
    return rank_sums, survivors, p_value


def _paired_wilcoxon(differences: np.ndarray) -> tuple[float, float]:
    """Two-sided paired Wilcoxon p-value and Hodges-Lehmann pseudomedian."""

    original = np.asarray(differences, dtype=float)
    had_zero = bool(np.any(original == 0))
    nonzero = original[original != 0]
    if nonzero.size == 0:
        return 1.0, 0.0
    absolute = np.abs(nonzero)
    ranks = _rankdata(absolute)
    positive = float(np.sum(ranks[nonzero > 0]))
    total = float(np.sum(ranks))
    has_ties = len(np.unique(absolute)) != len(absolute)
    # R's wilcox.test uses the exact distribution only for fewer than 50
    # nonzero pairs without zero differences or tied absolute ranks.
    if len(nonzero) < 50 and not had_zero and not has_ties:
        observed = int(round(min(positive, total - positive)))
        counts = np.zeros(int(total) + 1, dtype=np.int64)
        counts[0] = 1
        upper = 0
        for rank in range(1, len(nonzero) + 1):
            counts[rank : upper + rank + 1] += counts[: upper + 1]
            upper += rank
        p_value = min(
            1.0, 2.0 * float(np.sum(counts[: observed + 1])) / (2 ** len(nonzero))
        )
    else:
        _, tie_counts = np.unique(absolute, return_counts=True)
        mean = total / 2.0
        variance = (
            len(nonzero) * (len(nonzero) + 1) * (2 * len(nonzero) + 1)
            - 0.5 * float(np.sum(tie_counts**3 - tie_counts))
        ) / 24.0
        if variance <= 0:
            p_value = 1.0
        else:
            z = max(0.0, abs(positive - mean) - 0.5) / math.sqrt(variance)
            p_value = min(1.0, 2.0 * _normal_sf(z))
    walsh = (nonzero[:, None] + nonzero[None, :]) / 2.0
    pseudo_median = float(np.median(walsh[np.triu_indices(len(nonzero))]))
    return p_value, pseudo_median


def _chi_square_sf(value: float, degrees: int) -> float:
    if value <= 0:
        return 1.0
    return _regularized_gamma_q(degrees / 2.0, value / 2.0)


def _regularized_gamma_q(shape: float, value: float) -> float:
    """Regularized upper incomplete gamma, using stable NR expansions."""

    if value < 0 or shape <= 0:
        raise ValueError("Invalid incomplete-gamma arguments")
    if value == 0:
        return 1.0
    eps = 3e-14
    tiny = np.finfo(float).tiny / eps
    if value < shape + 1.0:
        term = 1.0 / shape
        total = term
        ap = shape
        for _ in range(10000):
            ap += 1.0
            term *= value / ap
            total += term
            if abs(term) <= abs(total) * eps:
                break
        lower = total * math.exp(-value + shape * math.log(value) - math.lgamma(shape))
        return float(np.clip(1.0 - lower, 0.0, 1.0))
    b = value + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / b
    fraction = d
    for i in range(1, 10000):
        an = -i * (i - shape)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) <= eps:
            break
    return float(
        np.clip(
            math.exp(-value + shape * math.log(value) - math.lgamma(shape)) * fraction,
            0.0,
            1.0,
        )
    )


def _beta_continued_fraction(a: float, b: float, value: float) -> float:
    maximum = 10000
    eps = 3e-14
    tiny = np.finfo(float).tiny / eps
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * value / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for m in range(1, maximum + 1):
        m2 = 2 * m
        aa = m * (b - m) * value / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        aa = -(a + m) * (qab + m) * value / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= eps:
            break
    return result


def _regularized_beta(value: float, a: float, b: float) -> float:
    if value <= 0:
        return 0.0
    if value >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(value)
        + b * math.log1p(-value)
    )
    if value < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, value) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - value) / b


def _student_t_cdf(value: float, degrees: int) -> float:
    beta = _regularized_beta(degrees / (degrees + value * value), degrees / 2.0, 0.5)
    return 1.0 - beta / 2.0 if value >= 0 else beta / 2.0


def _student_t_ppf(probability: float, degrees: int) -> float:
    if degrees <= 0 or not 0.0 < probability < 1.0:
        raise ValueError("Invalid Student-t quantile arguments")
    if probability == 0.5:
        return 0.0
    if probability < 0.5:
        return -_student_t_ppf(1.0 - probability, degrees)
    lower, upper = 0.0, 1.0
    while _student_t_cdf(upper, degrees) < probability:
        upper *= 2.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if _student_t_cdf(midpoint, degrees) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _normal_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2))


@lru_cache(maxsize=8)
def _hermite_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _studentized_range_sf_infinite(value: float, groups: int) -> float:
    """Survival function of Tukey's range for infinite degrees of freedom.

    For independent standard normals, ``P(range <= r)`` equals
    ``k * integral(phi(x) * (Phi(x+r)-Phi(x))**(k-1), x)``.  Gauss-Hermite
    quadrature keeps the base package free of a mandatory SciPy dependency
    while computing the family-wise Nemenyi probability.
    """

    if groups < 2:
        raise ValueError("Studentized range requires at least two groups.")
    if not math.isfinite(value):
        return 0.0 if value > 0 else 1.0
    if value <= 0:
        return 1.0
    nodes, weights = _hermite_rule(96)
    x = math.sqrt(2.0) * nodes
    shifted = x + value
    cdf_x = np.asarray([0.5 * math.erfc(-item / math.sqrt(2.0)) for item in x])
    cdf_shifted = np.asarray(
        [0.5 * math.erfc(-item / math.sqrt(2.0)) for item in shifted]
    )
    intervals = np.clip(cdf_shifted - cdf_x, 0.0, 1.0)
    cdf = groups * float(
        np.dot(weights, intervals ** (groups - 1)) / math.sqrt(math.pi)
    )
    return float(np.clip(1.0 - cdf, 0.0, 1.0))
