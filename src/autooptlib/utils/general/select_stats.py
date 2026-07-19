"""Statistical utilities for algorithm selection."""

from __future__ import annotations

import math
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

    q_matrix = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            diff = avg_ranks[i] - avg_ranks[j]
            q = diff / q_denom
            p = 2 * _normal_sf(abs(q))
            q_matrix[i, j] = p
    return avg_ranks, q_matrix


def _normal_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2))
