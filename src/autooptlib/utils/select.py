"""Exact performance selection for evaluated algorithm candidates."""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence

import numpy as np

from .design import Design
from .design._helpers import get_flex
from .general.select_stats import friedman_nemenyi


def _ensure_design_list(algs: Iterable[Design]) -> List[Design]:
    return algs if isinstance(algs, list) else list(algs)


def _collect_performance(
    algs: List[Design], setting: Any, seed_instance: Sequence[int]
) -> np.ndarray:
    runs = int(get_flex(setting, "alg_runs", 1))
    matrix = np.zeros((len(seed_instance) * runs, len(algs)))
    for idx, alg in enumerate(algs):
        perf = alg.get_performance(setting, seed_instance)
        matrix[:, idx] = np.asarray(perf).reshape(-1)
    return matrix


def _statistic_wins(matrix: np.ndarray, alpha: float) -> np.ndarray:
    try:
        avg_ranks, p_matrix = friedman_nemenyi(matrix)
    except ValueError:
        averages = np.mean(matrix, axis=0)
        order = np.argsort(averages)
        wins = np.zeros(matrix.shape[1], dtype=int)
        if len(order):
            wins[order[0]] = matrix.shape[1] - 1
        return wins
    wins = np.zeros(matrix.shape[1], dtype=int)
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[1]):
            if i == j or np.isnan(p_matrix[i, j]):
                continue
            if p_matrix[i, j] < alpha and avg_ranks[i] < avg_ranks[j]:
                wins[i] += 1
    return wins


def select(algs, problem, data, setting, seed_instance):
    if str(get_flex(setting, "Evaluate", "exact")).lower() != "exact":
        raise ValueError("Algorithm selection supports exact evaluation only.")
    alg_list = _ensure_design_list(algs)
    if not alg_list:
        return []
    all_perf = _collect_performance(alg_list, setting, seed_instance)
    compare = str(get_flex(setting, "Compare", "average")).lower()
    count = min(int(get_flex(setting, "AlgN", len(alg_list))), len(alg_list))
    if compare == "average":
        scorer = get_flex(setting, "EvalTrainingScorer", None)
        scores = (
            np.mean(all_perf, axis=0)
            if scorer is None
            else np.asarray(
                [
                    float(
                        scorer(
                            alg.get_performance(setting, seed_instance).reshape(
                                len(seed_instance), -1
                            ),
                            seed_instance,
                        )
                    )
                    for alg in alg_list
                ],
                dtype=float,
            )
        )
    elif compare == "statistic":
        scores = -_statistic_wins(all_perf, float(get_flex(setting, "Alpha", 0.05)))
    else:
        raise NotImplementedError(f"Unsupported compare mode: {compare}")
    return [alg_list[index] for index in np.argsort(scores)[:count]]
