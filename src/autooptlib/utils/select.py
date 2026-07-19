"""Selection utilities mirroring MATLAB Select.m."""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence

import numpy as np

from .design import Design
from .design._helpers import ensure_rng, get_flex
from .general.select_stats import friedman_nemenyi


def _ensure_design_list(algs: Iterable[Design]) -> List[Design]:
    if isinstance(algs, list):
        return algs
    return list(algs)


def _require_performance(
    algs: List[Design],
    problem: Any,
    data: Any,
    setting: Any,
    seed_instance: Sequence[int],
) -> None:
    evaluate_mode = get_flex(setting, "evaluate", "exact")
    if evaluate_mode != "racing":
        return
    for alg in algs:
        for seed in seed_instance:
            performance = getattr(alg, "performance", None)
            if performance is None:
                continue
            arr = np.asarray(performance)
            if arr.ndim == 0:
                arr = arr.reshape(1, -1)
            if seed >= arr.shape[0]:
                alg.evaluate(problem, data, setting, [seed])
                continue
            row = arr[seed]
            if np.sum(row) == 0:
                alg.evaluate(problem, data, setting, [seed])


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


def _new_algorithm_wins(matrix: np.ndarray, old_count: int, alpha: float) -> np.ndarray:
    """Count significant wins by new algorithms against incumbents only."""
    result = np.zeros(matrix.shape[1], dtype=int)
    try:
        avg_ranks, p_matrix = friedman_nemenyi(matrix)
    except ValueError:
        averages = np.mean(matrix, axis=0)
        for i in range(old_count, matrix.shape[1]):
            result[i] = int(np.any(averages[i] < averages[:old_count]))
        return result
    for i in range(old_count, matrix.shape[1]):
        for j in range(old_count):
            if (
                not np.isnan(p_matrix[i, j])
                and p_matrix[i, j] < alpha
                and avg_ranks[i] < avg_ranks[j]
            ):
                result[i] += 1
    return result


def select(
    algs: Iterable[Design],
    problem: Any,
    data: Any,
    setting: Any,
    seed_instance: Sequence[int],
) -> List[Design]:
    alg_list = _ensure_design_list(algs)
    if not alg_list:
        return []

    _require_performance(alg_list, problem, data, setting, seed_instance)
    all_perf = _collect_performance(alg_list, setting, seed_instance)

    compare = str(get_flex(setting, "compare", "average")).lower()
    evaluate_mode = str(get_flex(setting, "evaluate", "exact")).lower()
    alg_n = int(get_flex(setting, "alg_n", len(alg_list)))
    alpha = float(get_flex(setting, "alpha", 0.05))

    if compare == "average":
        averages = np.mean(all_perf, axis=0)
        order = np.argsort(averages)
        if evaluate_mode in {"exact", "approximate", "racing"}:
            top = order[: min(alg_n, len(order))]
            return [alg_list[i] for i in top]
        if evaluate_mode == "intensification":
            old_avgs = averages[:alg_n]
            threshold = float(np.max(old_avgs)) if old_avgs.size else np.inf
            return [
                alg
                for alg, avg in zip(alg_list[alg_n:], averages[alg_n:])
                if avg < threshold
            ]
        raise NotImplementedError(f"Unsupported evaluate mode: {evaluate_mode}")

    if compare == "statistic":
        wins = _statistic_wins(all_perf, alpha)
        order = np.argsort(-wins)
        if evaluate_mode in {"exact", "approximate"}:
            top = order[: min(alg_n, len(order))]
            return [alg_list[i] for i in top]
        if evaluate_mode == "racing":
            survivors = np.flatnonzero(wins > 0).tolist()
            if len(survivors) < min(alg_n, len(alg_list)):
                candidates = np.flatnonzero(wins == 0)
                needed = min(alg_n, len(alg_list)) - len(survivors)
                if needed:
                    sampled = ensure_rng(setting).choice(
                        candidates, size=needed, replace=False
                    )
                    survivors.extend(np.asarray(sampled, dtype=int).tolist())
            return [alg_list[i] for i in survivors]
        if evaluate_mode == "intensification":
            incumbent_wins = _new_algorithm_wins(all_perf, alg_n, alpha)
            return [
                alg_list[i]
                for i in range(alg_n, len(alg_list))
                if incumbent_wins[i] > 0
            ]
        raise NotImplementedError(f"Unsupported evaluate mode: {evaluate_mode}")

    raise NotImplementedError(f"Unsupported compare mode: {compare}")
