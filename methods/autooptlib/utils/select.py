"""Selection utilities mirroring MATLAB Select.m."""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence

import numpy as np

from .design import Design
from .design._helpers import ensure_rng, get_flex
from .design._search_control import (
    performance_diversity_select,
    performance_plateau_select,
)
from .general.select_stats import friedman_nemenyi, irace_friedman_race


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
            ledger = (getattr(alg, "metadata", {}) or {}).get("evaluation_ledger", [])
            if ledger:
                completed = {
                    (int(item["instance_index"]), int(item["run"]))
                    for item in ledger
                    if "instance_index" in item and "run" in item
                }
                required = {
                    (int(seed), run)
                    for run in range(int(get_flex(setting, "alg_runs", 1)))
                }
                missing = not required.issubset(completed)
            else:
                # Compatibility for old in-memory Design objects that predate
                # the evaluation ledger. New runs must use the ledger because
                # an objective value of exactly zero is a valid result.
                missing = bool(np.sum(row) == 0)
            if missing:
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


def racing_survivors(
    algs: Iterable[Design],
    setting: Any,
    seed_instance: Sequence[int],
    *,
    final: bool = False,
) -> List[Design]:
    """Apply irace's native Friedman/Wilcoxon racing elimination rule.

    Evidence accumulates over common problem-instance/run blocks.  Friedman is
    the omnibus gate for more than two candidates; the two-candidate case uses
    paired Wilcoxon.  At the final block, unresolved candidates are ordered
    deterministically so the design population has exactly ``AlgN`` members.
    """

    alg_list = _ensure_design_list(algs)
    target = min(int(get_flex(setting, "alg_n", len(alg_list))), len(alg_list))
    if len(alg_list) <= target:
        return alg_list
    matrix = _collect_performance(alg_list, setting, seed_instance)
    means = np.mean(matrix, axis=0)
    alpha = float(get_flex(setting, "alpha", 0.05))
    rank_sums, alive, _ = irace_friedman_race(matrix, alpha)
    order = sorted(
        range(len(alg_list)),
        key=lambda i: (float(rank_sums[i]), float(means[i]), i),
    )
    survivors = [i for i in range(len(alg_list)) if bool(alive[i])]
    if len(survivors) < target:
        selected = set(survivors)
        survivors.extend(i for i in order if i not in selected)
    if final:
        survivors = sorted(
            survivors,
            key=lambda i: (float(rank_sums[i]), float(means[i]), i),
        )[:target]
    return [alg_list[i] for i in survivors]


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
        scorer = get_flex(setting, "eval_training_scorer", None)
        if scorer is None:
            averages = np.mean(all_perf, axis=0)
        else:
            averages = np.asarray(
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
        order = np.argsort(averages)
        if evaluate_mode in {"exact", "approximate", "racing"}:
            if bool(get_flex(setting, "_search_plateau_novelty", False)):
                return performance_plateau_select(
                    alg_list,
                    count=min(alg_n, len(alg_list)),
                    costs=averages,
                    global_best=get_flex(setting, "_search_global_best", None),
                )
            diversity_weight = float(get_flex(setting, "search_diversity_weight", 0.0))
            if diversity_weight > 0 and alg_n > 1:
                return performance_diversity_select(
                    alg_list,
                    count=min(alg_n, len(alg_list)),
                    diversity_weight=diversity_weight,
                    global_best=get_flex(setting, "_search_global_best", None),
                    costs=averages,
                )
            top = order[: min(alg_n, len(order))]
            return [alg_list[i] for i in top]
        if evaluate_mode == "intensification":
            if alg_n != 1:
                raise ValueError("SMAC intensification requires AlgN=1.")
            return [alg_list[int(order[0])]]
        raise NotImplementedError(f"Unsupported evaluate mode: {evaluate_mode}")

    if compare == "statistic":
        wins = _statistic_wins(all_perf, alpha)
        order = np.argsort(-wins)
        if evaluate_mode in {"exact", "approximate"}:
            diversity_weight = float(get_flex(setting, "search_diversity_weight", 0.0))
            if diversity_weight > 0 and alg_n > 1:
                return performance_diversity_select(
                    alg_list,
                    count=min(alg_n, len(alg_list)),
                    diversity_weight=diversity_weight,
                    global_best=get_flex(setting, "_search_global_best", None),
                    costs=-wins.astype(float),
                )
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
            raise ValueError('SMAC intensification requires Compare="average".')
        raise NotImplementedError(f"Unsupported evaluate mode: {evaluate_mode}")

    raise NotImplementedError(f"Unsupported compare mode: {compare}")
