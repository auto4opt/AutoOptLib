from __future__ import annotations

from collections.abc import Mapping
from typing import Any, List, Sequence

import numpy as np

from ..solve import run_design
from ._helpers import Pathway, PathwayParam, get_flex, get_problem_type, problem_list


def _ensure_matrix(matrix: np.ndarray, rows: int, cols: int) -> np.ndarray:
    current_rows, current_cols = matrix.shape if matrix.ndim == 2 else (0, 0)
    target_rows = max(rows, current_rows)
    target_cols = max(cols, current_cols)
    if matrix.shape == (target_rows, target_cols):
        return matrix
    new_matrix = np.zeros((target_rows, target_cols))
    if matrix.size:
        r, c = matrix.shape
        new_matrix[:r, :c] = matrix
    return new_matrix


def _auc_score(
    fit_history: Sequence[float], tmax: Any, thres: Any, population_size: int
) -> float:
    """Reference AUC: reciprocal fraction of FE checkpoints meeting targets."""
    history = np.asarray(fit_history, dtype=float).reshape(-1)
    times = np.asarray(tmax if tmax is not None else [], dtype=float).reshape(-1)
    thresholds = np.asarray(thres if thres is not None else [], dtype=float).reshape(-1)
    if history.size == 0 or times.size == 0 or thresholds.size != times.size:
        return np.inf
    indices = np.ceil(times / max(1, population_size)).astype(int) - 1
    indices = np.clip(indices, 0, history.size - 1)
    success_fraction = float(np.mean(history[indices] <= thresholds))
    return float(1.0 / (success_fraction + np.finfo(float).eps))


def _update_sequential(problem, data, best_solution):
    advance = getattr(problem, "advance_sequence", None)
    if callable(advance):
        return advance(best_solution, data)
    name = getattr(problem, "name", None)
    if isinstance(name, str):
        try:
            module, func = name.rsplit(".", 1)
            module_obj = __import__(module, fromlist=[func])
            next_fn = getattr(module_obj, func)
            return next_fn(problem, data, best_solution, "sequence")
        except Exception:
            return problem, data
    return problem, data


def _select_initial_population(
    populations: Any,
    instance_index: int,
    run: int,
    *,
    instance_count: int,
    runs: int,
) -> np.ndarray | None:
    """Select an optional ``(population, dimension)`` matrix for one run."""

    if populations is None:
        return None
    if isinstance(populations, Mapping):
        if instance_index not in populations:
            return None
        selected = np.asarray(populations[instance_index])
        if selected.ndim == 2:
            return selected
        if selected.ndim == 3:
            return selected[run % selected.shape[0]]
        raise ValueError(
            "Each InitialPopulations mapping value must have shape (N,D) "
            "or (runs,N,D)."
        )
    array = np.asarray(populations)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        if array.shape[0] == instance_count and instance_count != runs:
            return array[instance_index]
        return array[run % array.shape[0]]
    if array.ndim == 4:
        return array[instance_index, run % array.shape[1]]
    raise ValueError(
        "InitialPopulations must have shape (N,D), (runs,N,D), or "
        "(instances,runs,N,D)."
    )


def evaluate(self, problem: Any, data: Any, setting: Any, seed_instance: Sequence[int]):
    if not self.operator_pheno:
        return self

    pathways: List[Pathway] = self.operator_pheno[0]
    params: List[PathwayParam] = self.parameter_pheno[0]

    problems = problem_list(problem)
    data_list = problem_list(data) if data is not None else [None] * len(problems)

    evaluate_mode = get_flex(setting, "evaluate", "exact")
    metric = get_flex(setting, "metric", "quality")
    runs = int(get_flex(setting, "alg_runs", 1))
    initial_populations = get_flex(setting, "InitialPopulations", None)

    max_seed = max(seed_instance) if seed_instance else 0
    rows_needed = max_seed + 1

    if evaluate_mode == "approximate":
        self.performance_approx = _ensure_matrix(
            self.performance_approx, rows_needed, runs
        )
        target = self.performance_approx
    else:
        self.performance = _ensure_matrix(self.performance, rows_needed, runs)
        target = self.performance

    for seed in seed_instance:
        problem_obj = problems[seed] if seed < len(problems) else problems[-1]
        data_obj = data_list[seed] if seed < len(data_list) else None
        problem_type = get_problem_type(problem_obj) or "continuous"
        behavior = get_flex(problem_obj, "type", [problem_type, "static"])
        mode = (
            behavior[1]
            if isinstance(behavior, Sequence) and len(behavior) > 1
            else "static"
        )

        for run in range(runs):
            initial_population = _select_initial_population(
                initial_populations,
                seed,
                run,
                instance_count=len(problems),
                runs=runs,
            )
            if mode == "static":
                had_initial = hasattr(setting, "_InitialPopulation")
                previous_initial = getattr(setting, "_InitialPopulation", None)
                setting._InitialPopulation = initial_population
                try:
                    result = run_design(
                        pathways, params, problem_obj, data_obj, setting
                    )
                finally:
                    if had_initial:
                        setting._InitialPopulation = previous_initial
                    else:
                        delattr(setting, "_InitialPopulation")
                fit_history = result["fit_history"]
                evaluations = result["evaluations"]
                elapsed = result["elapsed"]

                if metric == "quality":
                    value = float(fit_history[-1]) if fit_history else np.inf
                elif metric == "auc":
                    value = _auc_score(
                        fit_history,
                        get_flex(setting, "Tmax", None),
                        get_flex(setting, "Thres", None),
                        int(get_flex(problem_obj, "N", get_flex(setting, "ProbN", 1))),
                    )
                elif metric == "runtimeFE":
                    value = evaluations
                elif metric == "runtimeSec":
                    value = elapsed
                else:
                    value = float(fit_history[-1]) if fit_history else np.inf
                target[seed, run] = value

            elif mode == "sequential":
                cumulative = 0.0
                evaluations = 0
                elapsed = 0.0
                curr_prob = problem_obj
                curr_data = data_obj
                had_initial = hasattr(setting, "_InitialPopulation")
                previous_initial = getattr(setting, "_InitialPopulation", None)
                stage_initial = initial_population
                try:
                    while getattr(curr_data, "continue", False):
                        # The MATLAB sequential protocol allocates the configured
                        # ProbFE budget independently to every arriving stage.
                        setting._InitialPopulation = stage_initial
                        result = run_design(
                            pathways, params, curr_prob, curr_data, setting
                        )
                        stage_initial = None
                        fit_history = result["fit_history"]
                        best_solution = result["best_solution"]
                        evaluations += result["evaluations"]
                        elapsed += result["elapsed"]
                        if metric == "quality":
                            cumulative += (
                                float(fit_history[-1]) if fit_history else np.inf
                            )
                        elif metric in {"runtimeFE", "runtimeSec"}:
                            cumulative += (
                                result["elapsed"]
                                if metric == "runtimeSec"
                                else result["evaluations"]
                            )
                        elif metric == "auc":
                            cumulative += _auc_score(
                                fit_history,
                                get_flex(setting, "Tmax", None),
                                get_flex(setting, "Thres", None),
                                int(
                                    get_flex(
                                        curr_prob,
                                        "N",
                                        get_flex(setting, "ProbN", 1),
                                    )
                                ),
                            )
                        else:
                            cumulative += (
                                float(fit_history[-1]) if fit_history else np.inf
                            )
                        if best_solution is None:
                            break
                        curr_prob, curr_data = _update_sequential(
                            curr_prob, curr_data, best_solution
                        )
                        if curr_prob is problem_obj and curr_data is data_obj:
                            break
                finally:
                    if had_initial:
                        setting._InitialPopulation = previous_initial
                    else:
                        delattr(setting, "_InitialPopulation")
                if metric == "runtimeFE":
                    target[seed, run] = evaluations
                elif metric == "runtimeSec":
                    target[seed, run] = elapsed
                else:
                    target[seed, run] = cumulative
            else:
                raise NotImplementedError(f"Unsupported problem type behavior: {mode}")

    if evaluate_mode == "approximate":
        self.performance_approx = target
    else:
        self.performance = target

    return self
