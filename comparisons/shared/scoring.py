"""Dimension-safe task scoring for the registered paper protocol."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from problems import IOHInstance, known_ioh_optimum_raw


def dimension_median_mean(values: Sequence[float], dimensions: Sequence[int]) -> float:
    """One shared reducer: pooled run median per dimension, equal dimension mean.

    Inputs must already have been normalized per run against the known optimum.
    No candidate-pool scaling, nested instance median, or global run mean.
    """
    if len(values) != len(dimensions) or not len(values):
        raise ValueError("Every non-empty score vector needs dimension labels.")
    grouped: dict[int, list[float]] = {}
    for value, dimension in zip(values, dimensions):
        if not math.isfinite(float(value)):
            raise ValueError("Scores must be finite.")
        grouped.setdefault(int(dimension), []).append(float(value))
    return statistics.fmean(statistics.median(v) for v in grouped.values())


def checked_optimality_gap(value: float, optimum_cost: float) -> float:
    """Return a minimization gap and reject values beyond the known optimum."""

    cost = float(value)
    optimum = float(optimum_cost)
    if not math.isfinite(cost) or not math.isfinite(optimum):
        raise FloatingPointError("Training costs and known optima must be finite.")
    tolerance = 1e-8 * max(1.0, abs(cost), abs(optimum))
    if cost < optimum - tolerance:
        raise FloatingPointError(
            "Observed objective value is significantly better than the known "
            f"optimum: cost={cost}, optimum_cost={optimum}."
        )
    return max(0.0, cost - optimum)


def normalized_gap_from_cost(suite: str, value: float, optimum_cost: float) -> float:
    """Return the protocol's dimensionless score for one minimization cost."""

    optimum = float(optimum_cost)
    gap = checked_optimality_gap(value, optimum)
    if str(suite).lower() == "bbob":
        return math.log10(1.0 + gap)
    if str(suite).lower() == "pbo":
        return gap / max(1.0, abs(optimum))
    raise ValueError("suite must be 'bbob' or 'pbo'.")


@dataclass(frozen=True)
class TaskwiseNormalizedScorer:
    """Pickle-safe training score with fixed task normalization."""

    suite: str
    optimum_costs: tuple[float | None, ...]
    dimensions: tuple[int, ...] | None = None
    aggregation: str = "task_mean"

    def __post_init__(self) -> None:
        if self.suite not in {"bbob", "pbo"}:
            raise ValueError("suite must be 'bbob' or 'pbo'.")
        if not self.optimum_costs:
            raise ValueError("At least one training task is required.")
        if any(
            value is not None and not math.isfinite(value)
            for value in self.optimum_costs
        ):
            raise ValueError("Known training-task optima must be finite.")
        if any(value is None for value in self.optimum_costs) and self.suite != "pbo":
            raise ValueError("Unknown optima are supported only by PBO LABS.")
        if self.aggregation not in {"task_mean", "dimension_median_mean"}:
            raise ValueError("Unknown training score aggregation.")
        if self.dimensions is not None and len(self.dimensions) != len(
            self.optimum_costs
        ):
            raise ValueError("Each training task needs one dimension label.")
        if self.aggregation == "dimension_median_mean" and self.dimensions is None:
            raise ValueError("Dimension-median aggregation needs dimension labels.")

    @property
    def name(self) -> str:
        if any(value is None for value in self.optimum_costs):
            base = "taskwise_negated_raw_merit_factor"
        else:
            base = (
                "taskwise_log10_1_plus_gap"
                if self.suite == "bbob"
                else "taskwise_relative_gap"
            )
        prefix = (
            "mean_" if self.aggregation == "task_mean" else "dimension_median_mean_"
        )
        return prefix + base

    def __call__(self, values: Any, task_indices: Sequence[int] | None = None) -> float:
        array = np.asarray(values, dtype=float)
        if array.size == 0 or not np.all(np.isfinite(array)):
            return float("inf")
        scores = self.taskwise(values, task_indices)
        if self.aggregation == "task_mean":
            return float(np.mean(scores))
        assert self.dimensions is not None
        indices = (
            list(range(len(self.optimum_costs)))
            if task_indices is None
            else [int(index) for index in task_indices]
        )
        flattened = []
        labels = []
        for row, index in zip(scores, indices):
            flattened.extend(row)
            labels.extend([self.dimensions[index]] * len(row))
        return dimension_median_mean(flattened, labels)

    def taskwise(
        self, values: Any, task_indices: Sequence[int] | None = None
    ) -> np.ndarray:
        """Return normalized scores without averaging across training tasks."""

        array = np.asarray(values, dtype=float)
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise FloatingPointError("Task performance must be finite and non-empty.")
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2:
            raise ValueError(
                "Task performance must be a vector or two-dimensional matrix."
            )
        indices = (
            list(range(len(self.optimum_costs)))
            if task_indices is None
            else [int(index) for index in task_indices]
        )
        if array.shape[0] != len(indices):
            if array.size == len(indices):
                array = array.reshape(len(indices), 1)
            else:
                raise ValueError("Task scores do not match the selected task indices.")
        if not indices or min(indices) < 0 or max(indices) >= len(self.optimum_costs):
            raise IndexError(
                "Training task index is outside the scorer's optimum table."
            )
        scores = np.empty_like(array, dtype=float)
        for row_index, (row, index) in enumerate(zip(array, indices)):
            optimum = self.optimum_costs[index]
            for column, value in enumerate(row):
                scores[row_index, column] = (
                    float(value)
                    if optimum is None
                    else normalized_gap_from_cost(self.suite, value, optimum)
                )
        return scores


def make_taskwise_scorer(
    suite: str,
    function_id: int,
    instances: Sequence[Any],
    *,
    aggregation: str = "task_mean",
) -> TaskwiseNormalizedScorer:
    """Build the registered scorer from official IOH optimum values."""

    import ioh

    suite_name = str(suite).lower()
    problem_class = (
        ioh.ProblemClass.BBOB if suite_name == "bbob" else ioh.ProblemClass.PBO
    )
    optimum_costs = []
    for instance in instances:
        problem = ioh.get_problem(
            int(function_id),
            instance=int(instance.instance),
            dimension=int(instance.dimension),
            problem_class=problem_class,
        )
        task = IOHInstance(
            dimension=int(instance.dimension),
            instance=int(instance.instance),
            repeat=int(getattr(instance, "repeat", 0)),
            budget=getattr(instance, "budget", None),
        )
        optimum_raw = known_ioh_optimum_raw(
            suite_name,
            int(function_id),
            task,
            objective=problem,
        )
        if not math.isfinite(optimum_raw):
            if suite_name == "pbo" and int(function_id) == 18:
                # IOH's LABS objective is the dimensionless merit factor and
                # has no known finite optimum. Its minimization cost is already
                # ``-merit_factor``, so no fabricated optimum is required.
                optimum_costs.append(None)
                continue
            raise ValueError(
                f"{suite_name} f{function_id} has no finite optimum for "
                f"dimension={instance.dimension}, instance={instance.instance}."
            )
        optimum_costs.append(optimum_raw if suite_name == "bbob" else -optimum_raw)
    return TaskwiseNormalizedScorer(
        suite_name,
        tuple(optimum_costs),
        tuple(int(instance.dimension) for instance in instances),
        aggregation,
    )


__all__ = [
    "TaskwiseNormalizedScorer",
    "checked_optimality_gap",
    "make_taskwise_scorer",
    "normalized_gap_from_cost",
]
