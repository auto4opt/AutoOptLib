"""Budgeted IOH objectives with compact best-so-far logging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from comparisons.shared.protocol import EvaluationTask
from comparisons.shared.scoring import checked_optimality_gap
from problems import IOHInstance, known_ioh_optimum_raw

BBOB_TARGETS = tuple(float(value) for value in np.logspace(2, -8, 51))


def _checkpoints(budget: int, count: int) -> np.ndarray:
    if count <= 2 or budget <= 2:
        return np.asarray(sorted({1, budget}), dtype=int)
    geometric = np.geomspace(1, budget, num=count)
    return np.asarray(sorted({1, budget, *map(int, np.ceil(geometric))}), dtype=int)


@dataclass
class ObjectiveResult:
    evaluations: int
    best_cost: float
    best_raw: float
    optimum_cost: float | None
    optimum_raw: float | None
    gap: float | None
    metric_name: str
    metric_value: float
    trajectory: list[dict[str, float | int]]
    target_hits: dict[str, int]


class TrackedIOHObjective:
    """Fresh official IOH problem instance for exactly one algorithm run."""

    def __init__(self, task: EvaluationTask, *, trajectory_points: int = 101) -> None:
        try:
            import ioh
        except ImportError as exc:  # pragma: no cover - dependency specific
            raise ImportError(
                "Install `ioh>=0.3.18,<1` to run BBOB/PBO experiments."
            ) from exc
        problem_class = (
            ioh.ProblemClass.BBOB if task.suite == "bbob" else ioh.ProblemClass.PBO
        )
        self.problem = ioh.get_problem(
            task.function_id,
            instance=task.instance,
            dimension=task.dimension,
            problem_class=problem_class,
        )
        self.problem.reset()
        self.task = task
        self.minimize = task.suite == "bbob"
        self.bounds = np.vstack((self.problem.bounds.lb, self.problem.bounds.ub))
        known_optimum = known_ioh_optimum_raw(
            task.suite,
            task.function_id,
            IOHInstance(
                dimension=task.dimension,
                instance=task.instance,
                repeat=task.repeat,
                budget=task.budget,
            ),
            objective=self.problem,
        )
        self.optimum_raw = known_optimum if np.isfinite(known_optimum) else None
        self.optimum_cost = (
            None
            if self.optimum_raw is None
            else (self.optimum_raw if self.minimize else -self.optimum_raw)
        )
        self.evaluations = 0
        self.best_cost = float("inf")
        self.best_raw = float("nan")
        self.best_decision: np.ndarray | None = None
        self._points = _checkpoints(task.budget, trajectory_points)
        self._point_index = 0
        self.trajectory: list[dict[str, float | int]] = []
        self.target_hits: dict[str, int] = {}
        self._targets = BBOB_TARGETS if task.suite == "bbob" else (0.0,)

    @property
    def dtype(self):
        return float if self.minimize else int

    def __call__(self, decision: np.ndarray, _data: Any = None) -> float:
        # Some ask/tell implementations finish the current population after
        # their internal maxfevals counter fires.  Do not charge or influence
        # the incumbent with those excess calls.
        if self.evaluations >= self.task.budget:
            return self.best_cost if np.isfinite(self.best_cost) else 1e30
        candidate = np.asarray(decision, dtype=self.dtype)
        raw = float(self.problem(candidate))
        cost = raw if self.minimize else -raw
        if self.optimum_cost is not None:
            checked_optimality_gap(cost, self.optimum_cost)
        self.evaluations += 1
        if np.isfinite(cost) and cost < self.best_cost:
            self.best_cost = cost
            self.best_raw = raw
            self.best_decision = np.array(candidate, copy=True)
            if self.optimum_cost is not None:
                gap = max(0.0, self.best_cost - self.optimum_cost)
                for target in self._targets:
                    key = f"{target:.12g}"
                    if key not in self.target_hits and gap <= target:
                        self.target_hits[key] = self.evaluations
        while self._point_index < len(self._points) and self.evaluations >= int(
            self._points[self._point_index]
        ):
            self.trajectory.append(
                {
                    "evaluations": int(self._points[self._point_index]),
                    "best_cost": float(self.best_cost),
                    "best_raw": float(self.best_raw),
                }
            )
            self._point_index += 1
        return cost

    def result(self) -> ObjectiveResult:
        if self.evaluations <= 0:
            raise RuntimeError("The optimizer did not evaluate the objective.")
        if (
            not self.trajectory
            or self.trajectory[-1]["evaluations"] != self.evaluations
        ):
            self.trajectory.append(
                {
                    "evaluations": self.evaluations,
                    "best_cost": float(self.best_cost),
                    "best_raw": float(self.best_raw),
                }
            )
        gap = (
            None
            if self.optimum_cost is None
            else max(0.0, float(self.best_cost - self.optimum_cost))
        )
        return ObjectiveResult(
            evaluations=self.evaluations,
            best_cost=float(self.best_cost),
            best_raw=float(self.best_raw),
            optimum_cost=self.optimum_cost,
            optimum_raw=self.optimum_raw,
            gap=gap,
            metric_name="gap" if gap is not None else "negated_best_raw",
            metric_value=float(gap if gap is not None else self.best_cost),
            trajectory=list(self.trajectory),
            target_hits=dict(self.target_hits),
        )


__all__ = ["BBOB_TARGETS", "ObjectiveResult", "TrackedIOHObjective"]
