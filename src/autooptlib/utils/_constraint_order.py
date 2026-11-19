"""Shared feasibility-first ordering for evaluated candidates."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

CONSTRAINT_ORDER_REVISION = "feasibility-first-v1"


def solution_key(solution):
    """Compare solutions by feasibility, then objective or total violation."""
    read = (
        solution.get
        if isinstance(solution, Mapping)
        else lambda name: getattr(solution, name, None)
    )
    violation = read("con")
    if violation is None:
        violation = read("cons")
        if callable(violation):
            violation = violation()
    if violation is not None:
        if not np.isscalar(violation):
            violation = np.sum(np.maximum(0.0, np.asarray(violation)))
        if violation > 0:
            return True, float(violation)
    for name in ("fit", "fitness", "fits", "obj", "objs"):
        fitness = read(name)
        if fitness is not None:
            value = fitness() if callable(fitness) else fitness
            if not np.isscalar(value):
                array = np.asarray(value)
                if array.size != 1:
                    raise ValueError(
                        f"Expected a scalar-like value, got shape={array.shape}"
                    )
                value = array.item()
            return False, float(value)
    raise ValueError("A solution must expose a scalar fitness or objective.")


def selection_scores(fitness, violations):
    """Give scalar-only operators comparable scores within a population."""
    values = np.asarray(fitness, dtype=float)
    if violations is None:
        return values
    violations = np.asarray(violations, dtype=float)
    invalid = violations > 0
    if not invalid.any():
        return values
    if invalid.all():
        return violations
    # Rank each group independently to preserve exact ties and ordering at
    # every objective scale. Operators only compare these scores in this batch.
    scores = np.empty_like(values)
    feasible = ~invalid
    unique, ranks = np.unique(values[feasible], return_inverse=True)
    scores[feasible] = ranks
    _, ranks = np.unique(violations[invalid], return_inverse=True)
    scores[invalid] = len(unique) + ranks
    return scores


def objective_score(solution):
    """Objective metrics and targets require a feasible solution."""
    infeasible, score = solution_key(solution)
    return float("inf") if infeasible else score
