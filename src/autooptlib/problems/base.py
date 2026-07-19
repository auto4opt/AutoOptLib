"""Public contracts and helpers for user-defined optimization problems."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Protocol, Sequence, Union, runtime_checkable

import numpy as np

Objective = Callable[[np.ndarray, Any], float]
Constraint = Callable[[np.ndarray, Any], Any]
Bounds = Union[
    np.ndarray,
    Sequence[Sequence[float]],
    Callable[[Any], Any],
    None,
]


@runtime_checkable
class ProblemDefinition(Protocol):
    """Mode-based callable accepted by :func:`autooptlib.autoopt`.

    ``construct`` receives mutable problem records plus user instance
    identifiers and returns ``(problems, data, auxiliary)``.  Constructed
    problem records must expose ``type``, ``bound``, and an
    ``evaluate(data, decision)`` callable.
    """

    def __call__(
        self, problems: Iterable[Any], instances: Sequence[Any], mode: str
    ) -> tuple[Any, Any, Any]: ...


def _resolve_bound(bounds: Bounds, instance: Any, problem_type: str) -> np.ndarray:
    raw = bounds(instance) if callable(bounds) else bounds
    if problem_type == "permutation":
        if raw is None:
            dimension = int(instance)
            domain = np.arange(1, dimension + 1)
            return np.vstack((domain, domain))
    if raw is None:
        raise ValueError(f"{problem_type} problems require explicit bounds.")

    array = np.asarray(raw, dtype=float if problem_type == "continuous" else int)
    if array.ndim == 1 and array.size == 2:
        try:
            dimension = int(instance)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Scalar lower/upper bounds require an integer instance dimension."
            ) from exc
        array = np.vstack((np.full(dimension, array[0]), np.full(dimension, array[1])))
    return array


def make_problem(
    objective: Objective,
    *,
    bounds: Bounds = None,
    problem_type: str = "continuous",
    constraint: Constraint | None = None,
    data_factory: Callable[[Any], Any] | None = None,
    name: str | None = None,
) -> ProblemDefinition:
    """Create an AutoOptLib problem definition from ordinary Python callables.

    Parameters
    ----------
    objective:
        Callable ``objective(decision, data) -> scalar`` to minimize.
    bounds:
        A ``(2, dimension)`` array, a ``(lower, upper)`` pair expanded using
        an integer instance dimension, or a callable returning bounds for an
        instance.  Permutation problems may omit bounds when instances are
        dimensions.
    problem_type:
        ``continuous``, ``discrete``, or ``permutation``.
    constraint:
        Optional callable returning values where positive entries represent
        violations.
    data_factory:
        Optional callable mapping each instance identifier to objective data.
        Without it, the instance identifier itself is passed as data.
    """
    if not callable(objective):
        raise TypeError("objective must be callable")
    problem_type = str(problem_type).lower()
    if problem_type not in {"continuous", "discrete", "permutation"}:
        raise ValueError("problem_type must be continuous, discrete, or permutation")
    if constraint is not None and not callable(constraint):
        raise TypeError("constraint must be callable")
    if data_factory is not None and not callable(data_factory):
        raise TypeError("data_factory must be callable")

    problem_name = str(name or getattr(objective, "__name__", "user_problem"))

    def evaluate_one(data: Any, decision: np.ndarray) -> tuple[Any, Any, None]:
        decision = np.asarray(decision)
        # Keep the raw result so the central evaluator can reject vectors and
        # non-finite scalars consistently for built-in and user problems.
        value = objective(decision, data)
        violation = 0.0 if constraint is None else constraint(decision, data)
        return value, violation, None

    def definition(problems: Iterable[Any], instances: Sequence[Any], mode: str):
        normalized_mode = str(mode).lower()
        if normalized_mode == "construct":
            problem_list = list(problems)
            if len(problem_list) != len(instances):
                raise ValueError(
                    "The number of problem records and instances must match."
                )
            data_entries: list[Any] = []
            for problem, instance in zip(problem_list, instances):
                bound = _resolve_bound(bounds, instance, problem_type)
                problem.type = [problem_type, "static", "certain"]
                problem.bound = bound
                problem.dimension = int(bound.shape[1]) if bound.ndim == 2 else 0
                problem.name = problem_name
                problem.evaluate = evaluate_one
                data_entries.append(
                    data_factory(instance) if data_factory is not None else instance
                )
            return problem_list, data_entries, None
        if normalized_mode == "repair":
            return np.asarray(instances), None, None
        if normalized_mode == "evaluate":
            data = problems
            decisions = np.asarray(instances)
            single = decisions.ndim == 1
            decisions = np.atleast_2d(decisions)
            values = []
            violations = []
            for decision in decisions:
                value, violation, _ = evaluate_one(data, decision)
                values.append(value)
                violations.append(violation)
            if single:
                return values[0], violations[0], None
            return np.asarray(values), np.asarray(violations), None
        raise ValueError(f"Unsupported problem mode: {mode!r}")

    definition.__name__ = problem_name
    return definition


def validate_constructed_problems(problems: Sequence[Any], data: Sequence[Any]) -> None:
    """Validate the runtime problem contract with actionable error messages."""
    if not problems:
        raise ValueError("Problem construction returned no problem instances.")
    if len(data) != len(problems):
        raise ValueError(
            f"Problem construction returned {len(problems)} problems but {len(data)} data entries."
        )
    for index, problem in enumerate(problems):
        type_field = getattr(problem, "type", None)
        if (
            not isinstance(type_field, Sequence)
            or isinstance(type_field, (str, bytes))
            or not type_field
        ):
            raise TypeError(
                f"Problem {index} must define a non-empty sequence attribute 'type'."
            )
        problem_type = str(type_field[0]).lower()
        if problem_type not in {"continuous", "discrete", "permutation"}:
            raise ValueError(
                f"Problem {index} has unsupported type {problem_type!r}; expected continuous, "
                "discrete, or permutation."
            )
        bound = np.asarray(getattr(problem, "bound", None))
        if bound.ndim != 2 or bound.shape[0] != 2 or bound.shape[1] == 0:
            raise ValueError(
                f"Problem {index} bound must have shape (2, dimension), got {bound.shape}."
            )
        if not np.all(np.isfinite(bound)):
            raise ValueError(f"Problem {index} bounds must be finite.")
        if np.any(bound[0] > bound[1]):
            raise ValueError(
                f"Problem {index} has lower bounds greater than upper bounds."
            )
        if not callable(getattr(problem, "evaluate", None)):
            raise TypeError(
                f"Problem {index} must expose evaluate(data, decision) after construction."
            )
        for attribute in ("N", "Gmax"):
            value = getattr(problem, attribute, None)
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(
                    f"Problem {index} attribute {attribute} must be positive."
                )
