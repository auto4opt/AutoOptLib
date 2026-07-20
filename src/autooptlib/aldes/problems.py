"""Problem adapters used by the published ALDes experiments."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np

_PBO_DIMENSIONS = {1: 100, 2: 225, 3: 400, 4: 625, 5: 90}


def make_pbo_problem(problem_id: int, *, ioh_instance: int = 1):
    """Create an AutoOptLib problem callable for an IOH PBO function.

    ``ioh`` remains an optional dependency and is imported only when the
    returned problem is constructed.  IOH's PBO functions are maximization
    problems; their values are negated for AutoOptLib's minimization contract.
    """

    function_id = int(problem_id)
    if not 1 <= function_id <= 23:
        raise ValueError("ALDes PBO problem_id must be in the range 1..23.")

    def pbo(problems: Iterable[Any], instances: Sequence[int], mode: str):
        normalized_mode = str(mode).lower()
        if normalized_mode == "construct":
            try:
                import ioh
            except ImportError as exc:  # pragma: no cover - dependency specific
                raise ImportError(
                    "IOH is required for ALDes PBO experiments. Install "
                    "AutoOptLib with `pip install 'autooptlib[aldes]'`."
                ) from exc
            problem_list = list(problems)
            data: list[SimpleNamespace] = []
            for problem, instance in zip(problem_list, instances):
                try:
                    dimension = _PBO_DIMENSIONS[int(instance)]
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "PBO instances must be one of 1, 2, 3, 4, or 5."
                    ) from exc
                objective = ioh.get_problem(
                    function_id,
                    instance=int(ioh_instance),
                    dimension=dimension,
                    problem_class=ioh.ProblemClass.PBO,
                )
                problem.type = ["discrete", "static", "certain"]
                problem.bound = np.vstack(
                    (np.zeros(dimension, dtype=int), np.ones(dimension, dtype=int))
                )
                problem.dimension = dimension
                problem.name = f"ioh_pbo_f{function_id}"

                def evaluate(entry, decision):
                    value = entry.objective(np.asarray(decision, dtype=int))
                    return -float(value), 0.0, None

                problem.evaluate = evaluate
                data.append(SimpleNamespace(objective=objective))
            return problem_list, data, None

        if normalized_mode == "repair":
            return np.asarray(instances, dtype=int), None, None

        if normalized_mode == "evaluate":
            data = problems
            decisions = np.asarray(instances, dtype=int)
            single = decisions.ndim == 1
            decisions = np.atleast_2d(decisions)
            values = np.asarray(
                [-float(data.objective(decision)) for decision in decisions]
            )
            if single:
                return float(values[0]), 0.0, None
            return values, np.zeros_like(values), None

        raise ValueError(f"Unsupported problem mode: {mode!r}")

    pbo.__name__ = f"ioh_pbo_f{function_id}"
    return pbo


__all__ = ["make_pbo_problem"]
