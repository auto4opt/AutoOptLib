"""Official IOHexperimenter adapters for BBOB and PBO experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np

IOH_OPTIMUM_METADATA_REVISION = "ioh-optimum-v2-pbo-f22-once-20260922"


@dataclass(frozen=True)
class IOHInstance:
    """One immutable IOH benchmark task.

    ``repeat`` distinguishes repeated algorithm runs over the same mathematical
    instance.  It deliberately does not change the IOH objective; AutoOptLib's
    run seed controls the stochastic optimizer independently.
    """

    dimension: int
    instance: int = 1
    repeat: int = 0
    budget: int | None = None

    def __post_init__(self) -> None:
        if int(self.dimension) <= 0:
            raise ValueError("IOH dimension must be positive.")
        if int(self.instance) <= 0:
            raise ValueError("IOH instance ID must be positive.")
        if int(self.repeat) < 0:
            raise ValueError("IOH repeat must be non-negative.")
        if self.budget is not None and int(self.budget) <= 0:
            raise ValueError("IOH evaluation budget must be positive.")


def _make_objective(suite: str, function_id: int, task: IOHInstance):
    try:
        import ioh
    except ImportError as exc:  # pragma: no cover - dependency specific
        raise ImportError(
            "Official BBOB/PBO experiments require IOHexperimenter. "
            "Install AutoOptLib with `pip install 'autooptlib[experiments]'`."
        ) from exc
    problem_class = ioh.ProblemClass.BBOB if suite == "bbob" else ioh.ProblemClass.PBO
    objective = ioh.get_problem(
        int(function_id),
        instance=int(task.instance),
        dimension=int(task.dimension),
        problem_class=problem_class,
    )
    objective.reset()
    return objective


@lru_cache(maxsize=None)
def _pbo_objective_affine(instance: int, dimension: int) -> tuple[float, float]:
    """Recover the instance's positive affine objective transformation."""

    reference = IOHInstance(dimension=dimension, instance=instance)
    adjacent = IOHInstance(dimension=dimension + 1, instance=instance)
    transformed_d = float(_make_objective("pbo", 1, reference).optimum.y)
    transformed_d1 = float(_make_objective("pbo", 1, adjacent).optimum.y)
    scale = transformed_d1 - transformed_d
    return scale, transformed_d - scale * dimension


def known_ioh_optimum_raw(
    suite: str,
    function_id: int,
    task: IOHInstance,
    *,
    objective: Any | None = None,
) -> float:
    """Return a checked, once-transformed IOH optimum.

    IOHexperimenter 0.3.22 applies the positive affine PBO objective
    transformation twice to MIS (F22) metadata, while evaluations are
    transformed once.  Recover the affine map from OneMax optima at adjacent
    dimensions, then accept either corrected metadata from a future IOH
    release or the known double-transformed 0.3.22 value.  Any other mismatch
    fails closed instead of silently fabricating a benchmark target.
    """

    suite_name = str(suite).strip().lower()
    fid = int(function_id)
    problem = (
        objective if objective is not None else _make_objective(suite_name, fid, task)
    )
    reported = float(problem.optimum.y)
    if suite_name != "pbo" or fid != 22:
        return reported

    dimension = int(task.dimension)
    even_dimension = dimension - dimension % 2
    native_optimum = (
        even_dimension // 2 if even_dimension % 4 == 0 else even_dimension // 2 + 1
    )
    if int(task.instance) == 1:
        corrected = float(native_optimum)
        repeated = corrected
    else:
        scale, shift = _pbo_objective_affine(int(task.instance), dimension)
        corrected = scale * native_optimum + shift
        repeated = scale * corrected + shift

    tolerance = 1e-8 * max(1.0, abs(reported), abs(corrected), abs(repeated))
    if math.isclose(reported, corrected, rel_tol=1e-10, abs_tol=tolerance):
        return corrected
    if math.isclose(reported, repeated, rel_tol=1e-10, abs_tol=tolerance):
        return corrected
    raise RuntimeError(
        "PBO F22 optimum metadata matches neither the once-transformed nor "
        "the known twice-transformed value: "
        f"dimension={dimension}, instance={task.instance}, "
        f"reported={reported}, corrected={corrected}, repeated={repeated}."
    )


@dataclass
class _ResidentIOHData:
    """Pickle-safe IOH state reconstructed once inside every resident worker."""

    suite: str
    function_id: int
    task: IOHInstance
    dtype: type
    minimize: bool
    optimum_raw: float
    optimum_cost: float
    objective: Any = field(repr=False, compare=False)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(vars(self))
        # pybind11 IOH problems are not pickleable.  The compact construction
        # spec above is; worker unpickling recreates one objective which then
        # remains resident for the lifetime of that isolated worker process.
        state["objective"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        vars(self).update(state)
        self.objective = _make_objective(self.suite, self.function_id, self.task)


def _coerce_instance(value: Any) -> IOHInstance:
    if isinstance(value, IOHInstance):
        return value
    if isinstance(value, dict):
        return IOHInstance(
            dimension=int(value["dimension"]),
            instance=int(value.get("instance", 1)),
            repeat=int(value.get("repeat", 0)),
            budget=(None if value.get("budget") is None else int(value["budget"])),
        )
    if isinstance(value, (tuple, list)):
        if not 1 <= len(value) <= 4:
            raise ValueError(
                "IOH tuple instances use (dimension, instance, repeat, budget)."
            )
        converted = [int(item) for item in value[:3]]
        if len(value) == 4:
            converted.append(None if value[3] is None else int(value[3]))
        return IOHInstance(*converted)
    return IOHInstance(dimension=int(value))


def make_ioh_problem(suite: str, function_id: int):
    """Return an AutoOptLib problem backed by the official ``ioh`` package.

    BBOB is minimized natively.  PBO is maximized by IOH, so the adapter
    negates its values to preserve AutoOptLib's minimization contract.  The
    original objective value and known optimum remain available in the data
    record for reporting normalized gaps.
    """

    suite_name = str(suite).strip().lower()
    limits = {"bbob": 24, "pbo": 23}
    if suite_name not in limits:
        raise ValueError("IOH suite must be 'bbob' or 'pbo'.")
    fid = int(function_id)
    if not 1 <= fid <= limits[suite_name]:
        raise ValueError(
            f"{suite_name.upper()} function_id must be in 1..{limits[suite_name]}."
        )

    def definition(problems: Iterable[Any], instances: Sequence[Any], mode: str):
        normalized_mode = str(mode).lower()
        if normalized_mode == "construct":
            problem_list = list(problems)
            if len(problem_list) != len(instances):
                raise ValueError("Problem records and IOH instances must match.")
            data: list[_ResidentIOHData] = []
            for record, raw_instance in zip(problem_list, instances):
                task = _coerce_instance(raw_instance)
                objective = _make_objective(suite_name, fid, task)
                lb = np.asarray(objective.bounds.lb)
                ub = np.asarray(objective.bounds.ub)
                record.type = [
                    "continuous" if suite_name == "bbob" else "discrete",
                    "static",
                    "certain",
                ]
                dtype = float if suite_name == "bbob" else int
                record.bound = np.vstack((lb, ub)).astype(dtype, copy=False)
                record.dimension = int(task.dimension)
                record.name = f"ioh_{suite_name}_f{fid}"

                def evaluate(entry: SimpleNamespace, decision: np.ndarray):
                    raw = float(
                        entry.objective(np.asarray(decision, dtype=entry.dtype))
                    )
                    value = raw if entry.minimize else -raw
                    return value, 0.0, None

                record.evaluate = evaluate
                optimum_raw = known_ioh_optimum_raw(
                    suite_name,
                    fid,
                    task,
                    objective=objective,
                )
                data.append(
                    _ResidentIOHData(
                        suite=suite_name,
                        function_id=fid,
                        task=task,
                        dtype=dtype,
                        minimize=suite_name == "bbob",
                        optimum_raw=optimum_raw,
                        optimum_cost=(
                            optimum_raw if suite_name == "bbob" else -optimum_raw
                        ),
                        objective=objective,
                    )
                )
            return problem_list, data, None

        if normalized_mode == "repair":
            dtype = float if suite_name == "bbob" else int
            return np.asarray(instances, dtype=dtype), None, None

        if normalized_mode == "evaluate":
            entry = problems
            dtype = float if suite_name == "bbob" else int
            decisions = np.asarray(instances, dtype=dtype)
            single = decisions.ndim == 1
            decisions = np.atleast_2d(decisions)
            raw = np.asarray(
                [float(entry.objective(decision)) for decision in decisions],
                dtype=float,
            )
            values = raw if suite_name == "bbob" else -raw
            if single:
                return float(values[0]), 0.0, None
            return values, np.zeros_like(values), None

        raise ValueError(f"Unsupported problem mode: {mode!r}")

    definition.__name__ = f"ioh_{suite_name}_f{fid}"
    definition.ioh_suite = suite_name
    definition.ioh_function_id = fid
    definition.__autoopt_checkpoint_semantics__ = {
        "suite": suite_name,
        "function_id": fid,
    }
    return definition


def make_bbob_problem(function_id: int):
    """Create an AutoOptLib adapter for BBOB f1--f24."""

    return make_ioh_problem("bbob", function_id)


def make_pbo_problem(function_id: int):
    """Create an AutoOptLib adapter for the protocol's PBO F1--F23."""

    return make_ioh_problem("pbo", function_id)


__all__ = [
    "IOHInstance",
    "IOH_OPTIMUM_METADATA_REVISION",
    "known_ioh_optimum_raw",
    "make_bbob_problem",
    "make_ioh_problem",
    "make_pbo_problem",
]
