"""Worker-local lifecycle contracts for non-serializable black-box resources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkerContext:
    """Identity and CPU allocation supplied to a worker initializer."""

    worker_id: int
    generation: int
    pid: int
    cpu_ids: tuple[int, ...]
    native_threads: int
    numa_node: int | None = None


@dataclass
class EvaluationWorkerEnvironment:
    """Problem objects and opaque state constructed inside one worker."""

    problems: Any
    data: Any
    state: Any = None


def coerce_worker_environment(value: Any) -> EvaluationWorkerEnvironment:
    """Normalize convenient two/three-value initializer return forms."""

    if isinstance(value, EvaluationWorkerEnvironment):
        return value
    if isinstance(value, tuple):
        if len(value) == 2:
            return EvaluationWorkerEnvironment(value[0], value[1])
        if len(value) == 3:
            return EvaluationWorkerEnvironment(value[0], value[1], value[2])
    if hasattr(value, "problems") and hasattr(value, "data"):
        return EvaluationWorkerEnvironment(
            value.problems,
            value.data,
            getattr(value, "state", None),
        )
    raise TypeError(
        "EvalWorkerInitializer must return EvaluationWorkerEnvironment, "
        "(problems, data), or (problems, data, state)."
    )


def initialize_evaluation_session(session: Any, context: WorkerContext) -> Any:
    """Populate a design/solve session from a user worker initializer."""

    environment = coerce_worker_environment(
        session.worker_initializer(session.worker_config, context)
    )
    session.problems = environment.problems
    session.data = environment.data
    session.worker_environment = environment
    return session


def finalize_evaluation_session(session: Any, context: WorkerContext) -> None:
    """Invoke the optional user cleanup hook for a graceful worker exit."""

    if session.worker_finalizer is not None:
        session.worker_finalizer(session.worker_environment, context)


__all__ = ["EvaluationWorkerEnvironment", "WorkerContext"]
