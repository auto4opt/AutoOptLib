"""Pure-Python evaluation bridge between ALDes and AutoOptLib."""

from __future__ import annotations

import atexit
import math
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np

from ..problems.base import validate_constructed_problems
from ..utils.general.process import _build_problem_struct
from ..utils.space import space
from .codec import decode_sequence
from .vocabulary import normalize_sequence

_PROCESS_POOLS: dict[int, ProcessPoolExecutor] = {}
_PROCESS_POOL_LOCK = threading.Lock()


@dataclass(frozen=True)
class EvaluationConfig:
    """Execution budget used to score ALDes-generated algorithms."""

    population_size: int = 50
    evaluations: int = 5_000
    runs: int = 5
    inner_evaluations: int = 200
    metric: str = "quality"
    improvement_rate: float = -math.inf
    archive: tuple[str, ...] = ()
    seed: int | None = None
    initial_populations: Any = None

    def __post_init__(self) -> None:
        if self.population_size <= 0:
            raise ValueError("population_size must be positive.")
        if self.evaluations < self.population_size:
            raise ValueError("evaluations must be at least population_size.")
        if self.runs <= 0:
            raise ValueError("runs must be positive.")


def _setting(config: EvaluationConfig) -> SimpleNamespace:
    return SimpleNamespace(
        Mode="design",
        AlgP=1,
        AlgQ=3,
        Archive=list(config.archive),
        IncRate=config.improvement_rate,
        ProbN=config.population_size,
        ProbFE=config.evaluations,
        InnerFE=config.inner_evaluations,
        AlgN=1,
        AlgFE=1,
        AlgRuns=config.runs,
        Metric=config.metric,
        Evaluate="exact",
        Compare="average",
        LSRange=0.3,
        rng=np.random.default_rng(config.seed),
        Seed=config.seed,
        InitialPopulations=config.initial_populations,
    )


class AutoOptEvaluator:
    """Evaluate ALDes sequences with AutoOptLib's Python execution engine."""

    def __init__(
        self,
        problem: Any,
        instances: Sequence[Any],
        *,
        config: EvaluationConfig | None = None,
    ) -> None:
        self.problem_descriptor = problem
        self.instances = list(instances)
        if not self.instances:
            raise ValueError("instances cannot be empty.")
        self.config = config or EvaluationConfig()
        self.setting = _setting(self.config)
        self.problems = _build_problem_struct(
            self.problem_descriptor, self.instances, self.setting
        )
        self.problems, self.data, _ = self.problem_descriptor(
            self.problems, self.instances, "construct"
        )
        validate_constructed_problems(self.problems, self.data)
        self.setting = space(self.problems, self.setting)

    def evaluate(
        self,
        sequence: Sequence[int] | np.ndarray,
        *,
        instance_indices: Sequence[int] | None = None,
    ) -> np.ndarray:
        indices = (
            list(range(len(self.instances)))
            if instance_indices is None
            else [int(index) for index in instance_indices]
        )
        if not indices:
            raise ValueError("instance_indices cannot be empty.")
        if min(indices) < 0 or max(indices) >= len(self.instances):
            raise IndexError("instance index is outside the configured instances.")
        algorithm = decode_sequence(sequence, self.problems, self.setting)
        algorithm.evaluate(
            self.problems, self.data, self.setting, seed_instance=indices
        )
        return np.asarray(algorithm.performance[indices, :], dtype=float).copy()

    def evaluate_many(
        self,
        sequences: Iterable[Sequence[int] | np.ndarray],
        *,
        instance_indices: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        # Candidate algorithms must see the same initial random stream. This
        # makes their comparison independent of enumeration order while still
        # advancing the evaluator between PPO batches.
        initial_state = deepcopy(self.setting.rng.bit_generator.state)
        performances = []
        for sequence in sequences:
            self.setting.rng.bit_generator.state = deepcopy(initial_state)
            performances.append(
                self.evaluate(sequence, instance_indices=instance_indices)
            )
        self.setting.rng.bit_generator.state = deepcopy(initial_state)
        self.setting.rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
        means = np.asarray([float(np.mean(values)) for values in performances])
        return means, performances


def _shutdown_process_pools() -> None:
    for executor in _PROCESS_POOLS.values():
        executor.shutdown(wait=False, cancel_futures=True)
    _PROCESS_POOLS.clear()


atexit.register(_shutdown_process_pools)


def _process_pool(workers: int) -> ProcessPoolExecutor:
    """Return one persistent spawn-based pool for CPU objective evaluation."""

    with _PROCESS_POOL_LOCK:
        executor = _PROCESS_POOLS.get(workers)
        if executor is None:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
            _PROCESS_POOLS[workers] = executor
        return executor


def _resolve_evaluation_workers(requested: int | None, jobs: int) -> int:
    """Resolve an explicit or environment-controlled CPU worker count."""

    if jobs <= 1:
        return 1
    value: str | int | None = requested
    if value is None:
        value = os.environ.get("ALDES_EVAL_WORKERS", "auto")
    if isinstance(value, str) and value.strip().lower() in {"", "auto"}:
        count = os.cpu_count() or 1
    else:
        try:
            count = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "ALDES_EVAL_WORKERS must be 'auto' or a positive integer."
            ) from exc
        if count <= 0:
            raise ValueError(
                "ALDES_EVAL_WORKERS must be 'auto' or a positive integer."
            )
    return max(1, min(count, jobs))


def _evaluate_pbo_sequence(
    problem_id: int,
    sequence: tuple[int, ...],
    instances: tuple[int, ...],
    config: EvaluationConfig,
) -> tuple[float, np.ndarray]:
    """Evaluate one candidate in an isolated CPU worker process."""

    from .problems import make_pbo_problem

    evaluator = AutoOptEvaluator(
        make_pbo_problem(problem_id), list(instances), config=config
    )
    performance = evaluator.evaluate(sequence)
    return float(np.mean(performance)), performance


def _evaluate_pbo_sequences(
    actions: np.ndarray,
    problem_id: int,
    instances: Sequence[int],
    config: EvaluationConfig,
    *,
    workers: int | None,
) -> tuple[list[float], list[np.ndarray]]:
    """Evaluate unique candidates serially or with a persistent CPU pool."""

    canonical = [tuple(normalize_sequence(row)) for row in actions]
    unique = list(dict.fromkeys(canonical))
    worker_count = _resolve_evaluation_workers(workers, len(unique))
    results: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}
    instance_tuple = tuple(int(instance) for instance in instances)

    if worker_count == 1:
        for sequence in unique:
            results[sequence] = _evaluate_pbo_sequence(
                int(problem_id), sequence, instance_tuple, config
            )
    else:
        executor = _process_pool(worker_count)
        pending = {
            executor.submit(
                _evaluate_pbo_sequence,
                int(problem_id),
                sequence,
                instance_tuple,
                config,
            ): sequence
            for sequence in unique
        }
        for future in as_completed(pending):
            results[pending[future]] = future.result()

    means = [results[sequence][0] for sequence in canonical]
    performances = [np.array(results[sequence][1], copy=True) for sequence in canonical]
    return means, performances


def evaluate_pbo_actions(
    actions: Any,
    problem_id: int,
    *,
    evaluate_test: bool = False,
    seed: int | None = None,
    initial_populations: Any = None,
    workers: int | None = 1,
) -> tuple[list[float], list[np.ndarray]]:
    """Compatibility replacement for ALDes's MATLAB ``get_performance``.

    Training uses PBO instances 1--3 with five runs.  Test evaluation uses
    instance 4 with the paper's 50,000-FE/30-run protocol.
    """

    if hasattr(actions, "detach"):
        actions = actions.detach().cpu().numpy()
    array = np.asarray(actions)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if evaluate_test:
        instances = [4]
        config = EvaluationConfig(
            evaluations=50_000,
            runs=30,
            seed=seed,
            initial_populations=initial_populations,
        )
    else:
        instances = [1, 2, 3]
        config = EvaluationConfig(
            evaluations=5_000,
            runs=5,
            seed=seed,
            initial_populations=initial_populations,
        )
    return _evaluate_pbo_sequences(
        array,
        int(problem_id),
        instances,
        config,
        workers=workers,
    )


__all__ = ["AutoOptEvaluator", "EvaluationConfig", "evaluate_pbo_actions"]
