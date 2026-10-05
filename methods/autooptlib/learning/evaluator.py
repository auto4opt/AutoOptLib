"""Evaluation bridge for AutoOptLib graph-construction sequences."""

from __future__ import annotations

import math
import os
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np

from problems.base import validate_constructed_problems

from ..runtime import DesignEvaluationRuntime
from ..runtime.config import EvaluationRuntimeConfig
from ..utils.design._population import population_size_space
from ..utils.design._stream_graph import STREAM_GRAPH_SEMANTICS
from ..utils.general.process import _build_problem_struct
from ..utils.space import space
from .codec import LearningCodec
from .vocabulary import LearningVocabulary


@dataclass(frozen=True)
class EvaluationConfig:
    """Execution budget used to score learning-generated algorithms."""

    population_size: int = 50
    evaluations: int = 5_000
    runs: int = 5
    inner_evaluations: int = 200
    metric: str = "quality"
    improvement_rate: float = -math.inf
    archive: tuple[str, ...] = ()
    seed: int | None = None
    coordinate_seeds: tuple[int, ...] | None = None
    initial_populations: Any = None
    graph_semantics: str = "legacy_pathway_v1"

    def __post_init__(self) -> None:
        for name in ("population_size", "evaluations", "runs", "inner_evaluations"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.evaluations < self.population_size:
            raise ValueError("evaluations must be at least population_size.")
        if not isinstance(self.metric, str) or not self.metric.strip():
            raise ValueError("metric cannot be empty.")
        improvement_rate = float(self.improvement_rate)
        if not math.isfinite(improvement_rate) and improvement_rate != -math.inf:
            raise ValueError("improvement_rate must be finite or negative infinity.")
        if not isinstance(self.archive, tuple) or any(
            not isinstance(value, str) or not value for value in self.archive
        ):
            raise ValueError("archive must be a tuple of non-empty strings.")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer or None.")
        if self.coordinate_seeds is not None and any(
            type(value) is not int for value in self.coordinate_seeds
        ):
            raise ValueError("coordinate_seeds must contain integers.")
        if self.graph_semantics not in {"legacy_pathway_v1", STREAM_GRAPH_SEMANTICS}:
            raise ValueError("Unsupported evaluation graph semantics.")


def _setting(
    config: EvaluationConfig, pathways: int, search_components: int
) -> SimpleNamespace:
    return SimpleNamespace(
        Mode="design",
        AlgP=int(pathways),
        AlgQ=int(search_components),
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
        rng=np.random.default_rng(config.seed),
        Seed=config.seed,
        EvalCoordinateSeeds=config.coordinate_seeds,
        InitialPopulations=config.initial_populations,
        GraphSemantics=config.graph_semantics,
    )


class AutoOptEvaluator:
    """Evaluate learning sequences with AutoOptLib's shared execution engine."""

    def __init__(
        self,
        problem: Any,
        instances: Sequence[Any],
        *,
        config: EvaluationConfig | None = None,
        pathways: int = 1,
        search_components: int = 4,
        vocabulary: LearningVocabulary | None = None,
        workers: int | str = 1,
        worker_initializer: Any = None,
        worker_finalizer: Any = None,
        worker_config: Any = None,
        cores_per_worker: int = 1,
        max_cores_per_task: int | None = None,
        affinity: bool = True,
        adaptive: bool = True,
        memory_limit_bytes: int | None = None,
        max_oversubscription: float = 2.0,
        io_cpu_threshold: float = 0.35,
        shared_memory_threshold: int = 1_048_576,
        resource_estimator: Any = None,
        concurrency_profile_path: str | os.PathLike[str] | None = None,
        deduplicate: bool = True,
        training_scorer: Any = None,
    ) -> None:
        for name, value in (
            ("pathways", pathways),
            ("search_components", search_components),
            ("cores_per_worker", cores_per_worker),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if max_cores_per_task is not None and (
            type(max_cores_per_task) is not int or max_cores_per_task <= 0
        ):
            raise ValueError("max_cores_per_task must be a positive integer or None.")
        for name, value in (
            ("affinity", affinity),
            ("adaptive", adaptive),
            ("deduplicate", deduplicate),
        ):
            if type(value) is not bool:
                raise ValueError(f"{name} must be boolean.")
        self.problem_descriptor = problem
        self.instances = list(instances)
        if not self.instances:
            raise ValueError("instances cannot be empty.")
        self.config = config or EvaluationConfig()
        self.workers = self._normalize_workers(workers)
        self.deduplicate = deduplicate
        if training_scorer is not None and not callable(training_scorer):
            raise TypeError("training_scorer must be callable or None.")
        self.training_scorer = training_scorer
        if worker_initializer is not None and not callable(worker_initializer):
            raise TypeError("worker_initializer must be callable or None.")
        if worker_finalizer is not None and not callable(worker_finalizer):
            raise TypeError("worker_finalizer must be callable or None.")
        if worker_finalizer is not None and worker_initializer is None:
            raise ValueError("worker_finalizer requires worker_initializer.")
        self.setting = _setting(self.config, pathways, search_components)
        self.setting.EvalWorkers = self.workers
        self.setting.EvalBackend = (
            "process" if self.workers == "auto" or int(self.workers) > 1 else "serial"
        )
        self.setting.EvalCoresPerWorker = cores_per_worker
        self.setting.EvalMaxCoresPerTask = (
            cores_per_worker if max_cores_per_task is None else max_cores_per_task
        )
        self.setting.EvalAffinity = affinity
        self.setting.EvalTaskTimeoutSec = None
        self.setting.EvalAdaptive = adaptive
        self.setting.EvalMemoryLimitBytes = memory_limit_bytes
        self.setting.EvalMaxOversubscription = max_oversubscription
        self.setting.EvalIOCPUThreshold = io_cpu_threshold
        self.setting.EvalSharedMemoryThreshold = shared_memory_threshold
        self.setting.EvalResourceEstimator = resource_estimator
        self.setting.EvalConcurrencyProfilePath = concurrency_profile_path
        self.setting.EvalWorkerInitializer = worker_initializer
        self.setting.EvalWorkerFinalizer = worker_finalizer
        self.setting.EvalWorkerConfig = worker_config
        EvaluationRuntimeConfig.from_setting(self.setting)
        self.problems = _build_problem_struct(
            self.problem_descriptor, self.instances, self.setting
        )
        self.problems, self.data, _ = self.problem_descriptor(
            self.problems, self.instances, "construct"
        )
        validate_constructed_problems(self.problems, self.data)
        task_budgets = [
            int(budget)
            for item in self.data
            if (budget := getattr(getattr(item, "task", None), "budget", None))
            is not None
        ]
        if task_budgets:
            minimum_budget = min(task_budgets)
            values = [
                int(value)
                for value in population_size_space(self.setting)
                if int(value) <= minimum_budget
            ]
            if not values:
                raise ValueError(
                    "PopulationSizeSpace has no value within the smallest task budget."
                )
            self.setting.PopulationSizeSpace = values
        self.setting = space(self.problems, self.setting)
        self.codec = LearningCodec.from_problem(
            self.problems,
            self.setting,
            vocabulary=vocabulary,
        )
        self._runtime: DesignEvaluationRuntime | None = None
        self._runtime_workers: int | str = 0
        self._last_designs: dict[tuple[int, ...], Any] = {}
        self._restored_evaluations = 0
        self._restored_statistics: dict[str, Any] | None = None

    def evaluate(
        self,
        sequence: Sequence[int] | np.ndarray,
        *,
        instance_indices: Sequence[int] | None = None,
    ) -> np.ndarray:
        indices = self._instance_indices(instance_indices)
        _, performances = self.evaluate_many(
            [sequence], instance_indices=indices, workers=self.workers
        )
        return performances[0]

    def evaluate_many(
        self,
        sequences: Iterable[Sequence[int] | np.ndarray],
        *,
        instance_indices: Sequence[int] | None = None,
        workers: int | str | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        initial_state = deepcopy(self.setting.rng.bit_generator.state)
        values = [
            (
                self.codec.canonicalize(sequence)
                if self.codec.grammar.stream_graph
                else self.codec.grammar.normalize(sequence)
            )
            for sequence in sequences
        ]
        if not values:
            raise ValueError("sequences cannot be empty.")
        indices = self._instance_indices(instance_indices)
        worker_count = (
            self.workers if workers is None else self._normalize_workers(workers)
        )
        if self.deduplicate:
            groups: dict[tuple[int, ...], list[int]] = {}
            for index, sequence in enumerate(values):
                groups.setdefault(tuple(sequence), []).append(index)
            keys = list(groups)
            algorithms = [self.codec.decode(key) for key in keys]
        else:
            keys = [tuple(sequence) for sequence in values]
            groups = {key: [] for key in keys}
            algorithms = [self.codec.decode(key) for key in keys]
        if self._runtime is None or self._runtime_workers != worker_count:
            self.close()
            self.setting.EvalWorkers = worker_count
            self.setting.EvalBackend = (
                "process"
                if worker_count == "auto" or int(worker_count) > 1
                else "serial"
            )
            self._runtime = DesignEvaluationRuntime(
                self.problems, self.data, self.setting
            )
            self._runtime.total_evaluations = int(self._restored_evaluations)
            if self._restored_statistics:
                from ..runtime.executor import RuntimeStatistics

                self._runtime.statistics = RuntimeStatistics.from_dict(
                    self._restored_statistics
                )
            self._runtime_workers = worker_count
        self.setting.rng.bit_generator.state = deepcopy(initial_state)
        assert self._runtime is not None
        self._runtime.evaluate(algorithms, self.setting, indices)
        unique_performance = [
            np.asarray(algorithm.performance[indices, :], dtype=float).copy()
            for algorithm in algorithms
        ]
        if self.deduplicate:
            performances: list[np.ndarray] = [np.empty(0)] * len(values)
            for key, performance in zip(keys, unique_performance):
                for index in groups[key]:
                    performances[index] = performance.copy()
        else:
            performances = unique_performance
        self.setting.rng.bit_generator.state = deepcopy(initial_state)
        self.setting.rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
        self._last_designs = {
            key: deepcopy(algorithm) for key, algorithm in zip(keys, algorithms)
        }
        means = np.asarray(
            [
                (
                    float(self.training_scorer(performance, indices))
                    if self.training_scorer is not None
                    else float(np.mean(performance))
                )
                for performance in performances
            ]
        )
        return means, performances

    def restore_runtime_state(
        self, total_evaluations: int, statistics: dict[str, Any] | None = None
    ) -> None:
        """Restore cumulative accounting before the first resumed evaluation."""

        if self._runtime is not None:
            raise RuntimeError("Restore runtime state before evaluating candidates.")
        if type(total_evaluations) is not int or total_evaluations < 0:
            raise ValueError("total_evaluations must be a nonnegative integer.")
        if statistics is not None and not isinstance(statistics, dict):
            raise TypeError("statistics must be a dictionary or None.")
        self._restored_evaluations = total_evaluations
        self._restored_statistics = None if statistics is None else dict(statistics)

    @staticmethod
    def _normalize_workers(value: int | str) -> int | str:
        if isinstance(value, str):
            if value.lower() != "auto":
                raise ValueError("workers must be 'auto' or a positive integer.")
            return "auto"
        if type(value) is not int or value <= 0:
            raise ValueError("workers must be 'auto' or a positive integer.")
        return value

    def _instance_indices(self, values: Sequence[int] | None) -> list[int]:
        if values is None:
            return list(range(len(self.instances)))
        indices = list(values)
        if not indices:
            raise ValueError("instance_indices cannot be empty.")
        if any(type(index) is not int for index in indices):
            raise ValueError("instance_indices must contain integers.")
        if len(indices) != len(set(indices)):
            raise ValueError("instance_indices cannot contain duplicates.")
        if min(indices) < 0 or max(indices) >= len(self.instances):
            raise IndexError("instance index is outside the configured instances.")
        return indices

    def last_designs(
        self, sequences: Iterable[Sequence[int] | np.ndarray]
    ) -> list[Any]:
        """Return evaluated designs from the immediately preceding batch."""

        result = []
        for sequence in sequences:
            key = tuple(
                self.codec.canonicalize(sequence)
                if self.codec.grammar.stream_graph
                else self.codec.grammar.normalize(sequence)
            )
            if key not in self._last_designs:
                raise KeyError(
                    "Sequence was not part of the preceding evaluation batch."
                )
            result.append(deepcopy(self._last_designs[key]))
        return result

    def close(self) -> None:
        if self._runtime is not None:
            self._restored_evaluations = int(self._runtime.total_evaluations)
            self._restored_statistics = self._runtime.statistics.as_dict()
            self._runtime.close()
            self._runtime = None
            self._runtime_workers = 0

    @property
    def runtime_statistics(self) -> dict[str, Any] | None:
        """Cumulative statistics for the currently configured CPU runtime."""

        if self._runtime is None:
            return self._restored_statistics
        return self._runtime.statistics.as_dict()

    @property
    def actual_design_fes(self) -> int:
        """Cumulative objective evaluations performed by this evaluator."""

        return (
            int(self._restored_evaluations)
            if self._runtime is None
            else int(self._runtime.total_evaluations)
        )

    def __enter__(self) -> "AutoOptEvaluator":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["AutoOptEvaluator", "EvaluationConfig"]
