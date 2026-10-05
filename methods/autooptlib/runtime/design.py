"""Candidate-level parallelism for automatically designed algorithms."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .config import EvaluationRuntimeConfig
from .executor import PersistentProcessExecutor, RuntimeStatistics
from .lifecycle import finalize_evaluation_session, initialize_evaluation_session
from .resources import discover_cpu_topology
from .scheduler import ScheduledTask, TaskResources
from .shared import SharedArray, SharedArraySpec


@dataclass
class _DesignSession:
    problems: Any
    data: Any
    custom_components: Any
    worker_initializer: Any = None
    worker_finalizer: Any = None
    worker_config: Any = None
    worker_environment: Any = None


@dataclass
class _DesignTask:
    design: Any
    setting: Any
    coordinates: tuple[tuple[int, int], ...]
    random_seeds: tuple[int, ...]
    performance: SharedArraySpec
    performance_approx: SharedArraySpec
    evaluations: SharedArraySpec
    stream_events: SharedArraySpec
    stream_event_budgets: SharedArraySpec
    termination_codes: SharedArraySpec
    row: int
    evaluation_row: int


def _evaluate_design_task(session: _DesignSession, task: _DesignTask) -> None:
    from ..components import _restore_custom_components

    _restore_custom_components(session.custom_components)
    # Process workers naturally receive an isolated serialized task. The
    # coordinator deep-copies only the serial path before calling this handler.
    design = task.design
    metadata_before = dict(getattr(design, "metadata", {}) or {})
    ledger_before = len(metadata_before.get("evaluation_ledger", []))
    for (seed, run), random_seed in zip(task.coordinates, task.random_seeds):
        setting = deepcopy(task.setting)
        setting.EvalWorkers = 1
        setting.EvalBackend = "serial"
        setting.rng = np.random.default_rng(random_seed)
        design.evaluate(
            session.problems,
            session.data,
            setting,
            [seed],
            run_indices=[run],
        )
    perf_memory, performance = task.performance.attach()
    approx_memory, performance_approx = task.performance_approx.attach()
    evaluation_memory, evaluations = task.evaluations.attach()
    stream_event_memory, stream_events = task.stream_events.attach()
    stream_budget_memory, stream_event_budgets = task.stream_event_budgets.attach()
    termination_memory, termination_codes = task.termination_codes.attach()
    try:
        for seed, run in task.coordinates:
            performance[task.row, seed, run] = float(design.performance[seed, run])
            performance_approx[task.row, seed, run] = float(
                design.performance_approx[seed, run]
            )
        metadata_after = dict(getattr(design, "metadata", {}) or {})
        new_entries = list(metadata_after.get("evaluation_ledger", []))[ledger_before:]
        actual = sum(int(entry.get("evaluations", 0)) for entry in new_entries)
        if not new_entries:
            actual = len(task.coordinates) * int(getattr(task.setting, "ProbFE", 0))
        evaluations[task.evaluation_row] = actual
        stream_events[task.evaluation_row] = sum(
            int(entry.get("stream_events", 0)) for entry in new_entries
        )
        stream_budgets = [
            entry.get("stream_event_budget")
            for entry in new_entries
            if entry.get("stream_event_budget") is not None
        ]
        stream_event_budgets[task.evaluation_row] = (
            int(stream_budgets[-1]) if stream_budgets else -1
        )
        termination_codes[task.evaluation_row] = int(
            any(
                entry.get("termination_reason") == "stream_event_budget_exhausted"
                for entry in new_entries
            )
        )
    finally:
        perf_memory.close()
        approx_memory.close()
        evaluation_memory.close()
        stream_event_memory.close()
        stream_budget_memory.close()
        termination_memory.close()


class DesignEvaluationRuntime:
    """Evaluate heterogeneous candidate algorithms on persistent CPU workers."""

    def __init__(self, problems: Any, data: Any, setting: Any) -> None:
        self.problems = problems
        self.data = data
        self._problem_signature = self._build_problem_signature(problems)
        self.config = EvaluationRuntimeConfig.from_setting(setting)
        self._executor: PersistentProcessExecutor | None = None
        topology = discover_cpu_topology()
        self.statistics = RuntimeStatistics(
            requested_workers=self.config.workers,
            workers=1,
            spawned_workers=0,
            eligible_workers=1,
            active_workers=1,
            selected_workers=1,
            logical_cpus=len(topology),
            physical_cores=len({item.physical_core for item in topology}),
            numa_nodes=len({item.node_id for item in topology}),
        )
        self.total_evaluations = 0
        self._evaluation_wave = 0
        configured_seed = getattr(
            setting,
            "EvalCommonRandomSeed",
            getattr(setting, "Seed", getattr(setting, "seed", None)),
        )
        if configured_seed is None:
            rng = getattr(setting, "rng", None)
            if not isinstance(rng, np.random.Generator):
                rng = np.random.default_rng()
            configured_seed = int(
                rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
            )
        self._common_random_seed = int(configured_seed)

    def _coordinate_random_seed(self, setting: Any, coordinate: tuple[int, int]) -> int:
        """Resolve an optional protocol seed before falling back to CRN derivation."""

        configured = getattr(setting, "EvalCoordinateSeeds", None)
        seed_index, run = coordinate
        explicit: Any = None
        if isinstance(configured, dict):
            for key in (coordinate, f"{seed_index}:{run}", seed_index, str(seed_index)):
                if key in configured:
                    explicit = configured[key]
                    break
        elif configured is not None and seed_index < len(configured):
            explicit = configured[seed_index]
        if explicit is not None:
            base = int(explicit)
            if run == 0:
                return base
            return int(
                np.random.SeedSequence([base, run]).generate_state(1, dtype=np.uint64)[
                    0
                ]
            )
        return int(
            np.random.SeedSequence(
                [self._common_random_seed, seed_index, run]
            ).generate_state(1, dtype=np.uint64)[0]
        )

    def set_common_random_seed(self, seed: int) -> None:
        """Set the CRN seed used by subsequent evaluation waves."""

        self._common_random_seed = int(seed)

    @staticmethod
    def _build_problem_signature(problems: Any) -> str:
        """Identify objective families without inspecting their implementation."""

        if problems is None:
            return "<unknown>"
        try:
            problem_values = list(problems)
        except TypeError:
            problem_values = [problems]
        values: list[tuple[str, int | None, tuple[str, ...]]] = []
        for problem in problem_values:
            name = getattr(problem, "name", type(problem).__qualname__)
            if callable(name):
                module = getattr(name, "__module__", type(name).__module__)
                qualified = getattr(name, "__qualname__", type(name).__qualname__)
                name = f"{module}.{qualified}"
            problem_type = getattr(problem, "type", ())
            if isinstance(problem_type, str):
                types = (problem_type,)
            else:
                types = tuple(str(value) for value in problem_type)
            dimension = getattr(problem, "dimension", None)
            values.append(
                (
                    str(name),
                    None if dimension is None else int(dimension),
                    types,
                )
            )
        return repr(tuple(sorted(set(values))))

    def evaluate(
        self,
        designs: Iterable[Any],
        setting: Any,
        seeds: Sequence[int],
    ) -> list[Any]:
        values = list(designs)
        if not values:
            return values
        seed_values = tuple(int(seed) for seed in seeds)
        if not seed_values:
            return values
        _, columns = np.asarray(values[0].performance).shape
        coordinates = tuple(
            (seed, run) for seed in seed_values for run in range(columns)
        )
        return self.evaluate_coordinates(values, setting, coordinates)

    def evaluate_coordinates(
        self,
        designs: Iterable[Any],
        setting: Any,
        coordinates: Sequence[tuple[int, int]],
    ) -> list[Any]:
        """Evaluate explicit instance/run keys for SMAC-style intensification."""

        values = list(designs)
        if not values:
            return values
        rows, columns = np.asarray(values[0].performance).shape
        coordinates = tuple((int(seed), int(run)) for seed, run in coordinates)
        if not coordinates:
            return values
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("Evaluation coordinates must be unique within a wave.")
        if any(
            seed < 0 or seed >= rows or run < 0 or run >= columns
            for seed, run in coordinates
        ):
            raise IndexError(
                "Evaluation coordinate lies outside the performance matrix."
            )
        potential_jobs = max(1, len(values) * len(coordinates))
        workers = self.config.resolved_workers(potential_jobs)
        backend = self.config.resolved_backend(potential_jobs)
        task_count = len(values) * len(coordinates)
        with (
            SharedArray((len(values), rows, columns)) as performance,
            SharedArray((len(values), rows, columns)) as performance_approx,
            SharedArray((task_count,), dtype=np.int64) as evaluations,
            SharedArray((task_count,), dtype=np.int64) as stream_events,
            SharedArray((task_count,), dtype=np.int64) as stream_event_budgets,
            SharedArray((task_count,), dtype=np.uint8) as termination_codes,
        ):
            stream_event_budgets.array.fill(-1)
            for index, design in enumerate(values):
                performance.array[index, :, :] = np.asarray(
                    design.performance, dtype=float
                )
                performance_approx.array[index, :, :] = np.asarray(
                    design.performance_approx, dtype=float
                )

            # Use reproducible per-instance/run streams, shared across candidates.
            # The results therefore do not depend on worker count or chunking.
            coordinate_randoms = {
                coordinate: self._coordinate_random_seed(setting, coordinate)
                for coordinate in coordinates
            }
            # Every independent coordinate remains visible to the scheduler.
            # Combining several coordinates into a serialized chunk hides cost
            # skew and prevents strict instance/NUMA locality decisions.
            chunks = [(coordinate,) for coordinate in coordinates]
            tasks: list[ScheduledTask] = []
            evaluation_row = 0
            task_coordinates: list[tuple[int, tuple[int, int], int]] = []
            for index, design in enumerate(values):
                for chunk in chunks:
                    instances = tuple(sorted({coordinate[0] for coordinate in chunk}))
                    resident_key = "problem:" + ",".join(map(str, instances))
                    tasks.append(
                        ScheduledTask(
                            _DesignTask(
                                design=design,
                                setting=setting,
                                coordinates=chunk,
                                random_seeds=tuple(
                                    coordinate_randoms[coordinate]
                                    for coordinate in chunk
                                ),
                                performance=performance.spec,
                                performance_approx=performance_approx.spec,
                                evaluations=evaluations.spec,
                                stream_events=stream_events.spec,
                                stream_event_budgets=stream_event_budgets.spec,
                                termination_codes=termination_codes.spec,
                                row=index,
                                evaluation_row=evaluation_row,
                            ),
                            TaskResources(
                                resident_key=resident_key,
                                profile_key=(
                                    f"design:{self._problem_signature}:{resident_key}"
                                ),
                            ),
                        )
                    )
                    for coordinate in chunk:
                        task_coordinates.append((index, coordinate, evaluation_row))
                    evaluation_row += 1
            if backend == "process" and workers > 1:
                if self._executor is None or self._executor.worker_count != workers:
                    self.close()
                    worker_initializer = getattr(setting, "EvalWorkerInitializer", None)
                    worker_finalizer = getattr(setting, "EvalWorkerFinalizer", None)
                    worker_config = getattr(setting, "EvalWorkerConfig", None)
                    process_config = EvaluationRuntimeConfig(
                        backend="process",
                        workers=(
                            self.config.workers
                            if isinstance(self.config.workers, str)
                            else workers
                        ),
                        cores_per_worker=self.config.cores_per_worker,
                        max_cores_per_task=(
                            self.config.max_cores_per_task
                            or self.config.cores_per_worker
                        ),
                        affinity=self.config.affinity,
                        task_timeout=self.config.task_timeout,
                        adaptive=self.config.adaptive,
                        memory_limit_bytes=self.config.memory_limit_bytes,
                        max_oversubscription=self.config.max_oversubscription,
                        io_cpu_threshold=self.config.io_cpu_threshold,
                        shared_memory_threshold=self.config.shared_memory_threshold,
                        resource_estimator=self.config.resource_estimator,
                        concurrency_profile_path=(self.config.concurrency_profile_path),
                    )
                    self._executor = PersistentProcessExecutor(
                        _evaluate_design_task,
                        _DesignSession(
                            self.problems if worker_initializer is None else None,
                            self.data if worker_initializer is None else None,
                            self._custom_components(),
                            worker_initializer=worker_initializer,
                            worker_finalizer=worker_finalizer,
                            worker_config=worker_config,
                        ),
                        config=process_config,
                        jobs_hint=len(tasks),
                        initializer=(
                            initialize_evaluation_session
                            if worker_initializer is not None
                            else None
                        ),
                        finalizer=(
                            finalize_evaluation_session
                            if worker_initializer is not None
                            else None
                        ),
                    )
                previous = self._executor.statistics.snapshot()
                try:
                    self._executor.map(
                        tasks,
                        timeout=self.config.task_timeout,
                        # EvalRetries applies to individual black-box calls inside
                        # run_design, not to a whole algorithm/instance task.
                        retries=0,
                    )
                finally:
                    self.statistics.merge(
                        self._executor.statistics.delta_from(previous)
                    )
            else:
                session = _DesignSession(
                    self.problems,
                    self.data,
                    self._custom_components(),
                )
                started = time.monotonic()
                task_seconds = 0.0
                for task in tasks:
                    task_started = time.monotonic()
                    _evaluate_design_task(session, deepcopy(task.payload))
                    task_seconds += time.monotonic() - task_started
                wall_seconds = time.monotonic() - started
                self.statistics.merge(
                    RuntimeStatistics(
                        requested_workers=self.config.workers,
                        workers=1,
                        spawned_workers=0,
                        eligible_workers=1,
                        active_workers=1,
                        selected_workers=1,
                        submitted=len(tasks),
                        completed=len(tasks),
                        task_seconds=task_seconds,
                        wall_seconds=wall_seconds,
                        capacity_seconds=wall_seconds,
                        waves=1,
                        peak_concurrency=1 if tasks else 0,
                        logical_cpus=self.statistics.logical_cpus,
                        physical_cores=self.statistics.physical_cores,
                        numa_nodes=self.statistics.numa_nodes,
                    )
                )
            self._evaluation_wave += 1
            actual_wave_evaluations = int(np.sum(evaluations.array, dtype=np.int64))
            self.total_evaluations += actual_wave_evaluations
            for index, design in enumerate(values):
                design.performance = np.array(performance.array[index], copy=True)
                design.performance_approx = np.array(
                    performance_approx.array[index], copy=True
                )
                metadata = dict(getattr(design, "metadata", {}) or {})
                ledger = list(metadata.get("evaluation_ledger", []))
                for design_index, coordinate, task_row in task_coordinates:
                    if design_index != index:
                        continue
                    seed, run = coordinate
                    entry = {
                        "instance_index": seed,
                        "run": run,
                        "seed": coordinate_randoms[coordinate],
                        "evaluation_wave": self._evaluation_wave,
                        "evaluations": int(evaluations.array[task_row]),
                        "value": float(design.performance[seed, run]),
                    }
                    if int(stream_event_budgets.array[task_row]) >= 0:
                        entry.update(
                            {
                                "stream_events": int(stream_events.array[task_row]),
                                "stream_event_budget": int(
                                    stream_event_budgets.array[task_row]
                                ),
                                "termination_reason": (
                                    "stream_event_budget_exhausted"
                                    if int(termination_codes.array[task_row]) == 1
                                    else "completed"
                                ),
                            }
                        )
                    ledger.append(entry)
                metadata["evaluation_ledger"] = ledger
                design.metadata = metadata
        setting.EvalActualDesignFEs = self.total_evaluations
        setting.EvalRuntimeStats = self.statistics.as_dict()
        setting.EvalRuntimeProfiles = (
            self._executor.resource_profiles if self._executor is not None else {}
        )
        setting.EvalConcurrencyProfiles = (
            self._executor.concurrency_profiles if self._executor is not None else {}
        )
        return values

    @staticmethod
    def _custom_components() -> Any:
        from ..components import _custom_component_snapshot

        return _custom_component_snapshot()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.close()
            self._executor = None

    def __enter__(self) -> "DesignEvaluationRuntime":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
