"""Persistent, failure-isolated CPU process executor."""

from __future__ import annotations

import math
import multiprocessing
import os
import pickle
import queue
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .concurrency import AdaptiveConcurrencyController
from .config import EvaluationRuntimeConfig
from .lifecycle import WorkerContext
from .resources import (
    bind_current_process,
    bind_numa_memory_policy,
    discover_cpu_topology,
    discover_numa_nodes,
    hardware_topology_signature,
    limit_native_threads,
    partition_cpu_ids,
)
from .scheduler import (
    ResourceAwareScheduler,
    ResourceEstimator,
    ResourceSnapshot,
    ScheduledTask,
    TaskObservation,
    TaskResources,
    process_memory_snapshot,
)
from .transport import SharedPickle, dumps_shared, loads_shared


def _serializer():
    try:
        import cloudpickle
    except ImportError:
        return pickle
    return cloudpickle


def _dumps(value: Any) -> bytes:
    return _serializer().dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _loads(value: bytes) -> Any:
    return _serializer().loads(value)


@dataclass
class RuntimeStatistics:
    requested_workers: int | str = 1
    workers: int = 1
    spawned_workers: int = 1
    eligible_workers: int = 1
    active_workers: int = 1
    selected_workers: int = 1
    worker_trials: int = 0
    submitted: int = 0
    completed: int = 0
    failed_attempts: int = 0
    retries: int = 0
    timeouts: int = 0
    worker_restarts: int = 0
    task_seconds: float = 0.0
    wall_seconds: float = 0.0
    capacity_seconds: float = 0.0
    waves: int = 0
    bytes_submitted: int = 0
    bytes_returned: int = 0
    shared_bytes_submitted: int = 0
    shared_bytes_returned: int = 0
    observed_cpu_seconds: float = 0.0
    allocated_core_seconds: float = 0.0
    peak_task_memory_bytes: int = 0
    peak_concurrency: int = 0
    peak_concurrency_by_numa: dict[str, int] = field(default_factory=dict)
    logical_cpus: int = 1
    physical_cores: int = 1
    numa_nodes: int = 1
    resident_replicas: int = 0
    scheduler_stalls: int = 0

    @property
    def parallel_efficiency(self) -> float:
        capacity = self.capacity_seconds
        if capacity <= 0 and self.wall_seconds > 0:
            capacity = self.wall_seconds * max(1, self.workers)
        if capacity <= 0:
            return 0.0
        return min(1.0, self.task_seconds / capacity)

    def snapshot(self) -> "RuntimeStatistics":
        return RuntimeStatistics(
            requested_workers=self.requested_workers,
            workers=self.workers,
            spawned_workers=self.spawned_workers,
            eligible_workers=self.eligible_workers,
            active_workers=self.active_workers,
            selected_workers=self.selected_workers,
            worker_trials=self.worker_trials,
            submitted=self.submitted,
            completed=self.completed,
            failed_attempts=self.failed_attempts,
            retries=self.retries,
            timeouts=self.timeouts,
            worker_restarts=self.worker_restarts,
            task_seconds=self.task_seconds,
            wall_seconds=self.wall_seconds,
            capacity_seconds=self.capacity_seconds,
            waves=self.waves,
            bytes_submitted=self.bytes_submitted,
            bytes_returned=self.bytes_returned,
            shared_bytes_submitted=self.shared_bytes_submitted,
            shared_bytes_returned=self.shared_bytes_returned,
            observed_cpu_seconds=self.observed_cpu_seconds,
            allocated_core_seconds=self.allocated_core_seconds,
            peak_task_memory_bytes=self.peak_task_memory_bytes,
            peak_concurrency=self.peak_concurrency,
            peak_concurrency_by_numa=dict(self.peak_concurrency_by_numa),
            logical_cpus=self.logical_cpus,
            physical_cores=self.physical_cores,
            numa_nodes=self.numa_nodes,
            resident_replicas=self.resident_replicas,
            scheduler_stalls=self.scheduler_stalls,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "RuntimeStatistics":
        """Restore persisted counters while ignoring derived presentation fields."""

        if not value:
            return cls()
        names = cls.__dataclass_fields__
        return cls(**{name: value[name] for name in names if name in value})

    def delta_from(self, previous: "RuntimeStatistics") -> "RuntimeStatistics":
        """Return counters accumulated since a previous snapshot."""

        return RuntimeStatistics(
            requested_workers=self.requested_workers,
            workers=self.workers,
            spawned_workers=self.spawned_workers,
            eligible_workers=self.eligible_workers,
            active_workers=self.active_workers,
            selected_workers=self.selected_workers,
            worker_trials=self.worker_trials - previous.worker_trials,
            submitted=self.submitted - previous.submitted,
            completed=self.completed - previous.completed,
            failed_attempts=self.failed_attempts - previous.failed_attempts,
            retries=self.retries - previous.retries,
            timeouts=self.timeouts - previous.timeouts,
            worker_restarts=self.worker_restarts - previous.worker_restarts,
            task_seconds=self.task_seconds - previous.task_seconds,
            wall_seconds=self.wall_seconds - previous.wall_seconds,
            capacity_seconds=self.capacity_seconds - previous.capacity_seconds,
            waves=self.waves - previous.waves,
            bytes_submitted=self.bytes_submitted - previous.bytes_submitted,
            bytes_returned=self.bytes_returned - previous.bytes_returned,
            shared_bytes_submitted=(
                self.shared_bytes_submitted - previous.shared_bytes_submitted
            ),
            shared_bytes_returned=(
                self.shared_bytes_returned - previous.shared_bytes_returned
            ),
            observed_cpu_seconds=(
                self.observed_cpu_seconds - previous.observed_cpu_seconds
            ),
            allocated_core_seconds=(
                self.allocated_core_seconds - previous.allocated_core_seconds
            ),
            peak_task_memory_bytes=self.peak_task_memory_bytes,
            peak_concurrency=self.peak_concurrency,
            peak_concurrency_by_numa=dict(self.peak_concurrency_by_numa),
            logical_cpus=self.logical_cpus,
            physical_cores=self.physical_cores,
            numa_nodes=self.numa_nodes,
            resident_replicas=self.resident_replicas,
            scheduler_stalls=self.scheduler_stalls - previous.scheduler_stalls,
        )

    def merge(self, other: "RuntimeStatistics") -> None:
        """Accumulate a serial wave or one process-pool delta."""

        self.workers = max(self.workers, other.workers)
        self.requested_workers = other.requested_workers
        self.spawned_workers = max(self.spawned_workers, other.spawned_workers)
        self.eligible_workers = max(self.eligible_workers, other.eligible_workers)
        self.active_workers = other.active_workers
        self.selected_workers = other.selected_workers
        for name in (
            "worker_trials",
            "submitted",
            "completed",
            "failed_attempts",
            "retries",
            "timeouts",
            "worker_restarts",
            "task_seconds",
            "wall_seconds",
            "capacity_seconds",
            "waves",
            "bytes_submitted",
            "bytes_returned",
            "shared_bytes_submitted",
            "shared_bytes_returned",
            "observed_cpu_seconds",
            "allocated_core_seconds",
            "scheduler_stalls",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.peak_task_memory_bytes = max(
            self.peak_task_memory_bytes, other.peak_task_memory_bytes
        )
        self.peak_concurrency = max(self.peak_concurrency, other.peak_concurrency)
        for node, value in other.peak_concurrency_by_numa.items():
            self.peak_concurrency_by_numa[node] = max(
                self.peak_concurrency_by_numa.get(node, 0), value
            )
        self.logical_cpus = max(self.logical_cpus, other.logical_cpus)
        self.physical_cores = max(self.physical_cores, other.physical_cores)
        self.numa_nodes = max(self.numa_nodes, other.numa_nodes)
        self.resident_replicas = max(self.resident_replicas, other.resident_replicas)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable cumulative runtime counters."""

        return {
            "requested_workers": self.requested_workers,
            "workers": self.workers,
            "spawned_workers": self.spawned_workers,
            "eligible_workers": self.eligible_workers,
            "active_workers": self.active_workers,
            "selected_workers": self.selected_workers,
            "worker_trials": self.worker_trials,
            "submitted": self.submitted,
            "completed": self.completed,
            "failed_attempts": self.failed_attempts,
            "retries": self.retries,
            "timeouts": self.timeouts,
            "worker_restarts": self.worker_restarts,
            "task_seconds": self.task_seconds,
            "wall_seconds": self.wall_seconds,
            "capacity_seconds": self.capacity_seconds,
            "waves": self.waves,
            "parallel_efficiency": self.parallel_efficiency,
            "bytes_submitted": self.bytes_submitted,
            "bytes_returned": self.bytes_returned,
            "shared_bytes_submitted": self.shared_bytes_submitted,
            "shared_bytes_returned": self.shared_bytes_returned,
            "observed_cpu_seconds": self.observed_cpu_seconds,
            "allocated_core_seconds": self.allocated_core_seconds,
            "peak_task_memory_bytes": self.peak_task_memory_bytes,
            "peak_concurrency": self.peak_concurrency,
            "peak_concurrency_by_numa": dict(self.peak_concurrency_by_numa),
            "logical_cpus": self.logical_cpus,
            "physical_cores": self.physical_cores,
            "smt_threads_per_core": self.logical_cpus / max(1, self.physical_cores),
            "numa_nodes": self.numa_nodes,
            "resident_replicas": self.resident_replicas,
            "scheduler_stalls": self.scheduler_stalls,
        }


class EvaluationTaskError(RuntimeError):
    """Raised when a process evaluation task exhausts its retry policy."""


def _worker_main(
    worker_id: int,
    generation: int,
    input_queue: Any,
    output_queue: Any,
    handler_blob: bytes,
    session_blob: bytes,
    initializer_blob: bytes | None,
    finalizer_blob: bytes | None,
    cpu_ids: tuple[int, ...],
    numa_node: int | None,
    native_threads: int,
    affinity: bool,
    shared_memory_threshold: int,
) -> None:
    if affinity and cpu_ids:
        bind_current_process(cpu_ids)
    bind_numa_memory_policy(numa_node)
    with limit_native_threads(native_threads):
        context = WorkerContext(
            worker_id=worker_id,
            generation=generation,
            pid=os.getpid(),
            cpu_ids=cpu_ids,
            native_threads=native_threads,
            numa_node=numa_node,
        )
        initialized = False
        finalizer = None
        try:
            handler = _loads(handler_blob)
            session = _loads(session_blob)
            initializer = (
                _loads(initializer_blob) if initializer_blob is not None else None
            )
            finalizer = _loads(finalizer_blob) if finalizer_blob is not None else None
            if initializer is not None:
                session = initializer(session, context)
            initialized = True
        except BaseException:
            output_queue.put(
                (
                    "bootstrap_error",
                    worker_id,
                    generation,
                    os.getpid(),
                    traceback.format_exc(),
                )
            )
            return
        try:
            output_queue.put(("ready", worker_id, generation, os.getpid()))
            while True:
                message = input_queue.get()
                if message is None:
                    return
                task_id, attempt, payload_blob, allocation = message
                started = time.monotonic()
                output_queue.put(
                    ("started", worker_id, generation, task_id, attempt, started)
                )
                attached_memories = []
                result_payload = None
                task_started = started
                cpu_started: float | None = None
                cpu_finished: float | None = None
                rss_started, peak_rss_started = process_memory_snapshot()
                try:
                    if affinity and allocation.cpu_ids:
                        bind_current_process(allocation.cpu_ids)
                    payload, attached_memories = loads_shared(payload_blob)
                    # Resource learning describes the black-box call itself.
                    # Deserialization and memory probes are runtime overhead and
                    # are accounted by wall/capacity statistics instead.
                    task_started = time.monotonic()
                    cpu_started = time.process_time()
                    if allocation.native_threads == native_threads:
                        value = handler(session, payload)
                    else:
                        with limit_native_threads(allocation.native_threads):
                            value = handler(session, payload)
                    cpu_finished = time.process_time()
                    elapsed = time.monotonic() - task_started
                    rss_finished, peak_rss_finished = process_memory_snapshot()
                    observation = TaskObservation(
                        wall_seconds=elapsed,
                        cpu_seconds=max(0.0, cpu_finished - cpu_started),
                        rss_delta_bytes=max(
                            0,
                            rss_finished - rss_started,
                            peak_rss_finished - peak_rss_started,
                        ),
                        cpu_ids=allocation.cpu_ids,
                        numa_node=allocation.numa_node,
                        concurrent_tasks=allocation.concurrent_tasks,
                    )
                    result_payload = dumps_shared(value, shared_memory_threshold)
                    output_queue.put(
                        (
                            "result",
                            worker_id,
                            generation,
                            task_id,
                            attempt,
                            observation,
                            result_payload.blob,
                            result_payload.shared_bytes,
                        )
                    )
                except BaseException as exc:
                    elapsed = time.monotonic() - task_started
                    cpu_finished = time.process_time()
                    cpu_seconds = (
                        0.0
                        if cpu_started is None
                        else max(0.0, cpu_finished - cpu_started)
                    )
                    rss_finished, peak_rss_finished = process_memory_snapshot()
                    observation = TaskObservation(
                        wall_seconds=elapsed,
                        cpu_seconds=cpu_seconds,
                        rss_delta_bytes=max(
                            0,
                            rss_finished - rss_started,
                            peak_rss_finished - peak_rss_started,
                        ),
                        cpu_ids=allocation.cpu_ids,
                        numa_node=allocation.numa_node,
                        concurrent_tasks=allocation.concurrent_tasks,
                    )
                    output_queue.put(
                        (
                            "error",
                            worker_id,
                            generation,
                            task_id,
                            attempt,
                            observation,
                            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                        )
                    )
                finally:
                    for memory in attached_memories:
                        memory.close()
                    if result_payload is not None:
                        # Ownership transfers to the coordinator, which copies
                        # the public result and unlinks the segments.
                        result_payload.close(unlink=False)
        finally:
            if initialized and finalizer is not None:
                try:
                    finalizer(session, context)
                except BaseException:
                    output_queue.put(
                        (
                            "finalizer_error",
                            worker_id,
                            generation,
                            os.getpid(),
                            traceback.format_exc(),
                        )
                    )


@dataclass
class _Worker:
    process: Any
    input_queue: Any
    generation: int
    task_id: int | None = None
    attempt: int = 0
    started: float | None = None


class PersistentProcessExecutor:
    """Run heterogeneous tasks on persistent, individually restartable workers."""

    def __init__(
        self,
        handler: Callable[[Any, Any], Any],
        session: Any,
        *,
        config: EvaluationRuntimeConfig,
        jobs_hint: int | None = None,
        initializer: Callable[[Any, WorkerContext], Any] | None = None,
        finalizer: Callable[[Any, WorkerContext], None] | None = None,
        resource_estimator: ResourceEstimator | None = None,
    ) -> None:
        self.config = config.validate()
        self.worker_count = config.resolved_workers(jobs_hint)
        if self.worker_count <= 1:
            raise ValueError("PersistentProcessExecutor requires at least two workers.")
        self._handler_blob = _dumps(handler)
        self._session_blob = _dumps(session)
        self._initializer_blob = (
            _dumps(initializer) if initializer is not None else None
        )
        self._finalizer_blob = _dumps(finalizer) if finalizer is not None else None
        try:
            self._context = multiprocessing.get_context(config.start_method())
        except ValueError as exc:
            raise ValueError(
                f"Unsupported multiprocessing start method {config.start_method()!r}."
            ) from exc
        self._output_queue = self._context.Queue()
        self._cpu_groups = partition_cpu_ids(self.worker_count, config.cores_per_worker)
        topology = discover_numa_nodes()
        cpu_to_node = {cpu: node.node_id for node in topology for cpu in node.cpu_ids}
        self._worker_nodes = {
            worker_id: cpu_to_node.get(self._cpu_groups[worker_id][0])
            for worker_id in range(self.worker_count)
        }
        self._scheduler = ResourceAwareScheduler(
            worker_nodes=self._worker_nodes,
            default_cores=config.cores_per_worker,
            max_cores_per_task=(config.max_cores_per_task or config.cores_per_worker),
            memory_limit_bytes=config.memory_limit_bytes,
            max_oversubscription=config.max_oversubscription,
            adaptive=config.adaptive,
            io_cpu_threshold=config.io_cpu_threshold,
        )
        self._resource_estimator = resource_estimator or config.resource_estimator
        self._concurrency = (
            AdaptiveConcurrencyController(
                self.worker_count,
                preferred_workers=max(
                    1,
                    len({item.physical_core for item in discover_cpu_topology()})
                    // self.config.cores_per_worker,
                ),
                profile_path=config.concurrency_profile_path,
                hardware_signature=hardware_topology_signature(),
            )
            if config.adaptive
            and isinstance(config.workers, str)
            and config.workers.lower() == "auto"
            else None
        )
        self._workers: dict[int, _Worker] = {}
        self._message_backlog: list[Any] = []
        self._task_payloads: dict[int, SharedPickle] = {}
        self._task_resources: dict[int, TaskResources] = {}
        self._task_transfer_bytes: dict[int, int] = {}
        self._task_sequence = 0
        topology = discover_cpu_topology()
        self.statistics = RuntimeStatistics(
            requested_workers=config.workers,
            workers=self.worker_count,
            spawned_workers=0,
            eligible_workers=0,
            active_workers=0,
            selected_workers=self.worker_count,
            logical_cpus=len(topology),
            physical_cores=len({item.physical_core for item in topology}),
            numa_nodes=len({item.node_id for item in topology}),
        )
        self._closed = False
        try:
            initial_workers = 1 if self._concurrency is not None else self.worker_count
            for worker_id in range(initial_workers):
                self._spawn_worker(worker_id, generation=0)
            self._await_ready()
        except BaseException:
            self.close()
            raise

    def _spawn_worker(self, worker_id: int, generation: int) -> None:
        input_queue = self._context.Queue(maxsize=1)
        process = self._context.Process(
            target=_worker_main,
            args=(
                worker_id,
                generation,
                input_queue,
                self._output_queue,
                self._handler_blob,
                self._session_blob,
                self._initializer_blob,
                self._finalizer_blob,
                self._cpu_groups[worker_id],
                self._worker_nodes[worker_id],
                self.config.cores_per_worker,
                self.config.affinity,
                self.config.shared_memory_threshold,
            ),
            name=f"autoopt-eval-{worker_id}",
        )
        process.start()
        self._workers[worker_id] = _Worker(
            process=process, input_queue=input_queue, generation=generation
        )
        self.statistics.spawned_workers = max(
            self.statistics.spawned_workers, len(self._workers)
        )

    def _ensure_worker_count(self, count: int) -> None:
        """Lazily grow an automatic pool so pilot startup cost is measurable."""

        target = max(1, min(self.worker_count, int(count)))
        created: set[int] = set()
        for worker_id in range(target):
            if worker_id in self._workers:
                continue
            self._spawn_worker(worker_id, generation=0)
            created.add(worker_id)
        if created:
            self._await_ready(created)

    def _await_ready(self, worker_ids: set[int] | None = None) -> None:
        remaining = set(self._workers) if worker_ids is None else set(worker_ids)
        # Large shared allocations can start many process pools at once; give
        # workers enough time to import their runtime before declaring failure.
        deadline = time.monotonic() + 120.0
        while remaining:
            timeout = max(0.01, deadline - time.monotonic())
            if timeout <= 0:
                raise EvaluationTaskError(
                    "CPU evaluation workers did not start in time."
                )
            try:
                message = self._output_queue.get(timeout=timeout)
            except queue.Empty as exc:
                raise EvaluationTaskError(
                    "CPU evaluation workers did not start in time."
                ) from exc
            kind, worker_id, generation, *_rest = message
            worker = self._workers.get(worker_id)
            if worker is None or worker.generation != generation:
                continue
            if kind == "bootstrap_error":
                raise EvaluationTaskError(
                    "Could not initialize CPU evaluation worker:\n" + str(_rest[-1])
                )
            if kind == "ready":
                remaining.discard(worker_id)
            else:
                # A restart can overlap completions from other workers. Keep
                # those messages for map(); dropping one would leave its task
                # permanently marked as in flight.
                self._message_backlog.append(message)

    def _restart_worker(self, worker_id: int) -> None:
        worker = self._workers[worker_id]
        self._scheduler.release(worker_id)
        if worker.process.is_alive():
            worker.process.terminate()
            worker.process.join(timeout=1.0)
            if worker.process.is_alive() and hasattr(worker.process, "kill"):
                worker.process.kill()
                worker.process.join(timeout=1.0)
        try:
            worker.input_queue.close()
        except Exception:
            pass
        generation = worker.generation + 1
        self.statistics.worker_restarts += 1
        self._spawn_worker(worker_id, generation)
        # Only the replacement emits a new ready message. Waiting for every
        # existing worker here would deadlock after the first timeout/restart.
        self._await_ready({worker_id})

    @property
    def resource_snapshot(self) -> ResourceSnapshot:
        """Current CPU/memory reservations and learned-profile counts."""

        return self._scheduler.snapshot()

    @property
    def resource_profiles(self) -> dict[str, dict[str, int | float | None]]:
        """Learned duration, CPU, memory, and contention profiles by task key."""

        return self._scheduler.profile_snapshot()

    @property
    def concurrency_profiles(self) -> dict[str, dict[str, object]]:
        """Measured worker-count trials and selected limits by workload."""

        if self._concurrency is None:
            return {}
        return self._concurrency.profile_snapshot()

    def map(
        self,
        tasks: Iterable[Any],
        *,
        timeout: float | None = None,
        retries: int = 0,
    ) -> list[Any]:
        if self._closed:
            raise RuntimeError("CPU evaluation executor is closed.")
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ValueError("retries must be a non-negative integer.")
        if timeout is not None and (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
        ):
            raise ValueError("timeout must be positive and finite or None.")
        values = list(tasks)
        if not values:
            return []
        started_wall = time.monotonic()
        task_ids: list[int] = []
        task_keys: dict[str, int] = {}
        dependency_names: dict[int, tuple[str, ...]] = {}
        task_dependencies: dict[int, set[int]] = {}
        try:
            for value in values:
                if isinstance(value, ScheduledTask):
                    payload_value = value.payload
                    resources = TaskResources.coerce(value.resources)
                    key = value.key
                    depends_on = tuple(value.depends_on)
                else:
                    payload_value = value
                    resources = TaskResources()
                    key = None
                    depends_on = ()
                if key is not None:
                    if not isinstance(key, str) or not key:
                        raise ValueError(
                            "ScheduledTask.key must be a non-empty string."
                        )
                    if key in task_keys:
                        raise ValueError(f"Duplicate scheduled task key {key!r}.")
                if any(not isinstance(name, str) or not name for name in depends_on):
                    raise ValueError(
                        "ScheduledTask.depends_on must contain non-empty strings."
                    )
                if self._resource_estimator is not None:
                    resources = resources.overlay(
                        self._resource_estimator(payload_value)
                    )
                payload = dumps_shared(
                    payload_value, self.config.shared_memory_threshold
                )
                task_id = self._task_sequence
                self._task_sequence += 1
                self._task_payloads[task_id] = payload
                self._task_resources[task_id] = resources
                self._task_transfer_bytes[task_id] = payload.transfer_bytes
                task_ids.append(task_id)
                dependency_names[task_id] = depends_on
                if key is not None:
                    task_keys[key] = task_id
            for task_id, names in dependency_names.items():
                missing = [name for name in names if name not in task_keys]
                if missing:
                    raise ValueError(
                        f"Unknown scheduled task dependencies: {missing!r}."
                    )
                task_dependencies[task_id] = {task_keys[name] for name in names}
        except BaseException:
            for task_id in task_ids:
                self._task_payloads.pop(task_id).close(unlink=True)
                self._task_resources.pop(task_id, None)
                self._task_transfer_bytes.pop(task_id, None)
            raise
        self.statistics.submitted += len(task_ids)
        self.statistics.bytes_submitted += sum(
            len(self._task_payloads[task_id].blob) for task_id in task_ids
        )
        self.statistics.shared_bytes_submitted += sum(
            self._task_payloads[task_id].shared_bytes for task_id in task_ids
        )

        pending = [(task_id, 0) for task_id in task_ids]
        results: dict[int, Any] = {}
        errors: dict[int, str] = {}
        effective_capacity = max(
            1,
            min(
                len(task_ids),
                self.worker_count,
                self._scheduler.eligible_worker_count(
                    self._task_resources[task_id] for task_id in task_ids
                ),
            ),
        )
        self.statistics.eligible_workers = max(
            self.statistics.eligible_workers, effective_capacity
        )
        self.statistics.active_workers = effective_capacity
        concurrency_started = time.monotonic()
        if self._concurrency is not None:
            profile_counts: dict[str, int] = {}
            for task_id in task_ids:
                profile_key = self._task_resources[task_id].profile_key or "<default>"
                profile_counts[profile_key] = profile_counts.get(profile_key, 0) + 1
            jobs_bucket = 1 << max(0, len(task_ids).bit_length() - 1)
            signature = (
                jobs_bucket,
                tuple(sorted(profile_counts.items())),
                any(task_dependencies[task_id] for task_id in task_ids),
            )
            self._concurrency.begin_map(
                signature,
                len(task_ids),
                max_workers=effective_capacity,
            )
            self.statistics.active_workers = self._concurrency.active_limit
            self._ensure_worker_count(self._concurrency.active_limit)
            self._concurrency.note_workers_ready()

        def eligible_pending() -> list[tuple[int, int]]:
            for item in list(pending):
                task_id, _attempt = item
                failed = task_dependencies[task_id] & errors.keys()
                if failed:
                    pending.remove(item)
                    errors[task_id] = (
                        "Dependency failed before this task could be scheduled."
                    )
            completed = results.keys()
            return [item for item in pending if task_dependencies[item[0]] <= completed]

        def assign() -> bool:
            progressed = False
            running_before = sum(
                worker.task_id is not None for worker in self._workers.values()
            )
            slots = (
                self._concurrency.dispatch_slots(running_before)
                if self._concurrency is not None
                else effective_capacity - running_before
            )
            if slots <= 0:
                return False
            assigned = 0
            for worker_id, worker in self._workers.items():
                if assigned >= slots:
                    break
                if worker.task_id is not None:
                    continue
                eligible = eligible_pending()
                if not eligible:
                    continue
                selected = self._scheduler.select(
                    worker_id,
                    eligible,
                    self._task_resources,
                    self._task_transfer_bytes,
                )
                if selected is None:
                    continue
                task_id, attempt, allocation = selected
                pending.remove((task_id, attempt))
                worker.task_id = task_id
                worker.attempt = attempt
                # The task timeout measures user evaluation time, not queueing
                # or deserialization before the worker acknowledges start.
                worker.started = None
                worker.input_queue.put(
                    (
                        task_id,
                        attempt,
                        self._task_payloads[task_id].blob,
                        allocation,
                    )
                )
                if self._concurrency is not None:
                    self._concurrency.note_dispatch()
                assigned += 1
                progressed = True
            running = sum(
                worker.task_id is not None for worker in self._workers.values()
            )
            self.statistics.peak_concurrency = max(
                self.statistics.peak_concurrency, running
            )
            for node, count in self._scheduler.running_by_numa().items():
                key = "unknown" if node is None else str(node)
                self.statistics.peak_concurrency_by_numa[key] = max(
                    self.statistics.peak_concurrency_by_numa.get(key, 0), count
                )
            return progressed

        assign()
        effective_timeout = timeout if timeout is not None else self.config.task_timeout
        try:
            while len(results) + len(errors) < len(task_ids):
                if self._message_backlog:
                    message = self._message_backlog.pop(0)
                else:
                    try:
                        message = self._output_queue.get(timeout=0.05)
                    except queue.Empty:
                        message = None
                if message is not None:
                    kind, worker_id, generation, *rest = message
                    worker = self._workers.get(worker_id)
                    if worker is None or worker.generation != generation:
                        continue
                    if kind == "started":
                        task_id, attempt, _timestamp = rest
                        if worker.task_id == task_id and worker.attempt == attempt:
                            # Start the watchdog when the acknowledgement reaches
                            # the coordinator. This avoids false timeouts if both
                            # messages waited briefly in the queue.
                            worker.started = time.monotonic()
                    elif kind in {"result", "error"}:
                        task_id, attempt, observation, *payload_parts = rest
                        if worker.task_id != task_id or worker.attempt != attempt:
                            continue
                        self.statistics.task_seconds += observation.wall_seconds
                        self.statistics.observed_cpu_seconds += observation.cpu_seconds
                        self.statistics.allocated_core_seconds += (
                            observation.wall_seconds * max(1, len(observation.cpu_ids))
                        )
                        self.statistics.peak_task_memory_bytes = max(
                            self.statistics.peak_task_memory_bytes,
                            observation.rss_delta_bytes,
                        )
                        self._scheduler.release(worker_id, observation)
                        worker.task_id = None
                        worker.started = None
                        if self._concurrency is not None:
                            self._concurrency.note_completion(observation)
                        if kind == "result":
                            result_blob, shared_bytes = payload_parts
                            self.statistics.bytes_returned += len(result_blob)
                            self.statistics.shared_bytes_returned += int(shared_bytes)
                            result, memories = loads_shared(
                                result_blob, copy_arrays=True, unlink=True
                            )
                            # copy_arrays closes segments as they are loaded;
                            # this list is intentionally empty.
                            for memory in memories:
                                memory.close()
                            results[task_id] = result
                            self.statistics.completed += 1
                        else:
                            self.statistics.failed_attempts += 1
                            error_payload = payload_parts[0]
                            if attempt < retries:
                                self.statistics.retries += 1
                                pending.append((task_id, attempt + 1))
                            else:
                                errors[task_id] = str(error_payload)

                now = time.monotonic()
                for worker_id, worker in list(self._workers.items()):
                    if worker.task_id is None:
                        if not worker.process.is_alive():
                            self._restart_worker(worker_id)
                        continue
                    timed_out = (
                        effective_timeout is not None
                        and worker.started is not None
                        and now - worker.started > float(effective_timeout)
                    )
                    died = not worker.process.is_alive()
                    if not timed_out and not died:
                        continue
                    task_id = worker.task_id
                    attempt = worker.attempt
                    if timed_out:
                        self.statistics.timeouts += 1
                        reason = f"Task timed out after {float(effective_timeout):g}s."
                    else:
                        reason = "Evaluation worker exited unexpectedly."
                    self.statistics.failed_attempts += 1
                    self._restart_worker(worker_id)
                    if self._concurrency is not None:
                        self._concurrency.note_completion(None)
                    if attempt < retries:
                        self.statistics.retries += 1
                        pending.append((task_id, attempt + 1))
                    else:
                        errors[task_id] = reason
                running_count = sum(
                    worker.task_id is not None for worker in self._workers.values()
                )
                if self._concurrency is not None:
                    changed = self._concurrency.maybe_advance(
                        running=running_count,
                        pending=len(pending),
                    )
                    if changed:
                        self.statistics.active_workers = self._concurrency.active_limit
                        self._ensure_worker_count(self._concurrency.active_limit)
                        self._concurrency.note_workers_ready()
                progressed = assign()
                running = any(
                    worker.task_id is not None for worker in self._workers.values()
                )
                if pending and not progressed and not running:
                    self.statistics.scheduler_stalls += 1
                if pending and not running and not progressed:
                    eligible = eligible_pending()
                    if pending and not eligible:
                        task_id, _attempt = pending.pop(0)
                        errors[task_id] = "Task dependency graph contains a cycle."
                        continue
                    impossible = self._scheduler.impossible_memory_request(
                        eligible,
                        self._task_resources,
                        self._task_transfer_bytes,
                    )
                    if impossible is not None:
                        task_id, requested = impossible
                        errors[task_id] = (
                            f"Task requires {requested} bytes, exceeding the "
                            f"runtime memory budget of "
                            f"{self._scheduler.memory_budget} bytes."
                        )
                        pending = [item for item in pending if item[0] != task_id]
                    else:
                        task_id, _attempt = pending.pop(0)
                        errors[task_id] = (
                            "No worker can satisfy the task's CPU/NUMA resource "
                            "requirements."
                        )
        finally:
            finished_wall = time.monotonic()
            elapsed_wall = finished_wall - started_wall
            if self._concurrency is not None:
                self._concurrency.finish_map()
                adaptive_wall = max(0.0, finished_wall - concurrency_started)
                pre_adaptive_wall = max(0.0, elapsed_wall - adaptive_wall)
                self.statistics.capacity_seconds += (
                    self._concurrency.map_capacity_seconds + pre_adaptive_wall
                )
                self.statistics.selected_workers = self._concurrency.selected_workers
                self.statistics.active_workers = min(
                    effective_capacity, self._concurrency.selected_workers
                )
                self.statistics.worker_trials += self._concurrency.map_trials
            else:
                self.statistics.capacity_seconds += elapsed_wall * effective_capacity
                self.statistics.active_workers = effective_capacity
            self.statistics.resident_replicas = max(
                self.statistics.resident_replicas,
                sum(
                    len(nodes)
                    for nodes in self._scheduler.resident_replica_snapshot().values()
                ),
            )
            self.statistics.wall_seconds += elapsed_wall
            self.statistics.waves += 1
            for task_id in task_ids:
                payload = self._task_payloads.pop(task_id, None)
                if payload is not None:
                    payload.close(unlink=True)
                self._task_resources.pop(task_id, None)
                self._task_transfer_bytes.pop(task_id, None)
        if errors:
            first = next(task_id for task_id in task_ids if task_id in errors)
            raise EvaluationTaskError(
                f"CPU evaluation task {task_ids.index(first)} failed: {errors[first]}"
            )
        return [results[task_id] for task_id in task_ids]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for worker in self._workers.values():
            if worker.process.is_alive():
                try:
                    worker.input_queue.put_nowait(None)
                except Exception:
                    worker.process.terminate()
        for worker in self._workers.values():
            worker.process.join(timeout=2.0)
            if worker.process.is_alive():
                worker.process.terminate()
                worker.process.join(timeout=1.0)
            try:
                worker.input_queue.close()
            except Exception:
                pass
        try:
            self._output_queue.close()
        except Exception:
            pass
        for worker_id in tuple(self._workers):
            self._scheduler.release(worker_id)
        for payload in self._task_payloads.values():
            payload.close(unlink=True)
        self._workers.clear()
        self._task_payloads.clear()
        self._task_resources.clear()
        self._task_transfer_bytes.clear()
        self._message_backlog.clear()

    def __enter__(self) -> "PersistentProcessExecutor":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
