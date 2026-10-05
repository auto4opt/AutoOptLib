"""Resource-aware scheduling primitives for heterogeneous CPU tasks."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, Union

from .resources import (
    NumaNode,
    available_memory_bytes,
    discover_cpu_topology,
    discover_numa_nodes,
)


@dataclass(frozen=True)
class TaskResources:
    """Resources and locality requested by one black-box evaluation task.

    ``None`` means that the scheduler should use its default or a learned
    profile. ``memory_bytes`` is incremental working memory, excluding the
    worker's already-resident problem environment.
    """

    cores: int | None = None
    memory_bytes: int | None = None
    io_bound: bool | None = None
    numa_node: int | None = None
    resident_key: str | None = None
    profile_key: str | None = None
    priority: int = 0

    def validate(self) -> "TaskResources":
        if self.cores is not None and (
            not isinstance(self.cores, int)
            or isinstance(self.cores, bool)
            or self.cores <= 0
        ):
            raise ValueError("TaskResources.cores must be positive or None.")
        if self.memory_bytes is not None and (
            not isinstance(self.memory_bytes, int)
            or isinstance(self.memory_bytes, bool)
            or self.memory_bytes < 0
        ):
            raise ValueError("TaskResources.memory_bytes must be non-negative or None.")
        if self.io_bound is not None and not isinstance(self.io_bound, bool):
            raise ValueError("TaskResources.io_bound must be boolean or None.")
        if self.numa_node is not None and not isinstance(self.numa_node, int):
            raise ValueError("TaskResources.numa_node must be an integer or None.")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise ValueError("TaskResources.priority must be an integer.")
        return self

    @classmethod
    def coerce(cls, value: Any) -> "TaskResources":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value.validate()
        if isinstance(value, dict):
            return cls(**value).validate()
        raise TypeError("Task resources must be TaskResources, a mapping, or None.")

    def overlay(self, value: Any) -> "TaskResources":
        """Overlay non-None estimator fields on explicitly supplied metadata."""

        other = self.coerce(value)
        updates = {
            name: getattr(other, name)
            for name in (
                "cores",
                "memory_bytes",
                "io_bound",
                "numa_node",
                "resident_key",
                "profile_key",
            )
            if getattr(other, name) is not None
        }
        if other.priority:
            updates["priority"] = other.priority
        return replace(self, **updates).validate()


@dataclass(frozen=True)
class ScheduledTask:
    """A payload carrying scheduler-visible resource and locality metadata."""

    payload: Any
    resources: TaskResources = TaskResources()
    key: str | None = None
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResourceAllocation:
    """Concrete resources assigned to a task attempt."""

    cpu_ids: tuple[int, ...]
    numa_node: int | None
    native_threads: int
    memory_bytes: int
    cpu_weight: float
    concurrent_tasks: int = 1


@dataclass(frozen=True)
class TaskObservation:
    """Measurements returned by a worker after one task attempt."""

    wall_seconds: float
    cpu_seconds: float
    rss_delta_bytes: int
    cpu_ids: tuple[int, ...]
    numa_node: int | None
    concurrent_tasks: int = 1

    @property
    def cpu_ratio(self) -> float:
        if self.wall_seconds <= 0:
            return 1.0
        cores = max(1, len(self.cpu_ids))
        return max(0.0, min(1.0, self.cpu_seconds / (self.wall_seconds * cores)))


@dataclass
class ResourceProfile:
    samples: int = 0
    wall_seconds: float = 0.0
    cpu_ratio: float = 1.0
    memory_bytes: float = 0.0
    baseline_wall_seconds: float = 0.0
    contention_limit: int | None = None
    core_wall_seconds: dict[int, float] = field(default_factory=dict)
    core_samples: dict[int, int] = field(default_factory=dict)
    concurrency_wall_seconds: dict[int, float] = field(default_factory=dict)
    concurrency_samples: dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "samples": self.samples,
            "wall_seconds": self.wall_seconds,
            "cpu_ratio": self.cpu_ratio,
            "memory_bytes": self.memory_bytes,
            "baseline_wall_seconds": self.baseline_wall_seconds,
            "contention_limit": self.contention_limit,
            "recommended_cores": self.best_cores(),
        }

    def update(self, observation: TaskObservation) -> None:
        alpha = 1.0 if self.samples == 0 else 0.25
        self.wall_seconds = (
            alpha * observation.wall_seconds + (1.0 - alpha) * self.wall_seconds
        )
        self.cpu_ratio = alpha * observation.cpu_ratio + (1.0 - alpha) * self.cpu_ratio
        self.memory_bytes = (
            alpha * observation.rss_delta_bytes + (1.0 - alpha) * self.memory_bytes
        )
        cores = max(1, len(observation.cpu_ids))
        concurrency = max(1, int(observation.concurrent_tasks))
        concurrency_count = self.concurrency_samples.get(concurrency, 0)
        concurrency_previous = self.concurrency_wall_seconds.get(
            concurrency, observation.wall_seconds
        )
        self.concurrency_wall_seconds[concurrency] = (
            concurrency_previous * concurrency_count + observation.wall_seconds
        ) / (concurrency_count + 1)
        self.concurrency_samples[concurrency] = concurrency_count + 1
        if observation.concurrent_tasks <= 1:
            count = self.core_samples.get(cores, 0)
            previous = self.core_wall_seconds.get(cores, observation.wall_seconds)
            self.core_wall_seconds[cores] = (
                previous * count + observation.wall_seconds
            ) / (count + 1)
            self.core_samples[cores] = count + 1
        if observation.concurrent_tasks <= 1:
            self.baseline_wall_seconds = self.concurrency_wall_seconds[1]
        elif (
            self.baseline_wall_seconds > 0
            and self.concurrency_samples.get(1, 0) >= 2
            and self.concurrency_samples.get(concurrency, 0) >= 2
            and self.concurrency_wall_seconds[concurrency]
            > self.baseline_wall_seconds * 1.5
            and observation.cpu_ratio >= 0.7
        ):
            # CPU-active tasks getting substantially slower as concurrency
            # rises are commonly contending for memory bandwidth, cache, or a
            # native-library resource. Cap this profile below the observed
            # saturation point on future waves.
            learned_limit = max(1, observation.concurrent_tasks - 1)
            self.contention_limit = (
                learned_limit
                if self.contention_limit is None
                else min(self.contention_limit, learned_limit)
            )
        self.samples += 1

    def best_cores(self) -> int | None:
        if not self.core_wall_seconds:
            return None
        return min(
            self.core_wall_seconds,
            key=lambda cores: (self.core_wall_seconds[cores], cores),
        )

    def choose_cores(self, default: int, maximum: int) -> int:
        """Explore legal core counts once, then use the fastest observation."""

        maximum = max(1, maximum)
        if not self.core_wall_seconds:
            return min(maximum, max(1, default))
        candidates = []
        value = 1
        while value < maximum:
            candidates.append(value)
            value *= 2
        candidates.append(maximum)
        for candidate in candidates:
            if candidate not in self.core_wall_seconds:
                return candidate
        return int(self.best_cores() or default)


@dataclass(frozen=True)
class ResourceSnapshot:
    total_cpu_ids: tuple[int, ...]
    logical_cpus: int
    physical_cores: int
    numa_nodes: int
    available_memory_bytes: int
    memory_budget_bytes: int
    reserved_memory_bytes: int
    running_tasks: int
    profiles: int
    resident_keys: int
    resident_replicas: int


def process_rss_bytes() -> int:
    """Best-effort current resident set size without a mandatory dependency."""

    statm = Path("/proc/self/statm")
    try:
        fields = statm.read_text(encoding="utf-8").split()
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, IndexError, ValueError):
        pass
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:  # pragma: no cover - Windows fallback
        return 0
    # Linux reports KiB; macOS and BSD report bytes.
    return int(usage * 1024 if os.name == "posix" and Path("/proc").exists() else usage)


def process_peak_rss_bytes() -> int:
    """Best-effort process peak RSS for transient allocation observations."""

    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:  # pragma: no cover - Windows fallback
        return process_rss_bytes()
    return int(usage * 1024 if Path("/proc").exists() else usage)


def process_memory_snapshot() -> tuple[int, int]:
    """Return current and peak RSS for retained and transient task memory."""

    peak = process_peak_rss_bytes()
    statm = Path("/proc/self/statm")
    try:
        fields = statm.read_text(encoding="utf-8").split()
        current = int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, IndexError, ValueError):
        current = peak
    return current, peak


class ResourceAwareScheduler:
    """Work-conserving scheduler with CPU, memory, NUMA, and locality control."""

    def __init__(
        self,
        *,
        worker_nodes: dict[int, int | None],
        default_cores: int,
        max_cores_per_task: int,
        memory_limit_bytes: int | None,
        max_oversubscription: float,
        adaptive: bool,
        io_cpu_threshold: float,
    ) -> None:
        self.nodes = discover_numa_nodes()
        self.worker_nodes = dict(worker_nodes)
        self.default_cores = max(1, int(default_cores))
        self.max_cores_per_task = max(self.default_cores, int(max_cores_per_task))
        self.max_oversubscription = max(1.0, float(max_oversubscription))
        self.adaptive = bool(adaptive)
        self.io_cpu_threshold = float(io_cpu_threshold)
        detected_memory = max(1, available_memory_bytes())
        self._available_memory = detected_memory
        self._memory_checked_at = time.monotonic()
        self.memory_budget = int(
            memory_limit_bytes
            if memory_limit_bytes is not None
            else max(1, detected_memory * 0.8)
        )
        self._cpu_load = {cpu: 0.0 for node in self.nodes for cpu in node.cpu_ids}
        self._allocations: dict[int, tuple[ResourceAllocation, TaskResources]] = {}
        self._reserved_memory = 0
        self._profiles: dict[str, ResourceProfile] = {}
        self._worker_residents: dict[int, set[str]] = {
            worker_id: set() for worker_id in worker_nodes
        }
        # A worker process constructs/unpickles its session after NUMA binding,
        # so every node with an eligible worker can host a local copy. A
        # one-to-one key->node map incorrectly stranded all other sockets.
        self._resident_nodes: dict[str, set[int]] = {}

    @property
    def profiles(self) -> dict[str, ResourceProfile]:
        return self._profiles

    def profile_snapshot(self) -> dict[str, dict[str, int | float | None]]:
        return {key: profile.as_dict() for key, profile in self._profiles.items()}

    def snapshot(self) -> ResourceSnapshot:
        topology = discover_cpu_topology()
        return ResourceSnapshot(
            total_cpu_ids=tuple(self._cpu_load),
            logical_cpus=len(topology),
            physical_cores=len({item.physical_core for item in topology}),
            numa_nodes=len(self.nodes),
            available_memory_bytes=available_memory_bytes(),
            memory_budget_bytes=self.memory_budget,
            reserved_memory_bytes=self._reserved_memory,
            running_tasks=len(self._allocations),
            profiles=len(self._profiles),
            resident_keys=len(self._resident_nodes),
            resident_replicas=sum(
                len(nodes) for nodes in self._resident_nodes.values()
            ),
        )

    def eligible_worker_count(self, resources: Iterable[TaskResources]) -> int:
        """Return workers that can serve at least one task before dynamic limits."""

        values = tuple(resources)
        if not values:
            return 0
        eligible = 0
        for node_id in self.worker_nodes.values():
            node = self._node(node_id)
            for item in values:
                requested = item.cores or self.default_cores
                if item.numa_node is not None and item.numa_node != node_id:
                    continue
                if requested <= len(node.cpu_ids):
                    eligible += 1
                    break
        return eligible

    def running_by_numa(self) -> dict[int | None, int]:
        counts: dict[int | None, int] = {}
        for allocation, _resources in self._allocations.values():
            counts[allocation.numa_node] = counts.get(allocation.numa_node, 0) + 1
        return counts

    def resident_replica_snapshot(self) -> dict[str, tuple[int, ...]]:
        return {
            key: tuple(sorted(nodes)) for key, nodes in self._resident_nodes.items()
        }

    def _node(self, node_id: int | None) -> NumaNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return self.nodes[0]

    def _resolved(
        self, resources: TaskResources, transfer_bytes: int
    ) -> tuple[int, int, bool, float, ResourceProfile | None]:
        profile = (
            self._profiles.get(resources.profile_key)
            if resources.profile_key is not None
            else None
        )
        if resources.cores is not None:
            requested_cores = resources.cores
        elif self.adaptive and profile is not None:
            requested_cores = profile.choose_cores(
                self.default_cores, self.max_cores_per_task
            )
        else:
            requested_cores = self.default_cores
        cores = min(max(1, requested_cores), len(self._cpu_load))
        learned_memory = (
            int(math.ceil(profile.memory_bytes))
            if self.adaptive and profile is not None
            else 0
        )
        memory = max(resources.memory_bytes or 0, learned_memory, transfer_bytes)
        if resources.io_bound is None:
            io_bound = bool(
                self.adaptive
                and profile is not None
                and profile.samples > 0
                and profile.cpu_ratio < self.io_cpu_threshold
            )
        else:
            io_bound = resources.io_bound
        cpu_weight = (
            max(0.1, profile.cpu_ratio if profile else 0.25) if io_bound else 1.0
        )
        return cores, memory, io_bound, cpu_weight, profile

    def _memory_fits(self, requested: int) -> bool:
        if self._reserved_memory + requested > self.memory_budget:
            return False
        # Respect changing system pressure as well as the configured budget.
        now = time.monotonic()
        if now - self._memory_checked_at >= 0.1:
            self._available_memory = available_memory_bytes()
            self._memory_checked_at = now
        system_available = self._available_memory
        safety = min(self.memory_budget // 20, 512 * 1024 * 1024)
        return requested <= max(0, system_available - safety)

    def _candidate_allocation(
        self,
        worker_id: int,
        resources: TaskResources,
        transfer_bytes: int,
    ) -> tuple[ResourceAllocation, float] | None:
        home_node = self.worker_nodes.get(worker_id)
        if resources.numa_node is not None and resources.numa_node != home_node:
            return None
        node = self._node(home_node)
        cores, memory, io_bound, weight, profile = self._resolved(
            resources, transfer_bytes
        )
        if (
            self.adaptive
            and profile is not None
            and profile.contention_limit is not None
        ):
            same_profile = sum(
                running.profile_key == resources.profile_key
                for _allocation, running in self._allocations.values()
            )
            if same_profile >= profile.contention_limit:
                return None
        if cores > len(node.cpu_ids):
            if resources.cores is not None:
                return None
            cores = len(node.cpu_ids)
        if not self._memory_fits(memory):
            return None
        ordered = sorted(node.cpu_ids, key=lambda cpu: (self._cpu_load[cpu], cpu))
        selected = tuple(ordered[: min(cores, len(ordered))])
        if not selected:
            return None
        limit = self.max_oversubscription if io_bound else 1.0
        if any(self._cpu_load[cpu] + weight > limit + 1e-12 for cpu in selected):
            return None
        locality = 0.0
        if resources.resident_key:
            if resources.resident_key in self._worker_residents[worker_id]:
                locality += 1000.0
            if home_node in self._resident_nodes.get(resources.resident_key, set()):
                locality += 100.0
        duration = (
            profile.wall_seconds if self.adaptive and profile is not None else 0.0
        )
        score = resources.priority * 10_000.0 + locality + duration
        return (
            ResourceAllocation(
                cpu_ids=selected,
                numa_node=home_node,
                native_threads=len(selected),
                memory_bytes=memory,
                cpu_weight=weight,
            ),
            score,
        )

    def select(
        self,
        worker_id: int,
        pending: Sequence[tuple[int, int]],
        resources: dict[int, TaskResources],
        transfer_bytes: dict[int, int],
    ) -> tuple[int, int, ResourceAllocation] | None:
        """Choose the best currently feasible task for one idle worker."""

        best: tuple[float, int, int, int, ResourceAllocation] | None = None
        for position, (task_id, attempt) in enumerate(pending):
            candidate = self._candidate_allocation(
                worker_id, resources[task_id], transfer_bytes[task_id]
            )
            if candidate is None:
                continue
            allocation, score = candidate
            value = (score, -position, task_id, attempt, allocation)
            if best is None or value[:2] > best[:2]:
                best = value
        if best is None:
            return None
        _, _, task_id, attempt, allocation = best
        allocation = replace(allocation, concurrent_tasks=len(self._allocations) + 1)
        self._allocations[worker_id] = (allocation, resources[task_id])
        resident_key = resources[task_id].resident_key
        if resident_key and allocation.numa_node is not None:
            # Record a local replica rather than globally pinning this key to
            # its first NUMA node. Worker sessions are initialized after their
            # CPU/memory policy is bound and therefore already provide the
            # concrete per-node copy.
            self._resident_nodes.setdefault(resident_key, set()).add(
                allocation.numa_node
            )
        self._reserved_memory += allocation.memory_bytes
        for cpu in allocation.cpu_ids:
            self._cpu_load[cpu] += allocation.cpu_weight
        return task_id, attempt, allocation

    def release(
        self, worker_id: int, observation: TaskObservation | None = None
    ) -> None:
        current = self._allocations.pop(worker_id, None)
        if current is None:
            return
        allocation, resources = current
        self._reserved_memory = max(0, self._reserved_memory - allocation.memory_bytes)
        for cpu in allocation.cpu_ids:
            self._cpu_load[cpu] = max(0.0, self._cpu_load[cpu] - allocation.cpu_weight)
        if resources.resident_key:
            self._worker_residents[worker_id].add(resources.resident_key)
            if allocation.numa_node is not None:
                self._resident_nodes.setdefault(resources.resident_key, set()).add(
                    allocation.numa_node
                )
        if observation is not None and resources.profile_key:
            self._profiles.setdefault(resources.profile_key, ResourceProfile()).update(
                observation
            )

    def impossible_memory_request(
        self,
        pending: Iterable[tuple[int, int]],
        resources: dict[int, TaskResources],
        transfer_bytes: dict[int, int],
    ) -> tuple[int, int] | None:
        """Return a task whose declared memory can never fit the hard budget."""

        for task_id, _attempt in pending:
            _cores, memory, _io, _weight, _profile = self._resolved(
                resources[task_id], transfer_bytes[task_id]
            )
            if memory > self.memory_budget:
                return task_id, memory
        return None


ResourceEstimator = Callable[[Any], Union[TaskResources, dict[str, Any], None]]


__all__ = [
    "ResourceAllocation",
    "ResourceAwareScheduler",
    "ResourceEstimator",
    "ResourceProfile",
    "ResourceSnapshot",
    "ScheduledTask",
    "TaskObservation",
    "TaskResources",
    "process_memory_snapshot",
    "process_peak_rss_bytes",
    "process_rss_bytes",
]
