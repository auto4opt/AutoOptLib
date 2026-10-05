"""Validated configuration for the CPU evaluation runtime."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from .resources import available_cpu_ids

_BACKENDS = {"auto", "serial", "process"}


@dataclass(frozen=True)
class EvaluationRuntimeConfig:
    """Execution controls shared by solve and design workflows."""

    backend: str = "auto"
    workers: int | str = 1
    cores_per_worker: int = 1
    max_cores_per_task: int | None = None
    affinity: bool = True
    task_timeout: float | None = None
    adaptive: bool = True
    memory_limit_bytes: int | None = None
    max_oversubscription: float = 2.0
    io_cpu_threshold: float = 0.35
    shared_memory_threshold: int = 1_048_576
    resource_estimator: Any = None
    concurrency_profile_path: str | os.PathLike[str] | None = None

    @classmethod
    def from_setting(cls, setting: Any) -> "EvaluationRuntimeConfig":
        cores_per_worker = int(getattr(setting, "EvalCoresPerWorker", 1))
        max_cores_value = getattr(setting, "EvalMaxCoresPerTask", None)
        return cls(
            backend=str(getattr(setting, "EvalBackend", "auto")).lower(),
            workers=getattr(setting, "EvalWorkers", 1),
            cores_per_worker=cores_per_worker,
            max_cores_per_task=(
                cores_per_worker if max_cores_value is None else int(max_cores_value)
            ),
            affinity=bool(getattr(setting, "EvalAffinity", True)),
            task_timeout=getattr(setting, "EvalTaskTimeoutSec", None),
            adaptive=bool(getattr(setting, "EvalAdaptive", True)),
            memory_limit_bytes=getattr(setting, "EvalMemoryLimitBytes", None),
            max_oversubscription=float(
                getattr(setting, "EvalMaxOversubscription", 2.0)
            ),
            io_cpu_threshold=float(getattr(setting, "EvalIOCPUThreshold", 0.35)),
            shared_memory_threshold=int(
                getattr(setting, "EvalSharedMemoryThreshold", 1_048_576)
            ),
            resource_estimator=getattr(setting, "EvalResourceEstimator", None),
            concurrency_profile_path=getattr(
                setting, "EvalConcurrencyProfilePath", None
            ),
        ).validate()

    def validate(self) -> "EvaluationRuntimeConfig":
        if self.backend not in _BACKENDS:
            raise ValueError("EvalBackend must be 'auto', 'serial', or 'process'.")
        if isinstance(self.workers, str):
            if self.workers.lower() != "auto":
                raise ValueError("EvalWorkers must be 'auto' or a positive integer.")
        elif (
            not isinstance(self.workers, Integral)
            or isinstance(self.workers, bool)
            or int(self.workers) <= 0
        ):
            raise ValueError("EvalWorkers must be 'auto' or a positive integer.")
        if (
            not isinstance(self.cores_per_worker, Integral)
            or isinstance(self.cores_per_worker, bool)
            or int(self.cores_per_worker) <= 0
        ):
            raise ValueError("EvalCoresPerWorker must be a positive integer.")
        if self.max_cores_per_task is not None and (
            not isinstance(self.max_cores_per_task, Integral)
            or isinstance(self.max_cores_per_task, bool)
            or int(self.max_cores_per_task) < int(self.cores_per_worker)
        ):
            raise ValueError("EvalMaxCoresPerTask must be at least EvalCoresPerWorker.")
        if not isinstance(self.affinity, bool):
            raise ValueError("EvalAffinity must be a boolean.")
        if not isinstance(self.adaptive, bool):
            raise ValueError("EvalAdaptive must be a boolean.")
        if self.task_timeout is not None and (
            not math.isfinite(float(self.task_timeout)) or float(self.task_timeout) <= 0
        ):
            raise ValueError("EvalTaskTimeoutSec must be positive and finite or None.")
        if self.memory_limit_bytes is not None and (
            not isinstance(self.memory_limit_bytes, Integral)
            or isinstance(self.memory_limit_bytes, bool)
            or int(self.memory_limit_bytes) <= 0
        ):
            raise ValueError("EvalMemoryLimitBytes must be positive or None.")
        if (
            not math.isfinite(float(self.max_oversubscription))
            or float(self.max_oversubscription) < 1
        ):
            raise ValueError("EvalMaxOversubscription must be finite and at least 1.")
        if (
            not math.isfinite(float(self.io_cpu_threshold))
            or not 0 < float(self.io_cpu_threshold) < 1
        ):
            raise ValueError("EvalIOCPUThreshold must be between 0 and 1.")
        if (
            not isinstance(self.shared_memory_threshold, Integral)
            or isinstance(self.shared_memory_threshold, bool)
            or int(self.shared_memory_threshold) < 0
        ):
            raise ValueError("EvalSharedMemoryThreshold must be non-negative.")
        if self.resource_estimator is not None and not callable(
            self.resource_estimator
        ):
            raise ValueError("EvalResourceEstimator must be callable or None.")
        if self.concurrency_profile_path is not None and not isinstance(
            self.concurrency_profile_path, (str, os.PathLike)
        ):
            raise ValueError("EvalConcurrencyProfilePath must be a path or None.")
        return self

    def resolved_workers(self, jobs: int | None = None) -> int:
        if self.backend == "serial":
            return 1
        available = max(1, len(available_cpu_ids()))
        capacity = max(1, available // self.cores_per_worker)
        process_capacity = max(1, int(math.ceil(capacity * self.max_oversubscription)))
        if isinstance(self.workers, str):
            # Automatic CPU tuning explores up to the physical process-visible
            # core capacity.  Oversubscription is still available through an
            # explicit worker count for known I/O-bound workloads, but is not a
            # safe default before the black box has been observed.
            workers = capacity
        else:
            workers = min(int(self.workers), process_capacity)
        if jobs is not None:
            workers = min(workers, max(1, int(jobs)))
        return max(1, workers)

    def resolved_backend(self, jobs: int | None = None) -> str:
        if self.backend != "auto":
            return self.backend
        return "process" if self.resolved_workers(jobs) > 1 else "serial"

    def start_method(self) -> str:
        requested = os.environ.get("AUTOOPTLIB_MP_START_METHOD")
        if requested:
            return requested
        # Spawn avoids inheriting active numerical/torch thread pools.  Users
        # can explicitly choose forkserver/fork through the environment when
        # their objective cannot be serialized on a POSIX system.
        return "spawn"
