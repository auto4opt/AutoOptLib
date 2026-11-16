"""CPU execution runtime for expensive black-box evaluation workloads."""

from .concurrency import (
    AdaptiveConcurrencyController,
    ConcurrencyProfile,
    ConcurrencyTrial,
)
from .config import EvaluationRuntimeConfig
from .design import DesignEvaluationRuntime
from .executor import (
    EvaluationTaskError,
    PersistentProcessExecutor,
    RuntimeStatistics,
)
from .lifecycle import EvaluationWorkerEnvironment, WorkerContext
from .scheduler import (
    ResourceAllocation,
    ResourceProfile,
    ResourceSnapshot,
    ScheduledTask,
    TaskObservation,
    TaskResources,
)
from .shared import SharedArray, SharedArraySpec

__all__ = [
    "EvaluationRuntimeConfig",
    "AdaptiveConcurrencyController",
    "ConcurrencyProfile",
    "ConcurrencyTrial",
    "DesignEvaluationRuntime",
    "EvaluationTaskError",
    "EvaluationWorkerEnvironment",
    "PersistentProcessExecutor",
    "ResourceAllocation",
    "ResourceProfile",
    "ResourceSnapshot",
    "RuntimeStatistics",
    "ScheduledTask",
    "SharedArray",
    "SharedArraySpec",
    "TaskObservation",
    "TaskResources",
    "WorkerContext",
]
