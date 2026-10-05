"""Translation of MATLAB Utilities/General/Input.m."""

from __future__ import annotations

import math
import os
import re
import warnings
from copy import deepcopy
from numbers import Integral, Real
from types import SimpleNamespace
from typing import Any, Iterable, Sequence, cast

from ..design._population import (
    BOUNDARY_HANDLING_SPACE,
    offspring_size_space,
    population_size_space,
)

_PARAM_KEYS = {
    "GraphSemantics",
    "AlgP",
    "AlgQ",
    "Archive",
    "StructureMutationRate",
    "SearchArchiveSize",
    "SearchStagnation",
    "SearchRestartFraction",
    "SearchImprovementRate",
    "SearchParameterActionProbability",
    "SearchActionProbabilityGain",
    "SearchActionRewardEWMA",
    "ParameterCMAInitialSigma",
    "ParameterCMAOffspring",
    "StructureCandidatesPerAction",
    "ParameterCMABlockGenerations",
    "PostStructureCMAGenerations",
    "SearchMaxAttempts",
    "SearchInitialDesigns",
    "IncRate",
    "ProbN",
    "PopulationSizeSpace",
    "OffspringSizeSpace",
    "BoundaryHandling",
    "ProbFE",
    "InnerFE",
    "AlgN",
    "AlgFE",
    "DesignFEs",
    "AlgRuns",
    "Metric",
    "Compare",
    "Evaluate",
    "Tmax",
    "Thres",
    "RacingK",
    "Alpha",
    "StructureRacingMinFraction",
    "IntensificationMaxConfigCalls",
    "Surro",
    "AlgFile",
    "AlgName",
    "Seed",
    "EvalRetries",
    "EvalTimeoutSec",
    "EvalFailure",
    "EvalPenalty",
    "StreamEventBudgetMultiplier",
    "EvalCache",
    "EvalLog",
    "EvalWorkers",
    "EvalBackend",
    "EvalCommonRandomSeed",
    "EvalCoordinateSeeds",
    "EvalTrainingScorer",
    "EvalCoresPerWorker",
    "EvalMaxCoresPerTask",
    "EvalAffinity",
    "EvalTaskTimeoutSec",
    "EvalAdaptive",
    "EvalMemoryLimitBytes",
    "EvalMaxOversubscription",
    "EvalIOCPUThreshold",
    "EvalSharedMemoryThreshold",
    "EvalResourceEstimator",
    "EvalConcurrencyProfilePath",
    "EvalWorkerInitializer",
    "EvalWorkerFinalizer",
    "EvalWorkerConfig",
    "CheckpointDir",
    "CheckpointEvery",
    "CheckpointSignature",
    "CandidateLogPath",
    "Resume",
    "ResumeBudgetExtension",
    "ResumeBudgetExtensionSourceAlgFE",
    "ResumeBudgetExtensionSourceCheckpointSignature",
    "Designer",
    "LearningModel",
    "LearningInitialPopulations",
    "LearningDevice",
}

_DATA_KEYS = {"Mode", "Problem", "InstanceTrain", "InstanceTest", "InstanceSolve"}
_PUBLIC_KEYS = _DATA_KEYS | _PARAM_KEYS | {"OutputDir"}
_KEY_LOOKUP = {re.sub(r"[^a-z0-9]", "", key.lower()): key for key in _PUBLIC_KEYS}

_COMMON_DEFAULTS = {
    "GraphSemantics": "legacy_pathway_v1",
    "AlgP": 1,
    "AlgQ": 4,
    "Archive": [],
    "StructureMutationRate": 0.30,
    "SearchArchiveSize": 1,
    "SearchStagnation": 3,
    "SearchRestartFraction": 0.50,
    "SearchImprovementRate": 0.05,
    "SearchParameterActionProbability": 0.50,
    "SearchActionProbabilityGain": 0.30,
    "SearchActionRewardEWMA": 0.20,
    "ParameterCMAInitialSigma": 1.0,
    "ParameterCMAOffspring": 5,
    "StructureCandidatesPerAction": 3,
    "ParameterCMABlockGenerations": 3,
    "PostStructureCMAGenerations": 1,
    "SearchMaxAttempts": 50,
    "SearchInitialDesigns": [],
    "BoundaryHandling": "clip",
    "IncRate": 0.05,
    "Metric": "quality",
    "Compare": "average",
    "Evaluate": "exact",
    "Alpha": 0.05,
    "StructureRacingMinFraction": 0.50,
    "Seed": None,
    "EvalRetries": 0,
    "EvalTimeoutSec": None,
    "EvalFailure": "raise",
    "EvalPenalty": 1e30,
    "StreamEventBudgetMultiplier": None,
    "EvalCache": False,
    "EvalLog": None,
    "EvalWorkers": 1,
    "EvalBackend": "auto",
    "EvalCommonRandomSeed": None,
    "EvalCoordinateSeeds": None,
    "EvalTrainingScorer": None,
    "EvalCoresPerWorker": 1,
    "EvalMaxCoresPerTask": None,
    "EvalAffinity": True,
    "EvalTaskTimeoutSec": None,
    "EvalAdaptive": True,
    "EvalMemoryLimitBytes": None,
    "EvalMaxOversubscription": 2.0,
    "EvalIOCPUThreshold": 0.35,
    "EvalSharedMemoryThreshold": 1_048_576,
    "EvalResourceEstimator": None,
    "EvalConcurrencyProfilePath": None,
    "EvalWorkerInitializer": None,
    "EvalWorkerFinalizer": None,
    "EvalWorkerConfig": None,
    "CheckpointDir": None,
    "CheckpointEvery": 1,
    "CheckpointSignature": None,
    "CandidateLogPath": None,
    "Resume": False,
    "ResumeBudgetExtension": False,
    "ResumeBudgetExtensionSourceAlgFE": None,
    "ResumeBudgetExtensionSourceCheckpointSignature": None,
    "Designer": "search",
    "LearningModel": None,
    "LearningInitialPopulations": None,
    "LearningDevice": "auto",
    "IntensificationMaxConfigCalls": 3,
    "PopulationSizeSpace": list(range(4, 101)),
    "OffspringSizeSpace": list(range(1, 101)),
}

_DESIGN_DEFAULTS = {
    "ProbN": 20,
    "ProbFE": 5000,
    "InnerFE": 500,
    "AlgN": 10,
    "AlgFE": 5000,
    "AlgRuns": 5,
    "Tmax": None,
    "Thres": None,
}

_SOLVE_DEFAULTS = {
    "ProbN": 50,
    "ProbFE": 50000,
    "AlgRuns": 5,
    "AlgFile": "",
    "AlgName": "",
    "Tmax": None,
    "Thres": None,
}


def normalize_options(options: dict[str, Any]) -> dict[str, Any]:
    """Normalize public keyword spellings and reject unknown options.

    Public keys are case-insensitive and may use underscores, so ``archive``,
    ``Archive``, and ``ARCHIVE`` all resolve to ``Archive``.  Rejecting unknown
    keys prevents experiments from silently running with ignored settings.
    """
    normalized: dict[str, Any] = {}
    unknown: list[str] = []
    for key, value in options.items():
        if not isinstance(key, str):
            unknown.append(repr(key))
            continue
        token = re.sub(r"[^a-z0-9]", "", key.lower())
        canonical = _KEY_LOOKUP.get(token)
        if canonical is None:
            unknown.append(key)
            continue
        if canonical in normalized:
            raise TypeError(
                f"AutoOpt option {canonical!r} was supplied more than once through aliases."
            )
        normalized[canonical] = value
    if unknown:
        supported = ", ".join(sorted(_PUBLIC_KEYS))
        raise TypeError(
            f"Unknown AutoOpt option(s): {', '.join(unknown)}. Supported options: {supported}."
        )
    return normalized


def _to_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return (value,)


def _ensure_namespace(setting: Any) -> SimpleNamespace:
    if isinstance(setting, SimpleNamespace):
        return setting
    if isinstance(setting, dict):
        return SimpleNamespace(**setting)
    data = {
        name: getattr(setting, name)
        for name in dir(setting)
        if not name.startswith("__") and not callable(getattr(setting, name))
    }
    return SimpleNamespace(**data)


def _find_argument(arguments: Sequence[Any], name: str) -> tuple[bool, Any]:
    if not arguments:
        return False, None
    # Arguments are alternating key/value pairs. Searching the whole list
    # compares ``name`` against user values too; a NumPy feature array then
    # raises an ambiguous-truth-value ValueError and hides later options.
    for idx in range(0, len(arguments), 2):
        key = arguments[idx]
        if isinstance(key, str) and key == name:
            if idx + 1 >= len(arguments):
                raise ValueError(f'Missing value for argument "{name}"')
            return True, arguments[idx + 1]
    return False, None


def _to_list(obj: Any) -> list[Any]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    return [obj]


def input_handler(arguments: Iterable[Any], setting: Any, mode: str):
    """Reimplementation of the MATLAB Input.m logic."""
    args = list(arguments)
    set_ns = _ensure_namespace(setting)

    if mode == "data":
        problem_found, problem = _find_argument(args, "Problem")
        if not problem_found:
            raise ValueError("Please set the targeted problem.")
        if set_ns.Mode == "design":
            train_found, train = _find_argument(args, "InstanceTrain")
            test_found, test = _find_argument(args, "InstanceTest")
            if not train_found or not test_found:
                raise ValueError("Please set the targeted problem instance indexes.")
            return problem, _to_list(train), _to_list(test)

        if set_ns.Mode == "solve":
            solve_found, solve = _find_argument(args, "InstanceSolve")
            if not solve_found:
                raise ValueError("Please set the targeted problem instance indexes.")
            return problem, _to_list(solve)

        raise ValueError('Please set the mode to "design" or "solve".')

    if mode == "parameter":
        for key in _PARAM_KEYS:
            found, value = _find_argument(args, key)
            if found:
                setattr(set_ns, key, value)
        defaults = dict(_COMMON_DEFAULTS)
        defaults.update(
            _DESIGN_DEFAULTS if set_ns.Mode == "design" else _SOLVE_DEFAULTS
        )
        if set_ns.Mode == "design":
            train_found, train = _find_argument(args, "InstanceTrain")
            train_count = len(_to_list(train)) if train_found else 1
            defaults["RacingK"] = max(1, int(round(train_count * 0.2)))
            runs_found, runs = _find_argument(args, "AlgRuns")
            run_count = (
                int(cast(Any, runs))
                if runs_found
                else int(cast(Any, defaults["AlgRuns"]))
            )
            defaults["IntensificationMaxConfigCalls"] = min(
                3, max(1, train_count * run_count)
            )
            prob_fe: Any = getattr(set_ns, "ProbFE", defaults["ProbFE"])
            defaults["Surro"] = max(1, int(math.floor(float(prob_fe) * 0.3 + 0.5)))
        if (
            set_ns.Mode == "design"
            and str(getattr(set_ns, "Designer", "search")).lower() == "search"
        ):
            defaults.update(GraphSemantics="stream_graph_v2", AlgP=2, AlgQ=2)
        for key, value in defaults.items():
            if not hasattr(set_ns, key):
                setattr(set_ns, key, deepcopy(value))
        return set_ns

    if mode == "check":
        _check_setting(set_ns)
        return set_ns

    raise ValueError(f"Unsupported mode: {mode}")


def _check_setting(setting: SimpleNamespace) -> None:
    mode = getattr(setting, "Mode", None)
    if mode not in {"design", "solve"}:
        raise ValueError('Please set the mode to "design" or "solve".')
    # Resolve these once during validation so malformed design grids fail
    # before an expensive design run starts.
    setting.PopulationSizeSpace = list(population_size_space(setting))
    setting.OffspringSizeSpace = list(offspring_size_space(setting))
    boundary_handling = str(getattr(setting, "BoundaryHandling", "clip")).lower()
    if boundary_handling not in BOUNDARY_HANDLING_SPACE:
        choices = ", ".join(BOUNDARY_HANDLING_SPACE)
        raise ValueError(f"BoundaryHandling must be one of: {choices}.")
    setting.BoundaryHandling = boundary_handling
    graph_semantics = str(
        getattr(setting, "GraphSemantics", "legacy_pathway_v1")
    ).lower()
    if graph_semantics not in {"legacy_pathway_v1", "stream_graph_v2"}:
        raise ValueError(
            "GraphSemantics must be 'legacy_pathway_v1' or 'stream_graph_v2'."
        )
    setting.GraphSemantics = graph_semantics

    stream_event_budget_multiplier = getattr(
        setting, "StreamEventBudgetMultiplier", None
    )
    if stream_event_budget_multiplier is not None:
        if (
            not isinstance(stream_event_budget_multiplier, Real)
            or isinstance(stream_event_budget_multiplier, bool)
            or not math.isfinite(float(stream_event_budget_multiplier))
            or float(stream_event_budget_multiplier) <= 0
        ):
            raise ValueError(
                "StreamEventBudgetMultiplier must be a positive finite number or None."
            )
        setting.StreamEventBudgetMultiplier = float(stream_event_budget_multiplier)

    retries = getattr(setting, "EvalRetries", 0)
    if not isinstance(retries, Integral) or isinstance(retries, bool) or retries < 0:
        raise ValueError("EvalRetries must be a non-negative integer.")
    timeout = getattr(setting, "EvalTimeoutSec", None)
    if timeout is not None and (
        not isinstance(timeout, Real)
        or isinstance(timeout, bool)
        or not math.isfinite(float(timeout))
        or timeout <= 0
    ):
        raise ValueError("EvalTimeoutSec must be a positive finite number or None.")
    failure = str(getattr(setting, "EvalFailure", "raise")).lower()
    if failure not in {"raise", "penalize"}:
        raise ValueError("EvalFailure must be 'raise' or 'penalize'.")
    setting.EvalFailure = failure
    penalty = getattr(setting, "EvalPenalty", 1e30)
    if (
        not isinstance(penalty, Real)
        or isinstance(penalty, bool)
        or not math.isfinite(float(penalty))
    ):
        raise ValueError("EvalPenalty must be a finite numeric scalar.")
    if not isinstance(getattr(setting, "EvalCache", False), bool):
        raise ValueError("EvalCache must be a boolean.")
    eval_log = getattr(setting, "EvalLog", None)
    if eval_log is not None and not isinstance(eval_log, (str, os.PathLike)):
        raise ValueError("EvalLog must be a filesystem path or None.")
    eval_workers = getattr(setting, "EvalWorkers", 1)
    if isinstance(eval_workers, str):
        if eval_workers.lower() != "auto":
            raise ValueError("EvalWorkers must be 'auto' or a positive integer.")
        setting.EvalWorkers = "auto"
    elif (
        not isinstance(eval_workers, Integral)
        or isinstance(eval_workers, bool)
        or int(eval_workers) <= 0
    ):
        raise ValueError("EvalWorkers must be 'auto' or a positive integer.")
    backend = str(getattr(setting, "EvalBackend", "auto")).lower()
    if backend not in {"auto", "serial", "process"}:
        raise ValueError("EvalBackend must be 'auto', 'serial', or 'process'.")
    setting.EvalBackend = backend
    cores_per_worker = getattr(setting, "EvalCoresPerWorker", 1)
    if (
        not isinstance(cores_per_worker, Integral)
        or isinstance(cores_per_worker, bool)
        or int(cores_per_worker) <= 0
    ):
        raise ValueError("EvalCoresPerWorker must be a positive integer.")
    max_cores_per_task = getattr(setting, "EvalMaxCoresPerTask", cores_per_worker)
    if max_cores_per_task is None:
        max_cores_per_task = cores_per_worker
        setting.EvalMaxCoresPerTask = max_cores_per_task
    if (
        not isinstance(max_cores_per_task, Integral)
        or isinstance(max_cores_per_task, bool)
        or max_cores_per_task < cores_per_worker
    ):
        raise ValueError("EvalMaxCoresPerTask must be at least EvalCoresPerWorker.")
    if not isinstance(getattr(setting, "EvalAffinity", True), bool):
        raise ValueError("EvalAffinity must be a boolean.")
    if not isinstance(getattr(setting, "EvalAdaptive", True), bool):
        raise ValueError("EvalAdaptive must be a boolean.")
    task_timeout = getattr(setting, "EvalTaskTimeoutSec", None)
    if task_timeout is not None and (
        not isinstance(task_timeout, Real)
        or isinstance(task_timeout, bool)
        or not math.isfinite(float(task_timeout))
        or task_timeout <= 0
    ):
        raise ValueError("EvalTaskTimeoutSec must be positive and finite or None.")
    memory_limit = getattr(setting, "EvalMemoryLimitBytes", None)
    if memory_limit is not None and (
        not isinstance(memory_limit, Integral)
        or isinstance(memory_limit, bool)
        or memory_limit <= 0
    ):
        raise ValueError("EvalMemoryLimitBytes must be positive or None.")
    oversubscription = getattr(setting, "EvalMaxOversubscription", 2.0)
    if (
        not isinstance(oversubscription, Real)
        or isinstance(oversubscription, bool)
        or not math.isfinite(float(oversubscription))
        or oversubscription < 1
    ):
        raise ValueError("EvalMaxOversubscription must be finite and at least 1.")
    io_threshold = getattr(setting, "EvalIOCPUThreshold", 0.35)
    if (
        not isinstance(io_threshold, Real)
        or isinstance(io_threshold, bool)
        or not math.isfinite(float(io_threshold))
        or not 0 < float(io_threshold) < 1
    ):
        raise ValueError("EvalIOCPUThreshold must be between 0 and 1.")
    shared_threshold = getattr(setting, "EvalSharedMemoryThreshold", 1_048_576)
    if (
        not isinstance(shared_threshold, Integral)
        or isinstance(shared_threshold, bool)
        or shared_threshold < 0
    ):
        raise ValueError("EvalSharedMemoryThreshold must be non-negative.")
    resource_estimator = getattr(setting, "EvalResourceEstimator", None)
    if resource_estimator is not None and not callable(resource_estimator):
        raise ValueError("EvalResourceEstimator must be callable or None.")
    concurrency_profile_path = getattr(setting, "EvalConcurrencyProfilePath", None)
    if concurrency_profile_path is not None and not isinstance(
        concurrency_profile_path, (str, os.PathLike)
    ):
        raise ValueError("EvalConcurrencyProfilePath must be a path or None.")
    worker_initializer = getattr(setting, "EvalWorkerInitializer", None)
    worker_finalizer = getattr(setting, "EvalWorkerFinalizer", None)
    if worker_initializer is not None and not callable(worker_initializer):
        raise ValueError("EvalWorkerInitializer must be callable or None.")
    if worker_finalizer is not None and not callable(worker_finalizer):
        raise ValueError("EvalWorkerFinalizer must be callable or None.")
    if worker_finalizer is not None and worker_initializer is None:
        raise ValueError(
            "EvalWorkerFinalizer requires EvalWorkerInitializer to be configured."
        )
    checkpoint_dir = getattr(setting, "CheckpointDir", None)
    if checkpoint_dir is not None and not isinstance(
        checkpoint_dir, (str, os.PathLike)
    ):
        raise ValueError("CheckpointDir must be a filesystem path or None.")
    checkpoint_every = getattr(setting, "CheckpointEvery", 1)
    if (
        not isinstance(checkpoint_every, Integral)
        or isinstance(checkpoint_every, bool)
        or checkpoint_every <= 0
    ):
        raise ValueError("CheckpointEvery must be a positive integer.")
    checkpoint_signature = getattr(setting, "CheckpointSignature", None)
    if checkpoint_signature is not None and not isinstance(checkpoint_signature, str):
        raise ValueError("CheckpointSignature must be a string or None.")
    coordinate_seeds = getattr(setting, "EvalCoordinateSeeds", None)
    if coordinate_seeds is not None:
        if not isinstance(coordinate_seeds, (Sequence, dict)) or isinstance(
            coordinate_seeds, (str, bytes)
        ):
            raise ValueError(
                "EvalCoordinateSeeds must be a sequence, mapping, or None."
            )
        values = (
            coordinate_seeds.values()
            if isinstance(coordinate_seeds, dict)
            else coordinate_seeds
        )
        if any(
            not isinstance(value, Integral) or isinstance(value, bool) or value < 0
            for value in values
        ):
            raise ValueError(
                "EvalCoordinateSeeds values must be non-negative integers."
            )
    training_scorer = getattr(setting, "EvalTrainingScorer", None)
    if training_scorer is not None and not callable(training_scorer):
        raise ValueError("EvalTrainingScorer must be callable or None.")
    candidate_log = getattr(setting, "CandidateLogPath", None)
    if candidate_log is not None and not isinstance(candidate_log, (str, os.PathLike)):
        raise ValueError("CandidateLogPath must be a filesystem path or None.")
    if not isinstance(getattr(setting, "Resume", False), bool):
        raise ValueError("Resume must be a boolean.")
    budget_extension = getattr(setting, "ResumeBudgetExtension", False)
    if not isinstance(budget_extension, bool):
        raise ValueError("ResumeBudgetExtension must be a boolean.")
    if budget_extension:
        if not getattr(setting, "Resume", False):
            raise ValueError("ResumeBudgetExtension requires Resume=True.")
        source_alg_fe = getattr(setting, "ResumeBudgetExtensionSourceAlgFE", None)
        if (
            not isinstance(source_alg_fe, Integral)
            or isinstance(source_alg_fe, bool)
            or source_alg_fe <= 0
        ):
            raise ValueError(
                "ResumeBudgetExtensionSourceAlgFE must be a positive integer."
            )
        source_signature = getattr(
            setting,
            "ResumeBudgetExtensionSourceCheckpointSignature",
            None,
        )
        if not isinstance(source_signature, str) or not source_signature:
            raise ValueError(
                "ResumeBudgetExtensionSourceCheckpointSignature must be a "
                "non-empty string."
            )
    designer = str(getattr(setting, "Designer", "search")).lower()
    if designer not in {"search", "learning", "random"}:
        raise ValueError("Designer must be 'search', 'learning', or 'random'.")
    setting.Designer = designer
    for removed in ("LearningMode", "LearningFeatures"):
        if hasattr(setting, removed):
            raise ValueError(
                f"{removed} is no longer supported; use an unconditioned Learning model."
            )
    learning_device = str(getattr(setting, "LearningDevice", "auto")).lower()
    if not (
        learning_device in {"auto", "cpu", "cuda", "mps", "rocm", "amd", "hip"}
        or learning_device.startswith("cuda:")
    ):
        raise ValueError(
            "LearningDevice must be auto, cpu, cuda, cuda:N, mps, or rocm."
        )
    setting.LearningDevice = learning_device
    if mode == "design":
        structure_rate = getattr(setting, "StructureMutationRate", 0.30)
        if (
            not isinstance(structure_rate, Real)
            or isinstance(structure_rate, bool)
            or not math.isfinite(float(structure_rate))
            or not 0 < float(structure_rate) <= 1
        ):
            raise ValueError("StructureMutationRate must be in (0, 1].")
        archive_size = getattr(setting, "SearchArchiveSize", 1)
        stagnation = getattr(setting, "SearchStagnation", 3)
        max_attempts = getattr(setting, "SearchMaxAttempts", 50)
        for name, value in (
            ("SearchArchiveSize", archive_size),
            ("SearchStagnation", stagnation),
            ("SearchMaxAttempts", max_attempts),
        ):
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        restart_fraction = getattr(setting, "SearchRestartFraction", 0.50)
        if (
            not isinstance(restart_fraction, Real)
            or isinstance(restart_fraction, bool)
            or not math.isfinite(float(restart_fraction))
            or not 0 < float(restart_fraction) <= 1
        ):
            raise ValueError("SearchRestartFraction must be in (0, 1].")
        for name, default, lower, upper in (
            ("SearchImprovementRate", 0.05, 0.0, 1.0),
            ("SearchParameterActionProbability", 0.50, 0.0, 1.0),
            ("SearchActionProbabilityGain", 0.30, 0.0, 1.0),
            ("SearchActionRewardEWMA", 0.20, 0.0, 1.0),
        ):
            value = getattr(setting, name, default)
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not lower <= float(value) <= upper
            ):
                raise ValueError(f"{name} must be between {lower} and {upper}.")
        initial_probability = float(
            getattr(setting, "SearchParameterActionProbability", 0.50)
        )
        probability_gain = float(getattr(setting, "SearchActionProbabilityGain", 0.30))
        if probability_gain > min(initial_probability, 1.0 - initial_probability):
            raise ValueError(
                "SearchActionProbabilityGain must keep the un-clipped adaptive "
                "probability within [0, 1]."
            )
        initial_sigma = getattr(setting, "ParameterCMAInitialSigma", 1.0)
        if (
            not isinstance(initial_sigma, Real)
            or isinstance(initial_sigma, bool)
            or not math.isfinite(float(initial_sigma))
            or not 1e-3 <= float(initial_sigma) <= 5.0
        ):
            raise ValueError("ParameterCMAInitialSigma must be in [0.001, 5].")
        for name in (
            "ParameterCMAOffspring",
            "StructureCandidatesPerAction",
            "ParameterCMABlockGenerations",
            "PostStructureCMAGenerations",
        ):
            value = getattr(setting, name, None)
            if value is not None and (
                not isinstance(value, Integral) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer when provided.")
        for name in (
            "AlgP",
            "AlgQ",
            "ProbN",
            "ProbFE",
            "InnerFE",
            "AlgN",
            "AlgFE",
            "AlgRuns",
        ):
            value = getattr(setting, name, None)
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if (
            graph_semantics == "legacy_pathway_v1"
            and getattr(setting, "AlgP", 0) > 1
            and getattr(setting, "AlgQ", 0) > 1
        ):
            raise ValueError(
                "Use either multiple pathways (AlgP>1) or multiple search "
                "operators per pathway (AlgQ>1), not both."
            )
        if graph_semantics == "stream_graph_v2" and (
            getattr(setting, "AlgP", 0) > 2 or getattr(setting, "AlgQ", 0) > 2
        ):
            raise ValueError(
                "Search v6.0 stream graphs currently require AlgP<=2 and AlgQ<=2."
            )
        if getattr(setting, "AlgN", 0) > getattr(setting, "AlgFE", 0):
            raise ValueError(
                "The number of algorithms should not be larger than the evaluation budget."
            )
        if getattr(setting, "AlgRuns", 0) > getattr(setting, "ProbFE", 0):
            raise ValueError(
                "The number of runs should not exceed problem evaluations."
            )
        if getattr(setting, "ProbN", 0) > getattr(setting, "ProbFE", 0):
            raise ValueError(
                "ProbFE must be at least ProbN so the initial population fits the budget."
            )
        initial_designs = getattr(setting, "SearchInitialDesigns", [])
        if (
            isinstance(initial_designs, (str, bytes))
            or not isinstance(initial_designs, Sequence)
            or any(not isinstance(item, dict) for item in initial_designs)
        ):
            raise ValueError("SearchInitialDesigns must be a sequence of mappings.")
        if len(initial_designs) > int(getattr(setting, "AlgN", 0)):
            raise ValueError(
                "SearchInitialDesigns cannot contain more entries than AlgN."
            )
        if initial_designs and graph_semantics != "stream_graph_v2":
            raise ValueError(
                "SearchInitialDesigns currently requires GraphSemantics='stream_graph_v2'."
            )

        evaluate = str(getattr(setting, "Evaluate", "exact")).lower()
        compare = str(getattr(setting, "Compare", "average")).lower()
        metric_token = str(getattr(setting, "Metric", "quality")).lower()
        metric_names = {
            "quality": "quality",
            "runtimefe": "runtimeFE",
            "runtimesec": "runtimeSec",
            "auc": "auc",
        }
        if evaluate != "exact":
            raise ValueError("Search V6 supports Evaluate=exact only.")
        if compare not in {"average", "statistic"}:
            raise ValueError("Compare must be 'average' or 'statistic'.")
        if metric_token not in metric_names:
            raise ValueError(
                "Metric must be one of: quality, runtimeFE, runtimeSec, auc."
            )
        metric = metric_names[metric_token]
        setting.Evaluate = evaluate
        setting.Compare = compare
        setting.Metric = metric
        design_fes = getattr(setting, "DesignFEs", None)
        if design_fes is not None:
            if (
                not isinstance(design_fes, Integral)
                or isinstance(design_fes, bool)
                or design_fes <= 0
            ):
                raise ValueError("DesignFEs must be a positive integer.")
            if designer != "search" or evaluate not in {"exact", "racing"}:
                raise ValueError(
                    "DesignFEs currently supports exact/racing Search only."
                )
        if (
            compare == "statistic"
            and getattr(setting, "AlgRuns", 1) == 1
            and evaluate != "racing"
            and int(getattr(setting, "_DesignStatisticalBlocks", 1)) < 2
        ):
            raise ValueError(
                "Please use at least two statistical blocks (multiple training "
                'tasks or Setting.AlgRuns>1) with the "statistic" comparison '
                "method."
            )
        if getattr(setting, "ProbN", 0) < 5 and getattr(setting, "AlgP", 0) > 1:
            warnings.warn(
                "It is better to have a large population size if involving the EDA operator",
                stacklevel=2,
            )
        if getattr(setting, "AlgQ", 0) > 4:
            warnings.warn(
                "AlgQ is recommended to be larger than 4 for discrete and permutation problems due to the lack of so many search operators",
                stacklevel=2,
            )
    elif mode == "solve":
        for name in ("ProbN", "ProbFE", "AlgRuns"):
            value = getattr(setting, name, None)
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if setting.ProbN > setting.ProbFE:
            raise ValueError(
                "ProbFE must be at least ProbN so the initial population fits the budget."
            )
        if not getattr(setting, "AlgFile", None) and not getattr(
            setting, "AlgName", None
        ):
            raise ValueError(
                "Please specify an algorithm file in Setting.AlgFile or specify an algorithm name in Setting.AlgName."
            )

        metric_token = str(getattr(setting, "Metric", "quality")).lower()
        metric_names = {
            "quality": "quality",
            "runtimefe": "runtimeFE",
            "runtimesec": "runtimeSec",
            "auc": "auc",
        }
        if metric_token not in metric_names:
            raise ValueError(
                "Metric must be one of: quality, runtimeFE, runtimeSec, auc."
            )
        metric = metric_names[metric_token]
        setting.Metric = metric
        if metric == "runtimeFE":
            if getattr(setting, "Tmax", None) in (None, []):
                setting.Tmax = getattr(setting, "ProbFE", None)
            if getattr(setting, "Thres", None) in (None, []):
                raise ValueError(
                    'Please set "Setting.Thres" as the lowest acceptable performance of the design algorithms, '
                    "the performance can be the solution quality."
                )
        if metric == "runtimeSec":
            if getattr(setting, "Tmax", None) in (None, []):
                raise ValueError(
                    'Please set "Setting.Tmax" as the maximum runtime (seconds).'
                )
            if getattr(setting, "Thres", None) in (None, []):
                raise ValueError(
                    'Please set "Setting.Thres" as the lowest acceptable performance of the design algorithms, '
                    "the performance can be the solution quality."
                )
        if metric == "auc":
            tmax = getattr(setting, "Tmax", None)
            thres = getattr(setting, "Thres", None)
            if not isinstance(tmax, Sequence) or len(tmax) <= 1:
                raise ValueError(
                    '"Setting.Tmax" should contain multiple time points. The time points should the numbers of '
                    "function evaluations spent during the alorithm execution."
                )
            if not isinstance(thres, Sequence) or len(thres) != len(tmax):
                raise ValueError(
                    'The number of thresholds in "Setting.Thres" should be equal to the number of time points in '
                    '"Setting.Tmax". "Setting.Thres" refers to the lowest acceptable performance of the design '
                    "algorithms, the performance can be the solution quality."
                )
