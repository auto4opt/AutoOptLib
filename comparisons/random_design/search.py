"""Random graph-design baseline over AutoOptLib's native search space."""

from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from autooptlib.runtime import DesignEvaluationRuntime
from autooptlib.runtime.executor import RuntimeStatistics
from autooptlib.serialization import algorithm_to_dict
from autooptlib.utils.design import Design
from autooptlib.utils.design._population import (
    configuration_is_valid,
    sample_configuration,
    set_configuration,
)
from autooptlib.utils.general.candidate_log import (
    append_candidate_records,
    trim_candidate_records,
)
from autooptlib.utils.general.process import (
    _build_problem_struct,
    _checkpoint_value,
    _normalize_setting,
)
from autooptlib.utils.space import space

from problems.base import validate_constructed_problems


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    *,
    candidate_count: int,
    finalists: int,
    instance_train: Sequence[Any],
    instance_test: Sequence[Any],
    checkpoint_signature: str,
) -> dict[str, Any]:
    warnings.warn(
        "Loading an AutoOptLib checkpoint uses pickle and can execute arbitrary "
        "code. Resume only checkpoints produced by a trusted run.",
        UserWarning,
        stacklevel=2,
    )
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    expected = {
        "schema": "autooptlib.random-design-checkpoint",
        "schema_version": 4,
        "candidate_count": candidate_count,
        "finalists": finalists,
        "instance_train": list(instance_train),
        "instance_test": list(instance_test),
        "checkpoint_signature": checkpoint_signature,
    }
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid RandomDesign checkpoint: {path}")
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(
                f"RandomDesign checkpoint {name} does not match the current run."
            )
    return payload


def _representation_key(candidate: Design) -> str:
    document = algorithm_to_dict(candidate)
    return json.dumps(
        {
            "configuration": document.get("configuration"),
            "pathways": document["pathways"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _random_checkpoint_signature(problem_descriptor: Any, setting: Any) -> str:
    names = (
        "AlgP",
        "AlgQ",
        "AlgN",
        "AlgFE",
        "AlgRuns",
        "ProbN",
        "PopulationSizeSpace",
        "OffspringSizeSpace",
        "ProbFE",
        "InnerFE",
        "Metric",
        "Compare",
        "Evaluate",
        "Seed",
        "EvalCommonRandomSeed",
        "EvalCoordinateSeeds",
        "EvalTrainingScorer",
        "CheckpointSignature",
        "op_space",
        "para_space",
        "para_type_space",
    )
    payload = {
        "implementation": "autooptlib-random-design-v4",
        "problem": _checkpoint_value(problem_descriptor),
        "settings": {
            name: _checkpoint_value(
                getattr(setting, name, getattr(setting, name.lower(), None))
            )
            for name in names
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cost(
    candidate: Design,
    train_indices: Sequence[int],
    scorer: Any = None,
) -> float:
    values = np.asarray(candidate.performance, dtype=float)[list(train_indices), :]
    if values.size == 0 or not np.all(np.isfinite(values)):
        return float("inf")
    value = (
        float(scorer(values, train_indices))
        if scorer is not None
        else float(np.mean(values))
    )
    return value if np.isfinite(value) else float("inf")


def design_with_random(
    problem_descriptor: Any,
    instance_train: Sequence[Any],
    instance_test: Sequence[Any],
    *,
    setting: Any,
) -> tuple[list[Design], list[Design]]:
    """Sample independent graphs, evaluate them fairly, and return the best."""

    setting = _normalize_setting(setting)
    instances = list(instance_train) + list(instance_test)
    problems = _build_problem_struct(problem_descriptor, instances, setting)
    problems, data, _ = problem_descriptor(problems, instances, "construct")
    validate_constructed_problems(problems, data)
    setting = space(problems, setting)

    candidate_count = int(setting.AlgFE)
    finalist_count = int(setting.AlgN)
    seed = int(getattr(setting, "Seed", 0) or 0)
    sample_rng = np.random.default_rng(seed)
    setting.rng = sample_rng
    if getattr(setting, "EvalCommonRandomSeed", None) is None:
        # Candidate sampling advances ``sample_rng`` by a data-dependent
        # amount. A separate deterministic CRN root keeps evaluation streams
        # identical when resume skips already sampled candidates.
        setting.EvalCommonRandomSeed = int(
            np.random.SeedSequence([seed, 0x52414E44]).generate_state(
                1, dtype=np.uint64
            )[0]
        )
    train_indices = sample_rng.permutation(len(instance_train)).tolist()
    test_indices = (
        sample_rng.permutation(len(instance_test)) + len(instance_train)
    ).tolist()
    checkpoint_dir = getattr(setting, "CheckpointDir", None)
    checkpoint = (
        None if checkpoint_dir is None else Path(checkpoint_dir) / "random-design.pkl"
    )
    resume = bool(getattr(setting, "Resume", False))
    candidate_log_value = getattr(setting, "CandidateLogPath", None)
    candidate_log = None if candidate_log_value is None else Path(candidate_log_value)
    scorer = getattr(
        setting,
        "EvalTrainingScorer",
        getattr(setting, "evaltrainingscorer", None),
    )
    checkpoint_signature = _random_checkpoint_signature(problem_descriptor, setting)

    payload = None
    if resume and checkpoint is not None and checkpoint.exists():
        payload = _load_checkpoint(
            checkpoint,
            candidate_count=candidate_count,
            finalists=finalist_count,
            instance_train=instance_train,
            instance_test=instance_test,
            checkpoint_signature=checkpoint_signature,
        )
    if payload is None:
        candidates: list[Design] = []
        seen: set[str] = set()
        attempts = 0
        maximum_attempts = max(100, candidate_count * 100)
        while len(candidates) < candidate_count:
            candidate = Design(problems, setting)
            configuration = sample_configuration(setting, sample_rng, candidate)
            if not configuration_is_valid(candidate, configuration):
                attempts += 1
                if attempts >= maximum_attempts:
                    raise RuntimeError(
                        "RandomDesign could not sample a valid algorithm configuration."
                    )
                continue
            set_configuration(candidate, configuration)
            key = _representation_key(candidate)
            attempts += 1
            if key in seen:
                if attempts >= maximum_attempts:
                    raise RuntimeError(
                        "RandomDesign could not sample the requested number of "
                        "different algorithm representations."
                    )
                continue
            seen.add(key)
            metadata = dict(getattr(candidate, "metadata", {}) or {})
            metadata.update(
                {
                    "designer": "random",
                    "candidate": len(candidates) + 1,
                    "sampling_seed": seed,
                }
            )
            candidate.metadata = metadata
            candidates.append(candidate)
        evaluated = 0
        best_cost = float("inf")
        best_candidate: Design | None = None
        trace: list[Design] = []
        if candidate_log is not None:
            candidate_log.unlink(missing_ok=True)
    else:
        candidates = list(payload["candidates"])
        evaluated = int(payload["evaluated"])
        best_cost = float(payload["best_cost"])
        best_candidate = payload.get("best_candidate")
        if evaluated and best_candidate is None:
            # Schema v4 checkpoints always persist this value.  Keeping the
            # derivation makes the in-memory invariant explicit and protects
            # trusted programmatic callers that construct payloads in tests.
            best_candidate = deepcopy(
                min(
                    candidates[:evaluated],
                    key=lambda item: _cost(item, train_indices, scorer),
                )
            )
        trace = list(payload["trace"])
        train_indices = list(payload["train_indices"])
        test_indices = list(payload["test_indices"])
        trim_candidate_records(candidate_log, evaluated)
        if bool(payload.get("complete", False)):
            finalists = sorted(
                candidates, key=lambda item: _cost(item, train_indices, scorer)
            )[:finalist_count]
            setting.EvalRuntimeStats = dict(payload.get("runtime_statistics", {}))
            setting.EvalRuntimeProfiles = dict(payload.get("runtime_profiles", {}))
            setting.EvalConcurrencyProfiles = dict(
                payload.get("concurrency_profiles", {})
            )
            setting.EvalActualDesignFEs = int(payload.get("actual_design_fes", 0))
            return finalists, trace

    batch_size = min(
        candidate_count,
        max(16, int(getattr(setting, "CheckpointEvery", 1))),
    )
    runtime = DesignEvaluationRuntime(problems, data, setting)
    if payload is not None:
        runtime.total_evaluations = int(payload.get("actual_design_fes", 0))
        runtime.statistics = RuntimeStatistics.from_dict(
            payload.get("runtime_statistics")
        )
    try:
        while evaluated < candidate_count:
            stop = min(candidate_count, evaluated + batch_size)
            # Every batch receives the same common random streams. Batching is
            # therefore only a recovery boundary and cannot change rankings.
            setting.rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
            batch = candidates[evaluated:stop]
            runtime.evaluate(batch, setting, train_indices)
            batch_costs = [
                _cost(candidate, train_indices, scorer) for candidate in batch
            ]
            if not np.all(np.isfinite(batch_costs)):
                raise FloatingPointError(
                    "RandomDesign candidate evaluation produced a non-finite score."
                )
            for candidate, cost in zip(batch, batch_costs):
                metadata = dict(getattr(candidate, "metadata", {}) or {})
                metadata["training_score"] = float(cost)
                candidate.metadata = metadata
            best_offset = int(np.argmin(batch_costs))
            batch_best_cost = float(batch_costs[best_offset])
            if best_candidate is None or batch_best_cost < best_cost:
                best_cost = batch_best_cost
                best_candidate = deepcopy(batch[best_offset])
            # The design curve is an anytime best-so-far curve.  Recording
            # only the current batch winner can make the reported curve get
            # worse even though final selection still uses all candidates.
            generation_best = deepcopy(best_candidate)
            metadata = dict(getattr(generation_best, "metadata", {}) or {})
            metadata["generation"] = len(trace) + 1
            metadata["cumulative_candidates"] = stop
            metadata["cumulative_design_fes"] = runtime.total_evaluations
            generation_best.metadata = metadata
            trace.append(generation_best)
            append_candidate_records(
                candidate_log,
                batch,
                start=evaluated + 1,
                designer="random",
            )
            evaluated = stop
            if checkpoint is not None:
                _write_checkpoint(
                    checkpoint,
                    {
                        "schema": "autooptlib.random-design-checkpoint",
                        "schema_version": 4,
                        "candidate_count": candidate_count,
                        "finalists": finalist_count,
                        "instance_train": list(instance_train),
                        "instance_test": list(instance_test),
                        "checkpoint_signature": checkpoint_signature,
                        "train_indices": train_indices,
                        "test_indices": test_indices,
                        "candidates": candidates,
                        "evaluated": evaluated,
                        "best_cost": best_cost,
                        "best_candidate": best_candidate,
                        "trace": trace,
                        "actual_design_fes": runtime.total_evaluations,
                        "runtime_statistics": runtime.statistics.as_dict(),
                        "complete": False,
                    },
                )

        finalists = sorted(
            candidates, key=lambda item: _cost(item, train_indices, scorer)
        )[:finalist_count]
        if test_indices:
            setting.rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
            runtime.evaluate(finalists, setting, test_indices)
        setting.EvalRuntimeStats = runtime.statistics.as_dict()
        setting.EvalActualDesignFEs = int(runtime.total_evaluations)
        setting.EvalRuntimeProfiles = getattr(setting, "EvalRuntimeProfiles", {})
        setting.EvalConcurrencyProfiles = getattr(
            setting, "EvalConcurrencyProfiles", {}
        )
        if checkpoint is not None:
            _write_checkpoint(
                checkpoint,
                {
                    "schema": "autooptlib.random-design-checkpoint",
                    "schema_version": 4,
                    "candidate_count": candidate_count,
                    "finalists": finalist_count,
                    "instance_train": list(instance_train),
                    "instance_test": list(instance_test),
                    "checkpoint_signature": checkpoint_signature,
                    "train_indices": train_indices,
                    "test_indices": test_indices,
                    "candidates": candidates,
                    "evaluated": evaluated,
                    "best_cost": best_cost,
                    "best_candidate": best_candidate,
                    "trace": trace,
                    "actual_design_fes": runtime.total_evaluations,
                    "runtime_statistics": setting.EvalRuntimeStats,
                    "runtime_profiles": setting.EvalRuntimeProfiles,
                    "concurrency_profiles": setting.EvalConcurrencyProfiles,
                    "complete": True,
                },
            )
        return finalists, trace
    finally:
        runtime.close()


__all__ = ["design_with_random"]
