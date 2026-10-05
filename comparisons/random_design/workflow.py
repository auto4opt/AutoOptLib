"""RandomDesign experiment workflow, using the shared graph executor."""

from __future__ import annotations

import json
import time
from pathlib import Path

from autooptlib import autoopt, save_algorithm

from comparisons.shared.protocol import ExperimentProtocol
from comparisons.shared.scoring import make_taskwise_scorer
from comparisons.shared.store import atomic_json, utc_now
from methods.design import (
    _artifact_directory,
    _candidate_summary,
    _curve_from_trace,
    _random_evaluation_workers,
    _require_complete_exact_ledger,
)
from problems import make_ioh_problem


def design_random(
    protocol: ExperimentProtocol,
    suite: str,
    function_id: int,
    *,
    artifact_root: Path,
    resume: bool,
) -> Path:
    """Sample the native graph space without search or learned guidance."""
    method = "aol_random"
    output = _artifact_directory(artifact_root, suite, function_id)
    instances = protocol.training_instances(suite, function_id)
    maximum_run_budget = max((int(item.budget or 0) for item in instances))
    design = protocol.design
    candidate_budget = int(design["candidate_budget"])
    search = design["search"]
    evaluation_workers = _random_evaluation_workers(design["workers"])
    training_scorer = make_taskwise_scorer(suite, function_id, instances)
    normalization = training_scorer.name
    candidate_path = output / f"{method}-candidates.jsonl"
    started = time.perf_counter()
    (algorithms, trace) = autoopt(
        Mode="design",
        Designer="random",
        Problem=make_ioh_problem(suite, function_id),
        InstanceTrain=instances,
        InstanceTest=[],
        AlgP=1,
        AlgQ=int(design["components_max"]),
        AlgN=int(design["finalists"]),
        AlgFE=candidate_budget,
        AlgRuns=1,
        ProbN=int(protocol.suites[suite]["population_size"]),
        ProbFE=maximum_run_budget,
        InnerFE=min(500, maximum_run_budget),
        PopulationSizeSpace=list(range(4, int(search["population_size_max"]) + 1)),
        OffspringSizeSpace=list(range(1, int(search["offspring_size_max"]) + 1)),
        Evaluate=str(design["autooptlib_evaluation_policy"][method]),
        Compare="average",
        Seed=protocol.seed + 40000 + function_id + (0 if suite == "bbob" else 10000),
        EvalCommonRandomSeed=protocol.task_seed(
            suite, function_id, 0, 0, phase="training", dimension=0
        ),
        EvalCoordinateSeeds=tuple(
            (
                protocol.task_seed(
                    suite,
                    function_id,
                    instance.instance,
                    instance.repeat,
                    phase="training",
                    dimension=instance.dimension,
                )
                for instance in instances
            )
        ),
        EvalTrainingScorer=training_scorer,
        EvalWorkers=evaluation_workers,
        EvalBackend="auto",
        EvalAdaptive=False,
        EvalConcurrencyProfilePath=artifact_root / "runtime-concurrency-profiles.json",
        CheckpointDir=output / f"{method}-checkpoint",
        CheckpointEvery=int(design["learning"]["batch_size"]),
        CheckpointSignature=f"{protocol.fingerprint()}:aol-random-v5",
        Resume=resume,
        CandidateLogPath=candidate_path,
        OutputDir=output / f"{method}-autoopt-output",
    )
    elapsed_seconds = time.perf_counter() - started
    artifact = output / f"{method}.json"
    save_algorithm(
        algorithms[0],
        artifact,
        metadata={
            "designer": "random",
            "suite": suite,
            "function_id": function_id,
            "protocol_fingerprint": protocol.fingerprint(),
            "candidate_budget": candidate_budget,
            "training_tasks": len(instances),
            "population_size_max": int(search["population_size_max"]),
            "offspring_size_max": int(search["offspring_size_max"]),
        },
    )
    curve = _curve_from_trace(
        list(trace),
        initial_candidates=0,
        batch_size=1,
        training_tasks=len(instances),
        run_budget=protocol.candidate_training_fe(suite) // len(instances),
        candidate_budget=candidate_budget,
        scorer=training_scorer,
    )
    manifest = json.loads(
        (output / f"{method}-autoopt-output" / "experiment.json").read_text(
            encoding="utf-8"
        )
    )
    candidate_summary = _candidate_summary(candidate_path)
    actual_design_fes = int(candidate_summary["actual_design_fes"])
    if candidate_summary["different_candidates"] != candidate_budget:
        raise RuntimeError(
            f"{method} produced {candidate_summary['different_candidates']} different algorithms; expected {candidate_budget}."
        )
    _require_complete_exact_ledger(
        candidate_summary,
        method=method,
        candidate_budget=candidate_budget,
        training_tasks=len(instances),
    )
    atomic_json(
        output / f"{method}-design.json",
        {
            "schema": "autooptlib.paper-design",
            "schema_version": 1,
            "created_utc": utc_now(),
            "method": method,
            "suite": suite,
            "function_id": function_id,
            "protocol_fingerprint": protocol.fingerprint(),
            "candidate_budget": candidate_budget,
            "training_tasks": [vars(item) for item in instances],
            "artifact": str(artifact.resolve()),
            "normalization": normalization,
            "design_ledger": {
                **candidate_summary,
                "actual_design_fes": actual_design_fes,
                "elapsed_seconds_this_invocation": elapsed_seconds,
            },
            "runtime_statistics": manifest["options"].get("RuntimeStats"),
            "curve": curve,
        },
    )
    return artifact
