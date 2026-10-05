"""Final-run adapters for all methods in the registered comparison."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from autooptlib import autoopt

from comparisons.manual.algorithms import _bipop_cmaes, _iterated_local_search, _shade
from comparisons.shared.objectives import (
    BBOB_TARGETS,
    ObjectiveResult,
    TrackedIOHObjective,
)
from comparisons.shared.protocol import EvaluationTask
from comparisons.shared.scoring import checked_optimality_gap
from problems import make_problem


@dataclass
class MethodRun:
    result: ObjectiveResult
    elapsed_seconds: float
    implementation: str
    implementation_version: str | None = None
    extra: dict[str, Any] | None = None


MANUAL_METHOD_REVISIONS = {
    "bipop_cmaes": "pycma-bipop-fe-limited-v2",
    "shade": "shade-1.1.1-author-port-v3",
    "pso": "autooptlib-pso-preset-v1",
    "ga": "autooptlib-discrete-ga-preset-v1",
    "ils": "binary-ils-v2",
    "sa": "autooptlib-discrete-sa-preset-v1",
}


def artifact_path(root: Path, task: EvaluationTask) -> Path:
    return root / task.suite / f"f{task.function_id:02d}" / f"{task.method}.json"


def external_artifact_path(root: Path, task: EvaluationTask) -> Path:
    filename = {
        "paradiseo_irace": "paradiseo-irace.json",
        "sparkle_smac3": "sparkle-smac3.json",
    }.get(task.method)
    if filename is None:
        raise ValueError(f"{task.method!r} is not an external paper method.")
    return root / task.suite / f"f{task.function_id:02d}" / filename


def method_evaluation_identity(task: EvaluationTask, artifact_root: Path) -> str:
    """Return the implementation/artifact identity required for safe resume."""

    if task.method in {"aol_search", "aol_learning", "aol_random"}:
        path = artifact_path(artifact_root, task)
    elif task.method in {"paradiseo_irace", "sparkle_smac3"}:
        path = external_artifact_path(artifact_root, task)
    else:
        revision = MANUAL_METHOD_REVISIONS.get(task.method)
        if revision is None:
            raise KeyError(f"No evaluation identity registered for {task.method!r}.")
        return revision
    if not path.is_file():
        return f"{task.method}:artifact-missing"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"{task.method}:artifact-sha256:{digest}"


def _autoopt_problem(objective: TrackedIOHObjective):
    return make_problem(
        lambda decision, tracker: tracker(decision),
        bounds=lambda tracker: tracker.bounds,
        problem_type="continuous" if objective.task.suite == "bbob" else "discrete",
        data_factory=lambda tracker: tracker,
        name=f"tracked_{objective.task.suite}_f{objective.task.function_id}",
    )


def _run_autoopt(
    task: EvaluationTask,
    objective: TrackedIOHObjective,
    *,
    algorithm_name: str | None = None,
    algorithm_file: Path | None = None,
    population_size: int | None = None,
) -> None:
    options: dict[str, Any]
    if algorithm_file is not None:
        if not algorithm_file.exists():
            raise FileNotFoundError(
                f"Designed algorithm artifact is missing: {algorithm_file}. "
                "Run the `design` command first."
            )
        options = {"AlgFile": str(algorithm_file)}
    elif algorithm_name is not None:
        options = {"AlgName": algorithm_name}
    else:  # pragma: no cover - internal guard
        raise ValueError("An algorithm name or file is required.")
    # AutoOptLib's solve reports are intentionally file based.  Final paper
    # evaluations run in parallel processes, so every task needs its own output
    # directory; otherwise concurrent tasks can remove or replace one another's
    # Solutions.csv and experiment.json temporary files.
    with tempfile.TemporaryDirectory(
        prefix=f"autooptlib-paper-{task.task_id}-"
    ) as output_directory:
        autoopt(
            Mode="solve",
            Problem=_autoopt_problem(objective),
            InstanceSolve=[objective],
            AlgRuns=1,
            ProbN=int(population_size or task.population_size),
            ProbFE=task.budget,
            Seed=task.seed,
            EvalBackend="serial",
            EvalWorkers=1,
            OutputDir=output_directory,
            **options,
        )


def _autoopt_preset(
    task: EvaluationTask,
    objective: TrackedIOHObjective,
    name: str,
    population: int,
) -> tuple[str, dict[str, Any]]:
    _run_autoopt(task, objective, algorithm_name=name, population_size=population)
    return "AutoOptLib preset", {"preset": name, "population_size": population}


def _external(
    task: EvaluationTask,
    objective: TrackedIOHObjective,
    manifest_path: Path,
    artifact_root: Path,
) -> tuple[str, dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"External method manifest not found: {manifest_path}. "
            "Copy external-methods.example.json, set the built solver commands, "
            "and keep the exact executable/config commits in the manifest."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest.get(task.method)
    if not isinstance(entry, dict) or not isinstance(entry.get("command"), list):
        raise ValueError(f"No command registered for {task.method!r}.")
    substitutions = {
        "suite": task.suite,
        "function": task.function_id,
        "dimension": task.dimension,
        "instance": task.instance,
        "seed": task.seed,
        "budget": task.budget,
        "artifact": str(external_artifact_path(artifact_root, task).resolve()),
    }
    command = [str(token).format(**substitutions) for token in entry["command"]]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    payload = json.loads(completed.stdout)
    expected = {
        "suite": task.suite,
        "function": task.function_id,
        "dimension": task.dimension,
        "instance": task.instance,
        "seed": task.seed,
        "budget": task.budget,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(
                f"External result field {name!r} must be {value!r}, "
                f"got {payload.get(name)!r}."
            )
    evaluations = payload["evaluations"]
    if type(evaluations) is not int or evaluations != task.budget:
        raise ValueError(
            f"External solver used {evaluations} evaluations; expected {task.budget}."
        )
    best_raw = float(payload["best_raw"])
    if not np.isfinite(best_raw):
        raise ValueError("External solver returned a non-finite best_raw value.")
    if "optimum_raw" in payload:
        reported_optimum = payload["optimum_raw"]
        expected_optimum = objective.optimum_raw
        if expected_optimum is None:
            if reported_optimum is not None:
                raise ValueError(
                    "External optimum_raw contradicts the unknown optimum."
                )
        elif (
            type(reported_optimum) not in (int, float)
            or not np.isfinite(reported_optimum)
            or not math.isclose(
                reported_optimum, expected_optimum, rel_tol=1e-8, abs_tol=1e-8
            )
        ):
            raise ValueError(
                "External optimum_raw differs from the registered optimum."
            )
    best_cost = best_raw if task.suite == "bbob" else -best_raw
    gap = (
        None
        if objective.optimum_cost is None
        else checked_optimality_gap(best_cost, objective.optimum_cost)
    )
    targets = (
        BBOB_TARGETS if task.suite == "bbob" else (0.0,) if gap is not None else ()
    )
    raw_hits = payload.get("target_hits")
    if not isinstance(raw_hits, dict):
        raise ValueError("External solver must return exact target_hits.")
    expected_keys = {f"{target:.12g}" for target in targets}
    if set(raw_hits) - expected_keys:
        raise ValueError("External target_hits contains an unregistered target.")
    target_hits = {}
    for key, value in raw_hits.items():
        if type(value) is not int or not 1 <= value <= evaluations:
            raise ValueError(f"External target_hits[{key!r}] is outside the FE budget.")
        target_hits[key] = value
    if gap is not None:
        for target in targets:
            key = f"{target:.12g}"
            if (key in target_hits) != (gap <= target):
                raise ValueError(
                    f"External target_hits[{key!r}] contradicts the final gap."
                )
    raw_trajectory = payload.get("trajectory", [])
    if not isinstance(raw_trajectory, list):
        raise ValueError("External trajectory must be an array.")
    trajectory = []
    previous_fe = 0
    previous_cost = float("inf")
    for raw_item in raw_trajectory:
        if not isinstance(raw_item, dict) or "best_raw" not in raw_item:
            raise ValueError("External trajectory points need best_raw.")
        fe = raw_item.get("evaluations")
        raw = float(raw_item["best_raw"])
        cost = raw if task.suite == "bbob" else -raw
        if "best_cost" in raw_item and not math.isclose(
            float(raw_item["best_cost"]), cost, rel_tol=1e-10, abs_tol=1e-8
        ):
            raise ValueError("External trajectory best_cost contradicts best_raw.")
        if (
            type(fe) is not int
            or not previous_fe < fe <= evaluations
            or not np.isfinite(cost)
        ):
            raise ValueError("External trajectory has invalid FE or objective values.")
        if cost > previous_cost + 1e-8 * max(1.0, abs(previous_cost), abs(cost)):
            raise ValueError("External trajectory is not best-so-far monotone.")
        if objective.optimum_cost is not None:
            checked_optimality_gap(cost, objective.optimum_cost)
        trajectory.append({**raw_item, "best_cost": cost})
        previous_fe, previous_cost = fe, cost
    if trajectory and not math.isclose(
        previous_cost, best_cost, rel_tol=1e-10, abs_tol=1e-8
    ):
        raise ValueError("External trajectory does not end at the reported best_raw.")
    # External solvers operate on the same IOH objective and report raw values.
    objective.evaluations = evaluations
    objective.best_raw = best_raw
    objective.best_cost = best_cost
    objective.trajectory = trajectory
    objective.target_hits = target_hits
    return str(entry.get("implementation", task.method)), {
        "command": command,
        "provenance": entry.get("provenance", {}),
        "stderr": completed.stderr[-4000:],
    }


def run_method(
    task: EvaluationTask,
    *,
    artifact_root: Path,
    external_manifest: Path | None,
    trajectory_points: int,
    algorithm_file: Path | None = None,
) -> MethodRun:
    objective = TrackedIOHObjective(task, trajectory_points=trajectory_points)
    started = time.perf_counter()
    implementation: str
    extra: dict[str, Any]
    if task.method in {
        "aol_search",
        "aol_smac3_v6",
        "aol_random_configspace_v6",
        "aol_learning",
        "aol_random",
    }:
        path = algorithm_file or artifact_path(artifact_root, task)
        _run_autoopt(task, objective, algorithm_file=path)
        implementation, extra = "AutoOptLib designed algorithm", {"artifact": str(path)}
    elif task.method == "bipop_cmaes":
        implementation, extra = _bipop_cmaes(task, objective)
    elif task.method == "shade":
        implementation, extra = _shade(task, objective)
    elif task.method == "pso":
        implementation, extra = _autoopt_preset(
            task, objective, "Particle Swarm Optimization", task.population_size
        )
    elif task.method == "ga":
        implementation, extra = _autoopt_preset(
            task, objective, "Discrete Genetic Algorithm", task.population_size
        )
    elif task.method == "ils":
        implementation, extra = _iterated_local_search(task, objective)
    elif task.method == "sa":
        implementation, extra = _autoopt_preset(
            task, objective, "Discrete Simulated Annealing", 1
        )
    elif task.method in {"paradiseo_irace", "sparkle_smac3"}:
        if external_manifest is None:
            raise FileNotFoundError("External methods require --external-manifest.")
        implementation, extra = _external(
            task, objective, external_manifest, artifact_root
        )
    else:
        raise KeyError(f"Unknown experiment method {task.method!r}.")
    elapsed = time.perf_counter() - started
    version = extra.pop("version", None)
    return MethodRun(
        result=objective.result(),
        elapsed_seconds=elapsed,
        implementation=implementation,
        implementation_version=version,
        extra=extra,
    )


__all__ = [
    "MethodRun",
    "artifact_path",
    "external_artifact_path",
    "method_evaluation_identity",
    "run_method",
]
