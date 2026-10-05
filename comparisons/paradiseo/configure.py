"""Run official irace over the common native eoFastGA template."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

from comparisons.shared.platform_target import (
    DesignLedger,
    configuration_is_valid,
    configuration_key,
    ledger_completion_errors,
    ledger_is_complete,
    normalize_configuration,
)
from comparisons.shared.store import atomic_json

IRACE_PROPOSAL_HEADROOM = 0.10
IRACE_MAX_ATTEMPTS = 4
IRACE_SEED_STRIDE = 1_000_003


def _normalize_irace_row(
    row: dict[str, str | None], suite: str
) -> dict[str, float | int]:
    """Normalize one irace CSV row without treating inactive values as data.

    Official irace writes conditional parameters that are inactive for the
    selected configuration as the literal string ``NA``.  They must be omitted
    so the shared canonicalizer can supply the executable defaults.
    """

    missing = {"", "na", "nan", "<na>"}
    active = {
        name: value
        for name, value in row.items()
        if value is not None and str(value).strip().lower() not in missing
    }
    return normalize_configuration(active, suite)


def _configurations_per_iteration(
    candidate_budget: int,
    *,
    iterations: int,
    minimum_survivors: int,
) -> tuple[int, int]:
    """Over-propose while leaving the ledger as the exact budget authority."""

    proposal_budget = math.ceil(candidate_budget * (1 + IRACE_PROPOSAL_HEADROOM))
    per_iteration = (
        proposal_budget + (iterations - 1) * minimum_survivors + iterations - 1
    ) // iterations
    return per_iteration, proposal_budget


def _parameter_file(
    suite: str,
    *,
    population_size_max: int = 200,
    offspring_size_max: int = 200,
) -> str:
    crossover_max = 3
    mutation_max = 3 if suite == "bbob" else 8
    selector_values = "0,1,2,4,5" if suite == "bbob" else "0,1,2,3,4"
    lines = [
        "# name switch type range",
        'crossover_rate "--crossover_rate " r (0.0, 1.0)',
        f'crossover_selector "--crossover_selector " c ({selector_values})',
        f'crossover "--crossover " c (0,1,2,{crossover_max})',
        f'aftercross_selector "--aftercross_selector " c ({selector_values})',
        'mutation_rate "--mutation_rate " r (0.0, 1.0)',
        f'mutation_selector "--mutation_selector " c ({selector_values})',
        f'mutation "--mutation " c ({",".join(map(str, range(mutation_max + 1)))})',
        'replacement "--replacement " c (0,1,2,3)',
        f'population_size "--population_size " i (4,{int(population_size_max)})',
        f'offspring_size "--offspring_size " i (1,{int(offspring_size_max)})',
    ]
    if suite == "bbob":
        lines.extend(
            [
                'boundary_handling "--boundary_handling " c (0,1,2)',
                (
                    'elite_fraction "--elite_fraction " r (0.0,1.0) | '
                    "crossover_selector == 5 | aftercross_selector == 5 | "
                    "mutation_selector == 5"
                ),
                'de_f "--de_f " r (0.0,1.0) | mutation == 3',
                'de_cr "--de_cr " r (0.0,1.0) | mutation == 3',
                'de_p "--de_p " r (0.0,1.0) | mutation == 3',
            ]
        )
    lines.extend(
        [
            "",
            "[forbidden]",
            "(replacement != 0) & (offspring_size > population_size)",
            "",
        ]
    )
    return "\n".join(lines)


def _write_target_runner(
    path: Path,
    request: Path,
    ledger: Path,
    runner: Path,
    repository: Path,
    *,
    report_evaluations: bool,
) -> None:
    resource_flag = ', "--report-evaluations"' if report_evaluations else ""
    script = f"""#!/usr/bin/env python3
import os
import subprocess
import sys

command = [
    sys.executable, "-m", "comparisons.shared.platform_target",
    "--request", {str(request.resolve())!r},
    "--ledger", {str(ledger.resolve())!r},
    "--runner", {str(runner.resolve())!r},
    "--instance", sys.argv[4],
    {resource_flag.lstrip(", ")}
] + sys.argv[5:]
environment = dict(os.environ)
environment["PYTHONPATH"] = os.pathsep.join([
    {str(repository.resolve())!r},
    {str((repository / "methods").resolve())!r},
    environment.get("PYTHONPATH", ""),
])
raise SystemExit(subprocess.run(command, env=environment).returncode)
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def _ordered_training_tasks(request: dict) -> list[dict]:
    """Interleave dimensions while preserving every registered CRN task."""

    tasks = list(request["training_tasks"])
    dimensions = list(dict.fromkeys(int(task["dimension"]) for task in tasks))
    if len(dimensions) <= 1:
        return tasks
    grouped = {
        dimension: [task for task in tasks if int(task["dimension"]) == dimension]
        for dimension in dimensions
    }
    return [
        grouped[dimension][index]
        for index in range(max(map(len, grouped.values())))
        for dimension in dimensions
        if index < len(grouped[dimension])
    ]


def run_irace(
    request_path: Path,
    output: Path,
    runner: Path,
    work: Path,
    *,
    rscript: str,
    workers: int,
) -> dict:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    repository = Path(__file__).resolve().parents[2]
    work.mkdir(parents=True, exist_ok=True)
    workspace_identity_path = work / "autooptlib-request-identity.json"
    workspace_identity = DesignLedger.request_identity(request)
    if workspace_identity_path.exists():
        recorded_identity = json.loads(
            workspace_identity_path.read_text(encoding="utf-8")
        )
        if recorded_identity != workspace_identity:
            raise RuntimeError(
                "irace workspace belongs to a different immutable design request. "
                "Use a fresh work directory."
            )
    elif (work / "design-ledger.sqlite3").exists() or any(
        work.glob("elite-attempt-*.csv")
    ):
        raise RuntimeError(
            "Refusing to reuse a legacy irace workspace without request identity. "
            "Use a fresh work directory."
        )
    else:
        workspace_identity_path.write_text(
            json.dumps(workspace_identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    instances = work / "instances"
    instances.mkdir(exist_ok=True)
    native_instances = str(request["evaluation_policy"]) == "native_instances"
    paths = []
    if native_instances:
        for index, task in enumerate(_ordered_training_tasks(request)):
            path = instances / f"task-{index:03d}.json"
            path.write_text(json.dumps(task, sort_keys=True) + "\n", encoding="utf-8")
            paths.append(path)
    else:
        # The exact target already aggregates every registered task. Register it
        # once instead of presenting twelve duplicate copies to irace.
        path = instances / "all-training-tasks.json"
        path.write_text('{"aggregate": true}\n', encoding="utf-8")
        paths.append(path)
    (work / "parameters.txt").write_text(
        _parameter_file(
            str(request["suite"]),
            population_size_max=int(request.get("population_size_max", 200)),
            offspring_size_max=int(request.get("offspring_size_max", 200)),
        ),
        encoding="utf-8",
    )
    (work / "instances.txt").write_text(
        "\n".join(str(path.resolve()) for path in paths) + "\n", encoding="utf-8"
    )
    ledger = work / "design-ledger.sqlite3"
    _write_target_runner(
        work / "target-runner",
        request_path,
        ledger,
        runner,
        repository,
        report_evaluations=native_instances,
    )
    if native_instances:
        maximum_calls = 0
    else:
        fe_equivalent_calls = int(request["total_fe_cap"]) // int(
            request["candidate_run_budget"]
        )
        # Legacy exact mode counts cached calls in maxExperiments. Leave bounded
        # headroom while the ledger remains its authoritative candidate/FE cap.
        maximum_calls = max(300, fe_equivalent_calls * 2)
    if native_instances:
        iterations = None
        minimum_survivors = None
        configurations_per_iteration = None
        proposal_budget = None
    else:
        iterations = 3
        minimum_survivors = 5
        configurations_per_iteration, proposal_budget = _configurations_per_iteration(
            int(request["candidate_budget"]),
            iterations=iterations,
            minimum_survivors=minimum_survivors,
        )
    design_ledger = DesignLedger(ledger)
    design_ledger.bind_request(request)
    design_ledger.reclaim_stale_claims()
    summary = design_ledger.summary()
    attempts: list[dict[str, int | str]] = []
    elite_candidates: list[tuple[int, Path, dict[str, float | int]]] = []
    for elite_path in sorted(work.glob("elite-attempt-*.csv")):
        try:
            attempt = int(elite_path.stem.rsplit("-", 1)[-1])
            with elite_path.open(newline="", encoding="utf-8") as stream:
                elite_candidates.append(
                    (
                        attempt,
                        elite_path,
                        _normalize_irace_row(
                            next(csv.DictReader(stream)), str(request["suite"])
                        ),
                    )
                )
        except (OSError, StopIteration, TypeError, ValueError):
            # An interrupted write is not evidence. The durable ledger remains
            # authoritative and a later official attempt can replace it.
            continue

    # A race may finish before the durable contract is complete. Continue an
    # independent official attempt against the same cache; only previously
    # unseen candidate-task pairs advance the registered FE budget.
    existing_attempts = [item[0] for item in elite_candidates]
    first_attempt = max(existing_attempts, default=-1) + 1
    for attempt in range(first_attempt, first_attempt + IRACE_MAX_ATTEMPTS):
        if ledger_is_complete(summary, request):
            break
        attempt_seed = int(request["seed"]) + attempt * IRACE_SEED_STRIDE
        result_path = work / f"irace-result-attempt-{attempt:02d}.Rdata"
        elite_path = work / f"elite-attempt-{attempt:02d}.csv"
        program_path = work / f"run-irace-attempt-{attempt:02d}.R"
        recovery_path = None
        if (
            result_path.is_file()
            and result_path.stat().st_size > 0
            and not elite_path.exists()
        ):
            # irace rejects a scenario when recoveryFile and logFile name the
            # same path.  Preserve the interrupted state under a separate
            # snapshot name and let irace write the resumed state back to the
            # canonical result path for this attempt.
            recovery_path = work / f"irace-recovery-attempt-{attempt:02d}.Rdata"
            shutil.copy2(result_path, recovery_path)
        recovery_statement = (
            ""
            if recovery_path is None
            else "scenario$recoveryFile <- " + json.dumps(str(recovery_path.resolve()))
        )
        before = int(summary["different_candidates"])
        budget_statement = (
            f"scenario$maxTime <- {int(request['total_fe_cap'])}\n"
            "scenario$minMeasurableTime <- 1"
            if native_instances
            else f"scenario$maxExperiments <- {maximum_calls}"
        )
        race_shape_statement = (
            ""
            if native_instances
            else (
                f"scenario$nbIterations <- {iterations}\n"
                f"scenario$minNbSurvival <- {minimum_survivors}\n"
                f"scenario$nbConfigurations <- {configurations_per_iteration}"
            )
        )
        r_program = f"""
suppressPackageStartupMessages(library(irace))
scenario <- defaultScenario()
scenario$targetRunner <- {json.dumps(str((work / "target-runner").resolve()))}
scenario$parameterFile <- {json.dumps(str((work / "parameters.txt").resolve()))}
scenario$trainInstancesFile <- {json.dumps(str((work / "instances.txt").resolve()))}
scenario$execDir <- {json.dumps(str(work.resolve()))}
scenario$logFile <- {json.dumps(str(result_path.resolve()))}
{recovery_statement}
{budget_statement}
scenario$parallel <- {max(1, workers)}
scenario$seed <- {attempt_seed}
scenario$deterministic <- TRUE
{race_shape_statement}
# The durable ledger caches one fixed-seed observation per configuration/task.
# Default post-selection resamples the same finite deterministic task set.
# Those cached calls add neither FE nor information; the official racing elite
# is already produced by irace's tuning process.
scenario$postselection <- 0
elites <- irace(scenario=scenario)
write.csv(as.data.frame(elites[1, , drop=FALSE]), {json.dumps(str(elite_path.resolve()))}, row.names=FALSE)
"""
        program_path.write_text(r_program, encoding="utf-8")
        subprocess.run([rscript, str(program_path)], cwd=work, check=True)
        with elite_path.open(newline="", encoding="utf-8") as stream:
            elite = next(csv.DictReader(stream))
        attempt_configuration = _normalize_irace_row(elite, str(request["suite"]))
        elite_candidates.append((attempt, elite_path, attempt_configuration))
        summary = design_ledger.summary()
        attempts.append(
            {
                "attempt": attempt,
                "seed": attempt_seed,
                "candidates_before": before,
                "candidates_after": int(summary["different_candidates"]),
                "raw_result": str(result_path.resolve()),
                "elite": str(elite_path.resolve()),
                "recovered": recovery_path is not None,
            }
        )

    completion_errors = ledger_completion_errors(summary, request)
    if completion_errors:
        raise RuntimeError(
            "irace terminated with an incomplete durable design ledger after "
            f"{len(attempts)} new attempts: {'; '.join(completion_errors)}"
        )

    # Under the legacy aggregate contract, retries fill an exact distinct-
    # candidate budget and the best fully comparable elite can be selected by
    # its aggregate ledger cost. Under native racing, per-candidate observations
    # intentionally differ, so preserve irace's own final official elite rather
    # than comparing unequal subsets of instances ourselves.
    admitted = design_ledger.configurations()
    common_task_count = 0
    if native_instances:
        official_elites = [
            item
            for item in elite_candidates
            if configuration_key(str(request["suite"]), item[2]) in admitted
        ]
        if official_elites:
            official_keys = [
                configuration_key(str(request["suite"]), item[2])
                for item in official_elites
            ]
            comparable = design_ledger.best_on_common_tasks(official_keys)
            if comparable is not None:
                selected_key, _, common_task_count = comparable
                selected_attempt, selected_elite_path, configuration = max(
                    (
                        item
                        for item in official_elites
                        if configuration_key(str(request["suite"]), item[2])
                        == selected_key
                    ),
                    key=lambda item: item[0],
                )
                selection = "official_irace_elites_compared_on_common_tasks"
            else:
                selected_attempt, selected_elite_path, configuration = max(
                    official_elites, key=lambda item: item[0]
                )
                selection = "final_official_irace_native_racing_elite"
        else:
            configuration, _ = design_ledger.best_observed()
            selected_attempt = -1
            selected_elite_path = None
            selection = "ledger_recovery_no_persisted_native_elite"
    else:
        scored_elites: list[tuple[float, int, dict[str, float | int], Path]] = []
        for attempt, elite_path, candidate in elite_candidates:
            key = configuration_key(str(request["suite"]), candidate)
            if key not in admitted:
                continue
            cost = design_ledger.observed_cost(key)
            if cost is not None and math.isfinite(cost):
                scored_elites.append((cost, attempt, candidate, elite_path))
        if scored_elites:
            _, selected_attempt, configuration, selected_elite_path = min(
                scored_elites, key=lambda item: (item[0], item[1])
            )
            selection = "best_official_irace_elite_across_attempts"
        else:
            # Recovery from an interruption after all evaluations committed but
            # before irace wrote its elite CSV.
            configuration, _ = design_ledger.best_observed()
            selected_attempt = -1
            selected_elite_path = None
            selection = "ledger_best_observed_recovery"
    if not configuration_is_valid(
        str(request["suite"]),
        configuration,
        population_size_max=int(request.get("population_size_max", 200)),
        offspring_size_max=int(request.get("offspring_size_max", 200)),
    ):
        raise RuntimeError("irace selected a configuration outside eoFastGA's space.")
    artifact = {
        "schema": "autooptlib.external-design-artifact",
        "schema_version": 1,
        "method": "paradiseo_irace",
        "suite": request["suite"],
        "function_id": request["function_id"],
        "protocol_fingerprint": request["protocol_fingerprint"],
        "template_space_fingerprint": request["template_space_fingerprint"],
        "candidates_evaluated": summary["different_candidates"],
        "configuration": configuration,
        "design_ledger": summary,
        "provenance": {
            "configurator": "official irace R package",
            "evaluation_policy": request["evaluation_policy"],
            "native_budget": (
                {
                    "unit": "objective_evaluations",
                    "max_time": int(request["total_fe_cap"]),
                }
                if native_instances
                else {"unit": "target_calls", "max_experiments": maximum_calls}
            ),
            "max_experiments": 0 if native_instances else maximum_calls,
            "deterministic_common_random_numbers": True,
            "training_instance_order": (
                "dimension_round_robin" if native_instances else "single_aggregate"
            ),
            "iterations": iterations,
            "configurations_per_iteration": configurations_per_iteration,
            "candidate_proposal_budget": proposal_budget,
            "candidate_proposal_headroom": IRACE_PROPOSAL_HEADROOM,
            "postselection": 0,
            "max_attempts": IRACE_MAX_ATTEMPTS,
            "seed_stride": IRACE_SEED_STRIDE,
            "attempts": attempts,
            "selection": selection,
            "selection_common_tasks": common_task_count,
            "selected_attempt": selected_attempt,
            "selected_elite": (
                None
                if selected_elite_path is None
                else str(selected_elite_path.resolve())
            ),
            "rscript": shutil.which(rscript) or os.path.realpath(rscript),
            "raw_result": attempts[-1]["raw_result"] if attempts else None,
        },
    }
    atomic_json(output, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    work = args.work or args.output.with_suffix(".irace-work")
    print(
        json.dumps(
            run_irace(
                args.request,
                args.output,
                args.runner,
                work,
                rscript=args.rscript,
                workers=args.workers,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
