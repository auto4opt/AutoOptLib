"""Run official Sparkle with SMAC3 over the common eoFastGA template."""

from __future__ import annotations

import argparse
import ast
import configparser
import json
import math
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from comparisons.shared.platform_target import (
    PARAMETERS,
    DesignLedger,
    configuration_is_valid,
    configuration_key,
    ledger_completion_errors,
    ledger_is_complete,
    normalize_configuration,
)
from comparisons.shared.store import atomic_json

# Candidate count and the durable FE reservation are the registered termination
# rules. Keep SMAC's independent safety limit effectively disabled so it cannot
# silently become the active experimental budget on a slow machine.
SMAC3_CPU_TIME_LIMIT_SECONDS = 2_147_483_647
SPARKLE_MAX_ATTEMPTS = 4
SPARKLE_SEED_STRIDE = 1_000_003
SPARKLE_DRIVER_SCHEMA_VERSION = 7
SMAC3_FACADE = "AlgorithmConfigurationFacade"


def _sparkle_workspace_identity(request: dict[str, Any]) -> dict[str, Any]:
    return {
        **DesignLedger.request_identity(request),
        "sparkle_driver_schema_version": SPARKLE_DRIVER_SCHEMA_VERSION,
    }


def _load_sparkle_attempts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    attempts = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(attempts, list) or not all(
        isinstance(item, dict) for item in attempts
    ):
        raise RuntimeError("Sparkle attempt log is malformed.")
    return attempts


def _sparkle_attempt_seed(base_seed: int, attempt: int) -> int:
    """Derive the official Sparkle/SMAC seed for an independent attempt."""

    seed = (int(base_seed) + int(attempt) * SPARKLE_SEED_STRIDE) % (2**32)
    # Sparkle 0.9.6 uses ``latest_ini.seed or ...``. Zero would therefore be
    # treated as an absent seed rather than as a reproducible seed.
    return seed or 1


def _ordered_training_tasks(request: dict[str, Any]) -> list[dict[str, Any]]:
    """Interleave dimensions while preserving every registered CRN task."""

    tasks = list(request["training_tasks"])
    if not bool(request.get("promotion_requires_all_dimensions", False)):
        return tasks
    dimensions = list(dict.fromkeys(int(task["dimension"]) for task in tasks))
    grouped = {
        dimension: [task for task in tasks if int(task["dimension"]) == dimension]
        for dimension in dimensions
    }
    ordered = [
        grouped[dimension][index]
        for index in range(max(map(len, grouped.values())))
        for dimension in dimensions
        if index < len(grouped[dimension])
    ]
    if len(ordered) != len(tasks):
        raise RuntimeError("Sparkle task interleaving lost a registered instance.")
    return ordered


def _write_smac3_settings(path: Path, *, seed: int) -> Path:
    """Write the official Sparkle settings override used by the paper protocol."""

    path.write_text(
        "[general]\n"
        f"seed = {int(seed)}\n\n"
        "[smac3]\n"
        f"cputime_limit = {SMAC3_CPU_TIME_LIMIT_SECONDS}\n"
        f"facade = {SMAC3_FACADE}\n",
        encoding="utf-8",
    )
    return path


def _write_sparkle_latest_seed(path: Path, seed: int) -> Path:
    """Set the seed Sparkle 0.9.6 actually gives highest precedence.

    Sparkle reads ``Settings/latest.ini`` before the explicit settings file
    when it initialises its global random state. Preserve all existing options
    in that file and replace only the general seed.
    """

    parser = configparser.ConfigParser(interpolation=None)
    if path.exists():
        parser.read(path, encoding="utf-8")
    if not parser.has_section("general"):
        parser.add_section("general")
    parser.set("general", "seed", str(int(seed)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        parser.write(stream)
    return path


def _wait_for_unused_scenario_minute(work: Path) -> None:
    """Avoid Sparkle deleting a previous attempt with the same minute stamp."""

    configuration_root = work / "Output" / "Configuration"
    while configuration_root.exists():
        suffix = datetime.now().strftime("%Y%m%d-%H%M")
        if not any(
            path.is_dir() and path.name.endswith(suffix)
            for path in configuration_root.rglob("*")
        ):
            return
        time.sleep(1.0)


def _write_quiet_sitecustomize(directory: Path) -> Path:
    """Install process-local Sparkle/SMAC3/RunRunner compatibility safeguards.

    Sparkle 0.9.6 uses RunRunner's local backend both for the configurator and
    for the dependent validation run.  RunRunner can miss the dependency-
    completion wake-up when the dependency finishes between the initial state
    check and insertion into its wait list.  Its ``wait`` method then sleeps
    forever although every worker is idle.  Polling ``run_all`` while holding
    RunRunner's own queue lock closes that race without changing any command,
    configuration, seed, or result.

    Sparkle 0.9.6 also replaces SMAC3's optimiser after the facade has already
    registered its callbacks.  The replacement silently drops the intensifier
    callback, so no incumbent is ever created and every configuration is run on
    every instance.  Restore the callbacks immediately before optimisation.
    This preserves the official SMAC3 intensifier and only repairs Sparkle's
    optimiser replacement.
    """

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sitecustomize.py"
    path.write_text(
        "import logging\n"
        "import os\n"
        "import time\n\n"
        "if os.environ.get('AUTOOPTLIB_SPARKLE_DISABLE_CONSOLE_LOGGING') == '1':\n"
        "    logging.disable(logging.CRITICAL)\n\n"
        "if os.environ.get('AUTOOPTLIB_SPARKLE_RUNRUNNER_COMPAT') == '1':\n"
        "    from runrunner.local import LocalRun\n\n"
        "    _autooptlib_original_wait = LocalRun.wait\n\n"
        "    def _autooptlib_race_safe_wait(self, timeout=None):\n"
        "        started = time.monotonic()\n"
        "        while self._futures is None:\n"
        "            with LocalRun._queue_lock:\n"
        "                if self._futures is None:\n"
        "                    self.run_all()\n"
        "            if self._futures is None:\n"
        "                if timeout is not None and time.monotonic() - started >= timeout:\n"
        "                    raise TimeoutError('RunRunner dependency wait timed out')\n"
        "                time.sleep(0.1)\n"
        "        remaining = None\n"
        "        if timeout is not None:\n"
        "            remaining = max(0.0, timeout - (time.monotonic() - started))\n"
        "        return _autooptlib_original_wait(self, timeout=remaining)\n\n"
        "    LocalRun.wait = _autooptlib_race_safe_wait\n"
        "\n"
        "if os.environ.get('AUTOOPTLIB_SPARKLE_SMAC_CALLBACK_COMPAT') == '1':\n"
        "    from smac.facade.abstract_facade import AbstractFacade\n\n"
        "    _autooptlib_original_optimize = AbstractFacade.optimize\n\n"
        "    def _autooptlib_restore_smac_callbacks(self, *args, **kwargs):\n"
        "        registered = self._optimizer._callbacks\n"
        "        for callback in self._callbacks:\n"
        "            if not any(existing is callback for existing in registered):\n"
        "                self._optimizer.register_callback(callback)\n"
        "        if not any(\n"
        "            getattr(existing, 'intensifier', None) is self._intensifier\n"
        "            for existing in registered\n"
        "        ):\n"
        "            self._optimizer.register_callback(\n"
        "                self._intensifier.get_callback(), index=0\n"
        "            )\n"
        "        if not any(\n"
        "            getattr(existing, 'intensifier', None) is self._intensifier\n"
        "            for existing in self._optimizer._callbacks\n"
        "        ):\n"
        "            raise RuntimeError('SMAC3 intensifier callback was not restored')\n"
        "        return _autooptlib_original_optimize(self, *args, **kwargs)\n\n"
        "    AbstractFacade.optimize = _autooptlib_restore_smac_callbacks\n",
        encoding="utf-8",
    )
    return path


def _pcs(
    suite: str,
    *,
    population_size_max: int = 200,
    offspring_size_max: int = 200,
) -> str:
    mutation_max = 3 if suite == "bbob" else 8
    selector_choices = "0, 1, 2, 4, 5" if suite == "bbob" else "0, 1, 2, 3, 4"
    lines = [
        "crossover_rate real [0.0, 1.0] [0.8]",
        f"crossover_selector categorical {{{selector_choices}}} [0]",
        "crossover categorical {0, 1, 2, 3} [0]",
        f"aftercross_selector categorical {{{selector_choices}}} [0]",
        "mutation_rate real [0.0, 1.0] [0.8]",
        f"mutation_selector categorical {{{selector_choices}}} [0]",
        "mutation categorical {"
        + ", ".join(map(str, range(mutation_max + 1)))
        + "} [0]",
        "replacement categorical {0, 1, 2, 3} [0]",
        f"population_size integer [4, {int(population_size_max)}] [20]",
        f"offspring_size integer [1, {int(offspring_size_max)}] [20]",
    ]
    if suite == "bbob":
        lines.extend(
            [
                "boundary_handling categorical {0, 1, 2} [0]",
                "elite_fraction real [0.0, 1.0] [0.2]",
                "de_f real [0.0, 1.0] [0.5]",
                "de_cr real [0.0, 1.0] [0.5]",
                "de_p real [0.0, 1.0] [0.2]",
                (
                    "elite_fraction | crossover_selector in {5} || "
                    "aftercross_selector in {5} || mutation_selector in {5}"
                ),
                "de_f | mutation in {3}",
                "de_cr | mutation in {3}",
                "de_p | mutation in {3}",
            ]
        )
    lines.extend(
        [
            "{offspring_size > population_size && replacement == 1}",
            "{offspring_size > population_size && replacement == 2}",
            "{offspring_size > population_size && replacement == 3}",
            "",
        ]
    )
    return "\n".join(lines)


def _solver_wrapper(request: Path, ledger: Path, runner: Path, repository: Path) -> str:
    return f"""#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path

from sparkle.tools.solver_wrapper_parsing import parse_solver_wrapper_args

args = parse_solver_wrapper_args(sys.argv[1:])
parameter_names = {list(PARAMETERS)!r}
configuration = {{name: args[name] for name in parameter_names if name in args}}
command = [
    sys.executable, "-m", "comparisons.shared.platform_target",
    "--request", {str(request.resolve())!r},
    "--ledger", {str(ledger.resolve())!r},
    "--runner", {str(runner.resolve())!r},
    "--instance", str(Path(args["instance"]).resolve()),
    "--config-json", json.dumps(configuration, sort_keys=True),
]
environment = dict(os.environ)
environment["PYTHONPATH"] = os.pathsep.join([
    {str(repository.resolve())!r},
    {str((repository / "methods").resolve())!r},
    environment.get("PYTHONPATH", ""),
])
completed = subprocess.run(command, env=environment, text=True, capture_output=True)
cost = 1e30
if completed.returncode == 0:
    try:
        cost = float(completed.stdout.strip().splitlines()[-1])
    except Exception:
        pass
print({{"status": "SUCCESS" if cost < 1e30 else "CRASHED", "quality": cost, "solver_call": command}})
"""


def _run(
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    *,
    input_text: str | None = None,
) -> None:
    subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=True,
        input=input_text,
        text=input_text is not None,
    )


def _budget_is_complete(ledger: DesignLedger, request: dict[str, Any]) -> bool:
    """Return whether the durable ledger reached its registered stop rule."""

    ledger.reclaim_stale_claims()
    return ledger_is_complete(ledger.summary(), request)


def _stop_process_group(process: subprocess.Popen[Any]) -> None:
    """Stop Sparkle after its registered external-design budget is complete."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)


def _run_until_candidate_budget(
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    *,
    ledger_path: Path,
    request: dict[str, Any],
) -> bool:
    """Run official Sparkle and stop only after its durable budget is complete.

    Returns ``True`` when AutoOptLib stopped otherwise-superfluous solver calls;
    returns ``False`` when Sparkle exited normally first.

    The historical helper name is retained for checkpoint/test compatibility.
    In native mode the stop rule is total objective FE, not candidate count.
    """

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        start_new_session=True,
    )
    ledger = DesignLedger(ledger_path)
    native_instances = str(request["evaluation_policy"]) == "native_instances"
    if native_instances:
        while process.poll() is None:
            if _budget_is_complete(ledger, request):
                # The target commits configurations, costs and FE atomically to
                # the durable ledger before returning.  That ledger is the budget
                # authority and its completion rule also requires zero in-flight
                # claims/reservations.  Do not wait for SMAC JSON: after a resumed
                # or interrupted run, an older runhistory can be partial forever.
                # Finalization prefers an official incumbent when available and
                # otherwise selects from the complete ledger observations.
                _stop_process_group(process)
                return True
            time.sleep(1.0)
        if process.returncode != 0 and not _budget_is_complete(ledger, request):
            raise subprocess.CalledProcessError(process.returncode, command)
        return False
    while process.poll() is None:
        if _budget_is_complete(ledger, request):
            admitted = ledger.configurations()
            # Stop only after SMAC has durably persisted every completed
            # candidate. This replaces the old two-second timing guess, which
            # could interrupt the final result between wrapper return and
            # runhistory persistence.
            if _runhistory_completed_candidates(
                cwd, admitted, str(request["suite"])
            ) >= int(request["candidate_budget"]):
                _stop_process_group(process)
                return True
        time.sleep(1.0)
    if process.returncode != 0:
        if _budget_is_complete(ledger, request):
            admitted = ledger.configurations()
            if _runhistory_completed_candidates(
                cwd, admitted, str(request["suite"])
            ) >= int(request["candidate_budget"]):
                return False
        raise subprocess.CalledProcessError(process.returncode, command)
    return False


def _status_is_success(status: Any) -> bool:
    if status is None:
        return True
    if isinstance(status, (int, float)):
        return int(status) == 1
    return str(status).upper() in {"1", "SUCCESS", "STATUSTYPE.SUCCESS"}


def _runhistory_scores(
    work: Path, admitted: dict[str, dict[str, float | int]], suite: str
) -> tuple[dict[str, list[float]], dict[str, Path]]:
    scores: dict[str, list[float]] = {}
    sources: dict[str, Path] = {}
    histories = sorted(work.rglob("runhistory.json"))
    for path in histories:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # SMAC writes this file during optimization; an interrupted partial
            # file is ignored when another complete history is available.
            continue
        configs = {str(key): value for key, value in payload.get("configs", {}).items()}
        for entry in payload.get("data", []):
            try:
                if isinstance(entry, dict):
                    config_id = str(entry["config_id"])
                    cost = float(entry["cost"])
                    status = entry.get("status")
                else:
                    # Legacy SMAC runhistory JSON: [key, value].
                    config_id = str(entry[0][0])
                    raw_value = entry[1]
                    cost = float(
                        raw_value[0] if isinstance(raw_value, list) else raw_value
                    )
                    status = (
                        raw_value[2]
                        if isinstance(raw_value, list) and len(raw_value) > 2
                        else None
                    )
                configuration = normalize_configuration(configs[config_id], suite)
            except (KeyError, IndexError, TypeError, ValueError, OverflowError):
                continue
            key = configuration_key(suite, configuration)
            if (
                key in admitted
                and math.isfinite(cost)
                and cost < 1e30
                and _status_is_success(status)
            ):
                scores.setdefault(key, []).append(cost)
                sources[key] = path
    return scores, sources


def _runhistory_observations(
    work: Path, admitted: dict[str, dict[str, float | int]], suite: str
) -> set[tuple[str, str]]:
    """Return distinct successful (configuration, native-instance) records."""

    observations: set[tuple[str, str]] = set()
    for path in sorted(work.rglob("runhistory.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        configs = {str(key): value for key, value in payload.get("configs", {}).items()}
        for index, entry in enumerate(payload.get("data", [])):
            try:
                if isinstance(entry, dict):
                    config_id = str(entry["config_id"])
                    cost = float(entry["cost"])
                    status = entry.get("status")
                    instance = entry.get("instance", entry.get("instance_id"))
                else:
                    key_data = entry[0]
                    config_id = str(key_data[0])
                    instance = key_data[1] if len(key_data) > 1 else None
                    raw_value = entry[1]
                    cost = float(
                        raw_value[0] if isinstance(raw_value, list) else raw_value
                    )
                    status = (
                        raw_value[2]
                        if isinstance(raw_value, list) and len(raw_value) > 2
                        else None
                    )
                configuration = normalize_configuration(configs[config_id], suite)
            except (KeyError, IndexError, TypeError, ValueError, OverflowError):
                continue
            candidate_key = configuration_key(suite, configuration)
            if (
                candidate_key in admitted
                and math.isfinite(cost)
                and cost < 1e30
                and _status_is_success(status)
            ):
                instance_key = (
                    str(instance)
                    if instance is not None
                    else f"unknown:{path.resolve()}:{index}"
                )
                observations.add((candidate_key, instance_key))
    return observations


def _runhistory_completed_candidates(
    work: Path, admitted: dict[str, dict[str, float | int]], suite: str
) -> int:
    scores, _ = _runhistory_scores(work, admitted, suite)
    return len(scores)


def _best_from_runhistory(
    work: Path, admitted: dict[str, dict[str, float | int]], suite: str
) -> tuple[dict[str, float | int], Path]:
    histories = list(work.rglob("runhistory.json"))
    if not histories:
        raise FileNotFoundError("Sparkle completed without a SMAC3 runhistory.json")
    scores, sources = _runhistory_scores(work, admitted, suite)
    if not scores:
        raise RuntimeError("No admitted successful configuration exists in runhistory.")
    best_key = min(scores, key=lambda key: (sum(scores[key]) / len(scores[key]), key))
    return admitted[best_key], sources[best_key]


def _official_smac_incumbent(
    work: Path, admitted: dict[str, dict[str, float | int]], suite: str
) -> tuple[dict[str, float | int], Path] | None:
    """Read official incumbents and compare retries only on common tasks."""

    paths = sorted(
        work.rglob("intensifier.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )
    official: dict[str, tuple[dict[str, float | int], Path]] = {}
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            runhistory = path.with_name("runhistory.json")
            history = json.loads(runhistory.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        incumbent_ids = list(payload.get("incumbent_ids", []))
        if not incumbent_ids:
            trajectory = payload.get("trajectory", [])
            if trajectory and isinstance(trajectory[-1], dict):
                latest = trajectory[-1]
                incumbent_ids = list(
                    latest.get("config_ids", latest.get("incumbent_ids", []))
                )
        configurations = {
            str(key): value for key, value in history.get("configs", {}).items()
        }
        candidates: list[dict[str, float | int]] = []
        for identifier in incumbent_ids:
            raw = configurations.get(str(identifier))
            if raw is None:
                continue
            candidate = normalize_configuration(raw, suite)
            if configuration_key(suite, candidate) in admitted:
                candidates.append(candidate)
        for candidate in candidates:
            key = configuration_key(suite, candidate)
            official.setdefault(key, (candidate, runhistory))
    if not official:
        return None
    if len(official) == 1:
        return next(iter(official.values()))
    ledger = DesignLedger(work / "design-ledger.sqlite3")
    comparable = ledger.best_on_common_tasks(list(official))
    if comparable is not None:
        return official[comparable[0]]
    # With no common observation, retain the newest official incumbent instead
    # of comparing means over unequal native-instance subsets.
    newest_source = paths[0].with_name("runhistory.json")
    newest = [item for item in official.values() if item[1] == newest_source]
    return min(
        newest or list(official.values()),
        key=lambda item: configuration_key(suite, item[0]),
    )


def _extract_sparkle_configuration(
    work: Path,
) -> dict[str, Any] | None:
    """Best-effort read of the incumbent Sparkle wrote to its performance CSV."""

    performance = work / "Output" / "Performance_Data" / "performance_data.csv"
    if not performance.exists():
        return None
    for line in reversed(performance.read_text(encoding="utf-8").splitlines()):
        if not line.startswith("$"):
            continue
        # Sparkle appends `$solver,configuration_id,{python dictionary}` without
        # CSV quoting. Split only the first two commas so commas inside the
        # dictionary remain intact.
        fields = line.split(",", 2)
        if len(fields) != 3:
            continue
        try:
            value = ast.literal_eval(fields[2].strip())
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict) and "configuration_id" in value:
            return value
    return None


def _finalize_sparkle_artifact(
    request: dict[str, Any],
    output: Path,
    work: Path,
    ledger_path: Path,
    *,
    attempts: list[dict[str, Any]],
    budget_terminated: bool,
) -> dict[str, Any]:
    """Build the immutable artifact from the durable ledger and runhistory."""

    ledger = DesignLedger(ledger_path)
    ledger.bind_request(request)
    ledger.reclaim_stale_claims()
    summary = ledger.summary()
    completion_errors = ledger_completion_errors(summary, request)
    if completion_errors:
        raise RuntimeError(
            "Sparkle/SMAC3 ledger is incomplete: " + "; ".join(completion_errors)
        )
    admitted = ledger.configurations()
    native_instances = str(request["evaluation_policy"]) == "native_instances"
    official = (
        _official_smac_incumbent(work, admitted, str(request["suite"]))
        if native_instances
        else None
    )
    if official is not None:
        configuration, runhistory = official
        selection = "official_smac_native_racing_incumbent"
    else:
        comparable = (
            ledger.best_on_common_tasks(list(admitted)) if native_instances else None
        )
        if comparable is not None:
            configuration = admitted[comparable[0]]
            _, sources = _runhistory_scores(work, admitted, str(request["suite"]))
            runhistory = sources.get(comparable[0])
            if runhistory is not None:
                selection = "common_task_recovery_no_persisted_native_incumbent"
            else:
                histories = sorted(work.rglob("runhistory.json"))
                if not histories:
                    raise FileNotFoundError(
                        "Sparkle completed without a SMAC3 runhistory.json"
                    )
                # The durable ledger is the authoritative result store.  A
                # scheduler signal can interrupt SMAC while it rewrites JSON,
                # but must not change the winner already established on the
                # common native-instance tasks.
                runhistory = histories[-1]
                selection = "common_task_ledger_recovery_after_partial_runhistory"
        else:
            configuration, runhistory = _best_from_runhistory(
                work, admitted, str(request["suite"])
            )
            selection = (
                "runhistory_recovery_no_persisted_native_incumbent"
                if native_instances
                else "best_aggregate_runhistory_configuration"
            )
    if not native_instances:
        sparkle_incumbent = _extract_sparkle_configuration(work)
        if sparkle_incumbent is not None:
            candidate = normalize_configuration(
                sparkle_incumbent, str(request["suite"])
            )
            candidate_key = configuration_key(str(request["suite"]), candidate)
            best_key = configuration_key(str(request["suite"]), configuration)
            candidate_cost = ledger.observed_cost(candidate_key)
            best_cost = ledger.observed_cost(best_key)
            if (
                candidate_key in admitted
                and candidate_cost is not None
                and best_cost is not None
                and candidate_cost <= best_cost + 1e-15
            ):
                configuration = candidate
    if not configuration_is_valid(
        str(request["suite"]),
        configuration,
        population_size_max=int(request.get("population_size_max", 200)),
        offspring_size_max=int(request.get("offspring_size_max", 200)),
    ):
        raise RuntimeError("Sparkle selected a configuration outside eoFastGA's space.")
    artifact = {
        "schema": "autooptlib.external-design-artifact",
        "schema_version": 1,
        "method": "sparkle_smac3",
        "suite": request["suite"],
        "function_id": request["function_id"],
        "protocol_fingerprint": request["protocol_fingerprint"],
        "template_space_fingerprint": request["template_space_fingerprint"],
        "candidates_evaluated": summary["different_candidates"],
        "configuration": configuration,
        "design_ledger": summary,
        "provenance": {
            "configurator": "official Sparkle SMAC3 integration",
            "smac_facade": SMAC3_FACADE,
            "solver_deterministic": True,
            "evaluation_policy": request["evaluation_policy"],
            "training_instance_order": (
                "dimension_round_robin"
                if request.get("promotion_requires_all_dimensions", False)
                else "protocol_order"
            ),
            "selection": selection,
            "smac3_cputime_limit_seconds": SMAC3_CPU_TIME_LIMIT_SECONDS,
            "console_logging": "disabled to avoid undrained RunRunner PIPE backpressure",
            "runrunner_compatibility": (
                "race-safe local dependency wait; no search semantics changed"
            ),
            "smac_callback_compatibility": (
                "restored callbacks dropped by Sparkle 0.9.6 after replacing "
                "SMAC3's optimizer; official intensifier semantics preserved"
            ),
            "candidate_budget_termination": (
                (
                    "stopped after the durable target ledger exhausted the native "
                    "total-FE budget with no in-flight evaluations"
                    if budget_terminated
                    else "SMAC exited normally after exhausting the native total-FE "
                    "budget"
                )
                if native_instances
                else (
                    "stopped after durable registered candidate-budget rule; "
                    "remaining solver calls could not admit candidates or consume FE"
                    if budget_terminated
                    else "Sparkle exited normally at the exact budget"
                )
            ),
            "max_attempts_per_invocation": SPARKLE_MAX_ATTEMPTS,
            "seed_stride": SPARKLE_SEED_STRIDE,
            "attempts": attempts,
            "runhistory": str(runhistory.resolve()),
        },
    }
    atomic_json(output, artifact)
    return artifact


def run_sparkle(
    request_path: Path,
    output: Path,
    runner: Path,
    work: Path,
    *,
    sparkle: str,
    workers: int,
) -> dict[str, Any]:
    if workers != 1:
        raise ValueError(
            "The official Sparkle configure_solver CLI does not expose SMAC3's "
            "internal n_workers setting. Use --workers 1 and parallelize independent "
            "function-level design jobs instead."
        )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    repository = Path(__file__).resolve().parents[2]
    work.mkdir(parents=True, exist_ok=True)
    package = work / "source" / "AutoOptLib-eoFastGA"
    instances = work / "source" / "training-instances"
    package.mkdir(parents=True, exist_ok=True)
    instances.mkdir(parents=True, exist_ok=True)
    runtime_site = work / "source" / "runtime-site"
    _write_quiet_sitecustomize(runtime_site)
    ledger_path = work / "design-ledger.sqlite3"
    ledger = DesignLedger(ledger_path)
    ledger.bind_request(request)
    workspace_identity_path = work / "source" / "autooptlib-request-identity.json"
    workspace_identity = _sparkle_workspace_identity(request)
    marker = work / "Settings" / "sparkle_settings.ini"
    if workspace_identity_path.exists():
        recorded_identity = json.loads(
            workspace_identity_path.read_text(encoding="utf-8")
        )
        if recorded_identity != workspace_identity:
            raise RuntimeError(
                "Sparkle workspace belongs to a different immutable design request. "
                "Use a fresh work directory."
            )
    elif marker.exists():
        raise RuntimeError(
            "Refusing to reuse a legacy Sparkle workspace without request identity. "
            "Use a fresh work directory."
        )
    else:
        workspace_identity_path.write_text(
            json.dumps(workspace_identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (package / "configspace.pcs").write_text(
        _pcs(
            str(request["suite"]),
            population_size_max=int(request.get("population_size_max", 200)),
            offspring_size_max=int(request.get("offspring_size_max", 200)),
        ),
        encoding="utf-8",
    )
    wrapper = package / "sparkle_solver_wrapper.py"
    wrapper.write_text(
        _solver_wrapper(request_path, ledger_path, runner, repository),
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    registered_tasks = (
        _ordered_training_tasks(request)
        if str(request["evaluation_policy"]) == "native_instances"
        else [{"aggregate": True}]
    )
    for index, task in enumerate(registered_tasks):
        (instances / f"task-{index:03d}.json").write_text(
            json.dumps(task, sort_keys=True) + "\n", encoding="utf-8"
        )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(runtime_site.resolve()),
            str(repository.resolve()),
            str((repository / "methods").resolve()),
            environment.get("PYTHONPATH", ""),
        ]
    )
    environment["AUTOOPTLIB_SPARKLE_DISABLE_CONSOLE_LOGGING"] = "1"
    environment["AUTOOPTLIB_SPARKLE_RUNRUNNER_COMPAT"] = "1"
    environment["AUTOOPTLIB_SPARKLE_SMAC_CALLBACK_COMPAT"] = "1"
    if not marker.exists():
        # Sparkle 0.9.6 always runs initialisation interactively and asks whether
        # missing optional configurators should be installed.  The paper runner
        # is deliberately non-interactive and uses the already pinned SMAC3
        # environment, so decline every optional installer prompt explicitly.
        _run(
            [sparkle, "initialise"],
            work,
            environment,
            input_text="n\nn\nn\n",
        )
        _run(
            [
                sparkle,
                "add_solver",
                str(package.resolve()),
                "--deterministic",
                "--skip-checks",
            ],
            work,
            environment,
        )
        _run(
            [sparkle, "add_instances", str(instances.resolve())],
            work,
            environment,
        )
    native_instances = str(request["evaluation_policy"]) == "native_instances"
    maximum_calls = math.ceil(
        int(request["total_fe_cap"])
        / int(
            request["minimum_task_fe" if native_instances else "candidate_run_budget"]
        )
    )
    attempts_path = work / "source" / "sparkle-attempts.json"
    attempts = _load_sparkle_attempts(attempts_path)
    used_attempts = {
        int(item["attempt"])
        for item in attempts
        if isinstance(item.get("attempt"), int)
    }
    for settings_path in (work / "source").glob("paper-settings-attempt-*.ini"):
        try:
            used_attempts.add(int(settings_path.stem.rsplit("-", 1)[-1]))
        except ValueError:
            continue
    first_attempt = max(used_attempts, default=-1) + 1
    budget_terminated = any(
        bool(item.get("budget_terminated", False)) for item in attempts
    )
    for attempt in range(first_attempt, first_attempt + SPARKLE_MAX_ATTEMPTS):
        if _budget_is_complete(ledger, request):
            break
        before_summary = ledger.summary()
        before = int(before_summary["different_candidates"])
        before_fes = int(before_summary["actual_design_fes"])
        attempt_seed = _sparkle_attempt_seed(int(request["seed"]), attempt)
        settings = _write_smac3_settings(
            work / "source" / f"paper-settings-attempt-{attempt:02d}.ini",
            seed=attempt_seed,
        )
        # Sparkle 0.9.6 prioritises Settings/latest.ini over --settings-file
        # when seeding Python and NumPy. Synchronising it is therefore required
        # for the explicit paper seed to reach SMAC's generated run seed.
        _write_sparkle_latest_seed(work / "Settings" / "latest.ini", attempt_seed)
        _wait_for_unused_scenario_minute(work)
        command = [
            sparkle,
            "configure_solver",
            "--configurator",
            "SMAC3",
            "--solver",
            package.name,
            "--instance-set-train",
            instances.name,
            "--objectives",
            "quality",
            "--solver-calls",
            str(maximum_calls),
            "--number-of-runs",
            "1",
            "--run-on",
            "local",
            "--settings-file",
            str(settings.resolve()),
        ]
        stopped = _run_until_candidate_budget(
            command,
            work,
            environment,
            ledger_path=ledger_path,
            request=request,
        )
        budget_terminated = budget_terminated or stopped
        after_summary = ledger.summary()
        after = int(after_summary["different_candidates"])
        after_fes = int(after_summary["actual_design_fes"])
        attempts.append(
            {
                "attempt": attempt,
                "seed": attempt_seed,
                "candidates_before": before,
                "candidates_after": after,
                "design_fes_before": before_fes,
                "design_fes_after": after_fes,
                "budget_terminated": stopped,
                "command": command,
            }
        )
        atomic_json(attempts_path, attempts)
        if _budget_is_complete(ledger, request):
            break
        if (after_fes <= before_fes) if native_instances else (after <= before):
            raise RuntimeError(
                "Sparkle exited before its registered budget and added no durable "
                "design work; refusing a pointless retry."
            )
    return _finalize_sparkle_artifact(
        request,
        output,
        work,
        ledger_path,
        attempts=attempts,
        budget_terminated=budget_terminated,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--sparkle", default="sparkle")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Finalize an already complete durable ledger without restarting SMAC3.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Must be 1: Sparkle's official configure_solver CLI does not expose "
            "SMAC3's internal worker count."
        ),
    )
    args = parser.parse_args(argv)
    work = args.work or args.output.with_suffix(".sparkle-work")
    if args.finalize_only:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        workspace_identity_path = work / "source" / "autooptlib-request-identity.json"
        if not workspace_identity_path.exists() or json.loads(
            workspace_identity_path.read_text(encoding="utf-8")
        ) != _sparkle_workspace_identity(request):
            raise RuntimeError(
                "Cannot finalize a Sparkle workspace without the current immutable "
                "driver/request identity. Restart it in a fresh work directory."
            )
        artifact = _finalize_sparkle_artifact(
            request,
            args.output,
            work,
            work / "design-ledger.sqlite3",
            attempts=_load_sparkle_attempts(work / "source" / "sparkle-attempts.json"),
            budget_terminated=True,
        )
        print(json.dumps(artifact, indent=2))
        return 0
    print(
        json.dumps(
            run_sparkle(
                args.request,
                args.output,
                args.runner,
                work,
                sparkle=args.sparkle,
                workers=args.workers,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
