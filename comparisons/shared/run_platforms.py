"""Create registered training requests and invoke official configurators."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from comparisons.shared.platform_target import (
    configuration_is_valid,
    ledger_completion_errors,
    template_space_fingerprint,
)
from comparisons.shared.protocol import ExperimentProtocol, load_protocol
from comparisons.shared.scoring import make_taskwise_scorer
from comparisons.shared.store import atomic_json

METHODS = ("paradiseo_irace", "sparkle_smac3")


def _evaluation_policy(protocol: ExperimentProtocol, method: str, suite: str) -> str:
    """Return the registered evaluator contract for one external configurator.

    AutoOptLib candidates are deliberately scored on the full training-task
    mean.  External configurators, however, have native instance schedulers
    (irace racing and SMAC intensification).  A method-specific override keeps
    those schedulers real instead of presenting the same aggregate value as if
    it were twelve different observations.
    """

    configured = protocol.design.get("external_evaluation_policy", {})
    policy = str(
        configured.get(
            method,
            protocol.suites[suite].get("design_evaluate", protocol.design["evaluate"]),
        )
    )
    if policy not in {"exact", "native_instances"}:
        raise ValueError(
            f"Unsupported external evaluation policy {policy!r} for {method}."
        )
    return policy


def _method_space(
    protocol: ExperimentProtocol, method: str, suite: str
) -> dict[str, int]:
    method_space = protocol.design.get("external_search_spaces", {}).get(method, {})
    if "population_size_max" in method_space or "offspring_size_max" in method_space:
        # Backward compatibility with flat method-level protocol snapshots.
        configured = method_space
    else:
        configured = method_space.get(suite, {})
    return {
        "population_size_max": int(configured.get("population_size_max", 200)),
        "offspring_size_max": int(configured.get("offspring_size_max", 200)),
    }


def validate_external_artifact(
    payload: dict[str, Any],
    protocol: ExperimentProtocol,
    method: str,
    suite: str,
    function_id: int,
) -> None:
    """Validate identity, executable configuration, and exact FE completion."""

    evaluation_policy = _evaluation_policy(protocol, method, suite)
    expected = {
        "schema": "autooptlib.external-design-artifact",
        "schema_version": 1,
        "method": method,
        "suite": suite,
        "function_id": int(function_id),
        "protocol_fingerprint": protocol.fingerprint(),
        "template_space_fingerprint": template_space_fingerprint(
            suite, **_method_space(protocol, method, suite)
        ),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(
                f"External artifact field {name!r} is {payload.get(name)!r}; "
                f"expected {value!r}."
            )
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict) or not configuration_is_valid(
        suite, configuration
    ):
        raise ValueError("External artifact has no valid eoFastGA configuration.")
    method_space = _method_space(protocol, method, suite)
    if int(configuration["population_size"]) > method_space["population_size_max"]:
        raise ValueError("External artifact exceeds the population-size search cap.")
    if int(configuration["offspring_size"]) > method_space["offspring_size_max"]:
        raise ValueError("External artifact exceeds the offspring-size search cap.")
    ledger = payload.get("design_ledger")
    if not isinstance(ledger, dict):
        raise ValueError("External artifact has no design FE ledger.")
    request_contract = {
        "candidate_budget": int(protocol.design["candidate_budget"]),
        "evaluation_policy": evaluation_policy,
        "total_fe_cap": protocol.design_fe_cap(suite),
        "minimum_task_fe": min(
            int(item.budget) for item in protocol.training_instances(suite, function_id)
        ),
    }
    errors = ledger_completion_errors(ledger, request_contract)
    if errors:
        raise ValueError("Incomplete external design ledger: " + "; ".join(errors))
    candidates_evaluated = int(payload.get("candidates_evaluated", -1))
    ledger_candidates = int(ledger.get("different_candidates", -1))
    if candidates_evaluated != ledger_candidates:
        raise ValueError(
            "External artifact candidate count does not match its design ledger."
        )
    if evaluation_policy == "exact" and candidates_evaluated != int(
        protocol.design["candidate_budget"]
    ):
        raise ValueError("Exact external design did not evaluate the candidate budget.")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("External artifact has no protocol provenance.")
    if evaluation_policy == "native_instances" and method == "sparkle_smac3":
        if provenance.get("smac_facade") != "AlgorithmConfigurationFacade":
            raise ValueError(
                "Sparkle artifact did not use AlgorithmConfigurationFacade."
            )
        if provenance.get("solver_deterministic") is not True:
            raise ValueError(
                "Sparkle artifact did not register a deterministic solver."
            )
    if evaluation_policy == "native_instances" and method == "paradiseo_irace":
        native_budget = provenance.get("native_budget")
        if not isinstance(native_budget, dict) or native_budget.get("unit") != (
            "objective_evaluations"
        ):
            raise ValueError("irace artifact did not use its native FE budget.")
        if provenance.get("deterministic_common_random_numbers") is not True:
            raise ValueError("irace artifact did not register deterministic CRN tasks.")


def _design_request(
    protocol: ExperimentProtocol,
    method: str,
    suite: str,
    function_id: int,
) -> dict[str, Any]:
    instances = protocol.training_instances(suite, function_id)
    scorer = make_taskwise_scorer(suite, function_id, instances)
    method_space = _method_space(protocol, method, suite)
    evaluation_policy = _evaluation_policy(protocol, method, suite)
    minimum_task_fe = min(int(instance.budget) for instance in instances)
    total_fe_cap = protocol.design_fe_cap(suite)
    return {
        "schema": "autooptlib.external-design-request",
        "schema_version": 1,
        "method": method,
        "suite": suite,
        "function_id": function_id,
        "direction": "min" if suite == "bbob" else "max",
        "training_tasks": [
            {
                **vars(instance),
                "seed": protocol.task_seed(
                    suite,
                    function_id,
                    instance.instance,
                    instance.repeat,
                    phase="training",
                    dimension=instance.dimension,
                ),
                "optimum_cost": (None if optimum_cost is None else float(optimum_cost)),
                "optimum_raw": (
                    None
                    if optimum_cost is None
                    else float(optimum_cost if suite == "bbob" else -optimum_cost)
                ),
            }
            for instance, optimum_cost in zip(instances, scorer.optimum_costs)
        ],
        "training_score": scorer.name,
        "candidate_budget": int(protocol.design["candidate_budget"]),
        "candidate_run_budget": int(protocol.design["candidate_run_budget"]),
        "training_runs": int(protocol.design["training_runs"]),
        "total_fe_cap": total_fe_cap,
        # This is a concurrency-safety ceiling, not an experimental budget.
        # Under native racing, total FE is the authoritative budget and at most
        # one previously unseen configuration can be admitted per minimum-cost
        # target call.
        "candidate_limit": (
            int(protocol.design["candidate_budget"])
            if evaluation_policy == "exact"
            else total_fe_cap // minimum_task_fe
        ),
        "minimum_task_fe": minimum_task_fe,
        "components_max": int(protocol.design["components_max"]),
        "population_size": int(protocol.suites[suite]["population_size"]),
        "evaluation_policy": evaluation_policy,
        # Native racing remains free to eliminate weak configurations early,
        # but the deterministic instance order exposes every registered
        # dimension before repeating one dimension.
        "promotion_requires_all_dimensions": len(
            {int(instance.dimension) for instance in instances}
        )
        > 1,
        "seed": int(protocol.seed),
        "protocol_fingerprint": protocol.fingerprint(),
        **method_space,
        "template_space_fingerprint": template_space_fingerprint(suite, **method_space),
    }


def run_platform_design(
    protocol: ExperimentProtocol,
    manifest_path: Path,
    artifact_root: Path,
    methods: set[str],
    suites: set[str],
    functions: set[int] | None,
    *,
    resume: bool,
) -> list[Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = []
    for suite in sorted(suites):
        for function_id in map(int, protocol.suites[suite]["functions"]):
            if functions is not None and function_id not in functions:
                continue
            directory = artifact_root / suite / f"f{function_id:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            for method in sorted(methods):
                entry = manifest.get(method, {})
                template = entry.get("design_command")
                if not isinstance(template, list):
                    raise ValueError(f"No design_command registered for {method!r}.")
                output = directory / (
                    "paradiseo-irace.json"
                    if method == "paradiseo_irace"
                    else "sparkle-smac3.json"
                )
                if resume and output.exists():
                    payload = json.loads(output.read_text(encoding="utf-8"))
                    validate_external_artifact(
                        payload, protocol, method, suite, function_id
                    )
                    outputs.append(output)
                    continue
                request = directory / f"{method}-design-request.json"
                atomic_json(
                    request,
                    _design_request(protocol, method, suite, function_id),
                )
                substitutions = {
                    "request": str(request.resolve()),
                    "artifact": str(output.resolve()),
                    "suite": suite,
                    "function": function_id,
                }
                command = [str(token).format(**substitutions) for token in template]
                subprocess.run(command, check=True)
                if not output.exists():
                    raise RuntimeError(
                        f"{method} returned successfully but did not create {output}."
                    )
                payload = json.loads(output.read_text(encoding="utf-8"))
                validate_external_artifact(
                    payload, protocol, method, suite, function_id
                )
                outputs.append(output)
    return outputs


def _tokens(value: str) -> set[str]:
    return {token.strip() for token in value.split(",") if token.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--suites", default="bbob,pbo")
    parser.add_argument("--functions")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    args = parser.parse_args(argv)
    methods = _tokens(args.methods)
    suites = _tokens(args.suites)
    unknown = methods - set(METHODS)
    if unknown:
        parser.error(f"unknown methods: {sorted(unknown)}")
    functions = (
        None if args.functions is None else {int(x) for x in _tokens(args.functions)}
    )
    outputs = run_platform_design(
        load_protocol(args.protocol),
        args.manifest,
        args.artifact_root,
        methods,
        suites,
        functions,
        resume=args.resume,
    )
    print(json.dumps([str(path.resolve()) for path in outputs], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
