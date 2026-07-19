"""Run deterministic application baselines and retain every raw run.

This is an integration/repeatability harness, not a reproduction of the paper's
private industrial data or original wireless channels.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from autooptlib import (
    autoopt,
    generate_ris_instance,
    generate_stacking_instance,
    make_material_stacking_problem,
    make_ris_beamforming_problem,
)


def _build_problem(name: str, specifications: list[dict[str, Any]]):
    instances = {}
    for spec in specifications:
        instance_id = spec["id"]
        if name == "material_stacking":
            instances[instance_id] = generate_stacking_instance(
                spec["size"], seed=spec["seed"]
            )
        elif name == "ris_beamforming":
            instances[instance_id] = generate_ris_instance(
                spec["size"], seed=spec["seed"]
            )
        else:
            raise ValueError(f"Unknown application {name!r}.")
    factory = (
        make_material_stacking_problem
        if name == "material_stacking"
        else make_ris_beamforming_problem
    )
    return factory(instances), list(instances)


def run(config: dict[str, Any], output_directory: Path) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    base_seed = int(config["seed"])
    runs = int(config["runs"])

    for app_index, (application, app_config) in enumerate(
        config["applications"].items()
    ):
        problem, instance_ids = _build_problem(application, app_config["instances"])
        for algorithm_index, algorithm_spec in enumerate(config["algorithms"]):
            if isinstance(
                algorithm_spec, dict
            ) and application not in algorithm_spec.get("applications", [application]):
                continue
            if isinstance(algorithm_spec, str):
                algorithm = algorithm_spec
                algorithm_option = {"AlgName": algorithm_spec}
            else:
                algorithm = str(algorithm_spec["label"])
                algorithm_option = {"AlgFile": str(algorithm_spec["file"])}
            for instance_index, instance_id in enumerate(instance_ids):
                for run_index in range(runs):
                    seed = (
                        base_seed
                        + app_index * 100_000
                        + algorithm_index * 10_000
                        + instance_index * 100
                        + run_index
                    )
                    run_dir = (
                        output_directory
                        / "runs"
                        / application
                        / algorithm.lower().replace(" ", "-")
                        / str(instance_id)
                        / f"run-{run_index + 1}"
                    )
                    best, _ = autoopt(
                        Mode="solve",
                        Problem=problem,
                        InstanceSolve=[instance_id],
                        AlgRuns=1,
                        ProbN=int(config["prob_n"]),
                        ProbFE=int(config["prob_fe"]),
                        Seed=seed,
                        OutputDir=run_dir,
                        **algorithm_option,
                    )
                    records.append(
                        {
                            "application": application,
                            "algorithm": algorithm,
                            "instance": instance_id,
                            "run": run_index + 1,
                            "seed": seed,
                            "fitness": float(best[0][0].fit),
                        }
                    )

    groups: dict[tuple[str, str, str], list[float]] = {}
    for record in records:
        key = (record["application"], record["algorithm"], str(record["instance"]))
        groups.setdefault(key, []).append(record["fitness"])
    summary = [
        {
            "application": key[0],
            "algorithm": key[1],
            "instance": key[2],
            "mean": fmean(values),
            "std": pstdev(values),
            "runs": len(values),
        }
        for key, values in sorted(groups.items())
    ]
    return {
        "schema": "autooptlib.application-benchmark",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_scope": "synthetic reference instances; not paper result data",
        "config": config,
        "raw_runs": records,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("application-smoke.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/repeatability"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    document = run(config, args.output)
    target = args.output / "application-benchmarks.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(target)


if __name__ == "__main__":
    main()
