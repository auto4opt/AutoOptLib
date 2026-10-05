"""Validated protocol and deterministic task expansion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from problems.ioh import IOHInstance


@dataclass(frozen=True)
class EvaluationTask:
    suite: str
    function_id: int
    dimension: int
    instance: int
    repeat: int
    seed: int
    budget: int
    population_size: int
    method: str

    @property
    def task_id(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    @property
    def ioh_instance(self) -> IOHInstance:
        return IOHInstance(self.dimension, self.instance, self.repeat)

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "function_id": self.function_id,
            "dimension": self.dimension,
            "instance": self.instance,
            "repeat": self.repeat,
            "seed": self.seed,
            "budget": self.budget,
            "population_size": self.population_size,
            "method": self.method,
        }


class ExperimentProtocol:
    def __init__(self, document: dict[str, Any], source: Path) -> None:
        if document.get("schema") != "autooptlib.paper-protocol":
            raise ValueError("Not an AutoOptLib paper protocol.")
        if int(document.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported paper protocol version.")
        # The fingerprint identifies the registered experimental task protocol,
        # including artifacts already produced under legacy spellings. Runtime
        # migration below may remove retired Search knobs, but must not make
        # otherwise compatible completed methods appear to use a new protocol.
        self._fingerprint_document = json.loads(json.dumps(document))
        self.document = document
        self.source = source.resolve()
        self.seed = int(document["seed"])
        self.suites = dict(document["suites"])
        self.design = dict(document["design"])
        self.recording = dict(document["recording"])
        self._validate()
        if "scoring_contract" in self.document:
            from methods.search_settings import validate_contract

            validate_contract(self)

    def _validate(self) -> None:
        budget = int(self.design["candidate_budget"])
        if budget <= 0:
            raise ValueError("candidate_budget must be positive")
        if self.design.get("candidate_budget_policy", "exact") not in {
            "exact",
            "actual_fe",
        }:
            raise ValueError("Retired candidate budget policy")
        if any(
            key in self.design
            for key in (
                "staged_exact_selection",
                "budget_extension",
                "search_warm_start",
            )
        ):
            raise ValueError("Retired experiment overrides")
        for name, suite in self.suites.items():
            if name not in {"bbob", "pbo"}:
                raise ValueError(f"Unknown suite: {name}")
            upper = 24 if name == "bbob" else 23
            functions = suite["functions"]
            if not functions or any(
                type(fid) is not int or not 1 <= fid <= upper for fid in functions
            ):
                raise ValueError("Invalid function list")
            for fid in functions:
                instances = self.training_instances(name, fid)
                if len(instances) != int(self.design["training_runs"]):
                    raise ValueError(
                        f"{name} f{fid} must define exactly {self.design['training_runs']} design-training tasks"
                    )
                for phase in ("train", "validation", "test"):
                    ids, repeats = self._phase_schedule(name, fid, phase)
                    if not ids or min(ids) <= 0 or repeats <= 0:
                        raise ValueError(f"Invalid {phase} schedule")
            if self.design_fe_cap(name) <= 0 or self.candidate_training_fe(name) <= 0:
                raise ValueError("FE budgets must be positive")
            if self.design.get(
                "candidate_budget_policy", "exact"
            ) == "exact" and self.design_fe_cap(
                name
            ) != budget * self.candidate_training_fe(name):
                raise ValueError("FE cap does not match candidate budget")
            if "aol_search" in suite.get("methods", []):
                from methods.search_settings import require_search_protocol

                require_search_protocol(self)
        checkpoints = self.validation_checkpoint_candidates()
        if checkpoints and (
            checkpoints != sorted(set(checkpoints))
            or checkpoints[-1] != budget
            or checkpoints[0] <= 0
        ):
            raise ValueError("Invalid checkpoint schedule")
        learning = self.design.get("learning", {})
        if learning:
            if learning.get("update_method") != "archive_imitation" or learning.get(
                "dual_origin_sampling", False
            ):
                raise ValueError(
                    "Only single-policy archive-imitation Learning is supported"
                )
            generated = int(learning["full_batches"]) * int(
                learning["batch_size"]
            ) + int(learning["last_batch_size"])
            if generated != budget:
                raise ValueError("Learning batches must exactly match candidate budget")
        if self.design.get("compare") != "average":
            raise ValueError(
                "Paper methods require the scalar-score selection route ('average')"
            )
        policies = self.design.get("autooptlib_evaluation_policy", {})
        if any(value != "exact" for value in policies.values()):
            raise ValueError("Current methods require exact evaluation")

    def _seed_for(
        self,
        suite: str,
        function_id: int,
        instance: int,
        repeat: int,
        *,
        phase: str = "test",
        dimension: int = 0,
    ) -> int:
        suite_name = str(suite).lower()
        digest = hashlib.sha256(
            (
                f"{self.seed}:{suite_name}:{int(function_id)}:{phase}:"
                f"{int(dimension)}:{int(instance)}:{int(repeat)}"
            ).encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF

    def task_seed(
        self,
        suite: str,
        function_id: int,
        instance: int,
        repeat: int,
        *,
        phase: str = "training",
        dimension: int = 0,
    ) -> int:
        """Return the shared method-independent seed for one benchmark task."""

        return self._seed_for(
            suite,
            function_id,
            instance,
            repeat,
            phase=phase,
            dimension=dimension,
        )

    def _phase_schedule(
        self, suite: str, function_id: int | None, phase: str
    ) -> tuple[list[int], int]:
        """Resolve a suite schedule, including a registered function exception."""

        keys = {
            "train": ("train_instances", "runs_per_train_instance"),
            "validation": (
                "validation_instances",
                "runs_per_validation_instance",
            ),
            "confirmation": (
                "confirmation_instances",
                "runs_per_confirmation_instance",
            ),
            "test": ("test_instances", "runs_per_test_instance"),
        }
        try:
            instance_key, run_key = keys[phase]
        except KeyError as exc:
            raise ValueError(f"Unknown experiment phase {phase!r}.") from exc
        config = self.suites[suite]
        override: dict[str, Any] = {}
        if function_id is not None:
            raw = config.get("function_overrides", {}).get(str(int(function_id)), {})
            if not isinstance(raw, dict):
                raise ValueError(
                    f"{suite} f{function_id} function override must be an object."
                )
            override = raw
        instances = override.get(instance_key, config[instance_key])
        runs = override.get(run_key, config[run_key])
        return [int(value) for value in instances], int(runs)

    def training_run_budget(self, suite: str, dimension: int) -> int:
        config = self.suites[suite]
        budgets = config.get("train_budgets")
        if isinstance(budgets, dict):
            return int(budgets[str(int(dimension))])
        return int(self.design["candidate_run_budget"])

    def final_run_budget(self, suite: str, dimension: int) -> int:
        config = self.suites[suite]
        budgets = config.get("test_budgets")
        if isinstance(budgets, dict):
            return int(budgets[str(int(dimension))])
        return int(config["final_budget"])

    def candidate_training_fe(self, suite: str) -> int:
        return sum(
            int(instance.budget or self.design.get("candidate_run_budget", 0))
            for instance in self.training_instances(suite)
        )

    def design_fe_cap(self, suite: str) -> int:
        value = self.design["total_fe_cap"]
        return int(value[suite] if isinstance(value, dict) else value)

    def validation_checkpoint_candidates(self) -> list[int]:
        return [
            int(value)
            for value in self.design.get("validation", {}).get(
                "checkpoint_candidates", []
            )
        ]

    def validation_methods(self) -> set[str]:
        return {
            str(value) for value in self.design.get("validation", {}).get("methods", [])
        }

    def evaluation_tasks(
        self,
        *,
        suites: set[str] | None = None,
        methods: set[str] | None = None,
        functions: set[int] | None = None,
    ) -> Iterator[EvaluationTask]:
        for suite_name, suite in self.suites.items():
            if suites is not None and suite_name not in suites:
                continue
            dimensions = [
                int(value)
                for value in suite.get("test_dimensions", [suite.get("test_dimension")])
            ]
            for function_id in map(int, suite["functions"]):
                if functions is not None and function_id not in functions:
                    continue
                test_instances, test_repeats = self._phase_schedule(
                    suite_name, function_id, "test"
                )
                for method in map(str, suite["methods"]):
                    if methods is not None and method not in methods:
                        continue
                    for dimension in dimensions:
                        for instance in test_instances:
                            for repeat in range(test_repeats):
                                yield EvaluationTask(
                                    suite=suite_name,
                                    function_id=function_id,
                                    dimension=dimension,
                                    instance=instance,
                                    repeat=repeat,
                                    seed=self._seed_for(
                                        suite_name,
                                        function_id,
                                        instance,
                                        repeat,
                                        phase="test",
                                        dimension=dimension,
                                    ),
                                    budget=self.final_run_budget(suite_name, dimension),
                                    population_size=int(suite["population_size"]),
                                    method=method,
                                )

    def training_instances(
        self, suite: str, function_id: int | None = None
    ) -> list[IOHInstance]:
        config = self.suites[suite]
        if "runs_per_train_instance" in config:
            train_instances, train_repeats = self._phase_schedule(
                suite, function_id, "train"
            )
            return [
                IOHInstance(
                    int(dimension),
                    int(instance),
                    int(repeat),
                    self.training_run_budget(suite, int(dimension)),
                )
                for dimension in config["train_dimensions"]
                for instance in train_instances
                for repeat in range(train_repeats)
            ]
        if suite == "bbob":
            if "training_runs_per_dimension" in config:
                instances = [int(value) for value in config["train_instances"]]
                repeats = int(config["training_runs_per_dimension"])
                if not instances or repeats <= 0:
                    raise ValueError(
                        "BBOB balanced training requires instances and positive "
                        "training_runs_per_dimension."
                    )
                return [
                    IOHInstance(
                        int(dimension),
                        instances[repeat % len(instances)],
                        repeat,
                        self.training_run_budget(suite, int(dimension)),
                    )
                    for dimension in config["train_dimensions"]
                    for repeat in range(repeats)
                ]
            return [
                IOHInstance(
                    int(dimension),
                    int(instance),
                    0,
                    self.training_run_budget(suite, int(dimension)),
                )
                for dimension in config["train_dimensions"]
                for instance in config["train_instances"]
            ]
        dimensions = [int(value) for value in config["train_dimensions"]]
        count = int(config.get("training_runs", self.design["training_runs"]))
        # A deterministic balanced pool: every dimension appears before any
        # receives its next repeat, satisfying the PBO promotion constraint.
        return [
            IOHInstance(
                dimensions[index % len(dimensions)],
                1,
                index,
                self.training_run_budget(suite, dimensions[index % len(dimensions)]),
            )
            for index in range(count)
        ]

    def validation_instances(self, suite: str, function_id: int) -> list[IOHInstance]:
        """Return the held-out tasks used only for checkpoint selection."""

        config = self.suites[suite]
        instances, repeats = self._phase_schedule(suite, function_id, "validation")
        return [
            IOHInstance(
                int(dimension),
                int(instance),
                int(repeat),
                self.final_run_budget(suite, int(dimension)),
            )
            for dimension in config["test_dimensions"]
            for instance in instances
            for repeat in range(repeats)
        ]

    def confirmation_instances(self, suite: str, function_id: int) -> list[IOHInstance]:
        """Return the independent tasks used to confirm screened finalists."""

        config = self.suites[suite]
        instances, repeats = self._phase_schedule(suite, function_id, "confirmation")
        return [
            IOHInstance(
                int(dimension),
                int(instance),
                int(repeat),
                self.final_run_budget(suite, int(dimension)),
            )
            for dimension in config["test_dimensions"]
            for instance in instances
            for repeat in range(repeats)
        ]

    def fingerprint(self) -> str:
        payload = json.dumps(
            self._fingerprint_document, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_protocol(path: str | Path) -> ExperimentProtocol:
    source = Path(path)
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError("Protocol root must be a JSON object.")
    return ExperimentProtocol(document, source)


__all__ = ["EvaluationTask", "ExperimentProtocol", "load_protocol"]
