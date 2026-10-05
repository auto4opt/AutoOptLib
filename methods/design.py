"""One-shot automated design workflows with a shared FE ledger."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from autooptlib import autoopt, save_algorithm
from autooptlib.runtime.resources import discover_cpu_topology
from autooptlib.serialization import algorithm_from_dict, algorithm_to_dict
from autooptlib.utils.design._stream_graph import STREAM_GRAPH_SEMANTICS
from autooptlib.utils.general.candidate_log import (
    append_candidate_records,
    trim_candidate_records,
)

from comparisons.shared.protocol import ExperimentProtocol
from comparisons.shared.scoring import make_taskwise_scorer
from comparisons.shared.selection_identity import selection_structure_key
from comparisons.shared.store import atomic_json, utc_now
from methods.stream_graph_bridge import IMPLEMENTATION_REVISION as SEARCH_V6_REVISION
from methods.stream_graph_bridge import fastga_initial_spec
from problems import (
    IOH_OPTIMUM_METADATA_REVISION,
    IOHInstance,
    known_ioh_optimum_raw,
    make_ioh_problem,
)

LEARNING_IMPLEMENTATION_REVISION = "learning-transfer-frozen-20260930"
CHECKPOINT_EXPORT_REVISION = "checkpoint-export-audit6"
DUAL_ORIGIN_FRESH_SEED_OFFSET = 1000003


def _expected_search_revision(protocol: ExperimentProtocol) -> str:
    return SEARCH_V6_REVISION


def _artifact_directory(root: Path, suite: str, function_id: int) -> Path:
    path = root / suite / f"f{function_id:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _search_evaluation_workers(configured: int | str) -> int:
    """Resolve Search's heterogeneous candidate batches to physical CPU cores.

    Adaptive throughput trials compare distinct candidate algorithms, whose
    runtimes can differ by orders of magnitude.  Their observed throughput is
    therefore not a valid worker-count comparison.  Use one worker per physical
    core for Search and keep its per-coordinate deterministic seeds unchanged.
    """
    override = os.environ.get("AOL_SEARCH_EVAL_WORKERS")
    if override is not None:
        workers = int(override)
        if workers <= 0:
            raise ValueError("AOL_SEARCH_EVAL_WORKERS must be positive.")
        return workers
    if not isinstance(configured, str):
        return int(configured)
    if configured.lower() != "auto":
        raise ValueError("Search workers must be 'auto' or a positive integer.")
    return max(1, len({item.physical_core for item in discover_cpu_topology()}))


def _random_evaluation_workers(configured: int | str) -> int:
    """Resolve RandomDesign workers without heterogeneous online tuning.

    Randomly sampled algorithms can differ greatly in runtime, so worker-count
    trials performed on different candidates do not provide a valid throughput
    comparison.  Slurm campaigns may explicitly match this value to the CPUs
    reserved for each outer design task.
    """
    override = os.environ.get("AOL_RANDOM_EVAL_WORKERS")
    if override is not None:
        workers = int(override)
        if workers <= 0:
            raise ValueError("AOL_RANDOM_EVAL_WORKERS must be positive.")
        return workers
    return _search_evaluation_workers(configured)


def _trim_jsonl_records(path: Path, completed: int) -> None:
    """Roll an append-only JSONL file back to its checkpoint boundary."""
    if not path.exists():
        if completed:
            raise RuntimeError(f"History {path} is missing {completed} records.")
        return
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) < completed:
        raise RuntimeError(
            f"History {path} has {len(lines)} records; checkpoint requires {completed}."
        )
    for index, line in enumerate(lines[:completed], 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"History {path} has invalid JSON at record {index}."
            ) from exc
    if len(lines) == completed:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = "\n".join(lines[:completed])
    temporary.write_text(text + ("\n" if text else ""), encoding="utf-8")
    temporary.replace(path)


def _torch_rng_payload(torch) -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "get_rng_state"):
        try:
            state["mps"] = mps.get_rng_state()
        except RuntimeError:
            pass
    return state


def _restore_torch_rng(torch, state: dict[str, Any]) -> None:
    if "cpu" not in state:
        raise ValueError("Learning checkpoint has no CPU RNG state.")
    torch.set_rng_state(state["cpu"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    mps = getattr(torch, "mps", None)
    if (
        "mps" in state
        and mps is not None
        and hasattr(mps, "set_rng_state")
        and torch.backends.mps.is_available()
    ):
        mps.set_rng_state(state["mps"])


def _learning_batch_schedule(
    candidate_budget: int,
    batch_size: int,
    boundaries: list[int] | set[int] | tuple[int, ...] = (),
) -> list[int]:
    """Split a unique-candidate budget into stable policy update batches.

    Saved evaluation checkpoints are observational and must not change the
    optimizer's batch sizes.  ``boundaries`` remains accepted for callers that
    also use it to plan checkpoint exports, but it deliberately has no effect
    on the update schedule.
    """
    if candidate_budget <= 0 or batch_size <= 0:
        raise ValueError("candidate_budget and batch_size must be positive.")
    del boundaries
    (full_batches, remainder) = divmod(candidate_budget, batch_size)
    return [batch_size] * full_batches + ([remainder] if remainder else [])


def _rolling_unique_rate(outcomes: list[int] | tuple[int, ...] | deque[int]) -> float:
    """Return the fraction of raw policy samples that produced a new graph."""
    return float(sum(outcomes) / len(outcomes)) if outcomes else 1.0


def _pad_learning_sequence_tensors(
    torch: Any, sequence_tensors: list[Any], *, end_index: int
) -> Any:
    """Pad independently generated sequences with the grammar end token."""
    if not sequence_tensors:
        raise ValueError("At least one generated sequence is required.")
    rows = []
    for tensor in sequence_tensors:
        row = torch.as_tensor(tensor, dtype=torch.long)
        if row.ndim == 2 and row.shape[0] == 1:
            row = row.squeeze(0)
        if row.ndim != 1 or row.numel() < 2:
            raise ValueError("Generated sequences must be non-empty token rows.")
        rows.append(row)
    return torch.nn.utils.rnn.pad_sequence(
        rows, batch_first=True, padding_value=int(end_index)
    )


def _learning_grouped_task_costs(scorer, performances, instances):
    """Normalize each run and average repeats within dimension/instance tasks."""
    groups = {}
    for index, task in enumerate(instances):
        groups.setdefault((task.dimension, task.instance), []).append(index)
    return np.asarray(
        [
            [float(np.mean(scores[indices])) for indices in groups.values()]
            for scores in (scorer.taskwise(values) for values in performances)
        ],
        dtype=float,
    )


def _unique_graph_sample_indices(
    sequence_keys: list[tuple[int, ...]],
    graph_keys: list[str],
    *,
    historical_sequences: set[tuple[int, ...]],
    historical_graphs: set[str],
    batch_sequences: set[tuple[int, ...]],
    batch_graphs: set[str],
) -> tuple[list[int], int, int, int]:
    """Select graphs unseen in history and the pending policy batch.

    Sequence duplication is diagnostic only: graph identity is the canonical
    criterion because different token sequences can decode to one algorithm.
    """
    if len(sequence_keys) != len(graph_keys):
        raise ValueError("sequence_keys and graph_keys must have equal length.")
    accepted: list[int] = []
    duplicate_sequences = 0
    duplicate_graphs = 0
    cache_hits = 0
    observed_sequences = set(batch_sequences)
    observed_graphs = set(batch_graphs)
    for index, (sequence_key, graph_key) in enumerate(zip(sequence_keys, graph_keys)):
        if sequence_key in historical_sequences or sequence_key in observed_sequences:
            duplicate_sequences += 1
        if graph_key in historical_graphs:
            duplicate_graphs += 1
            cache_hits += 1
        elif graph_key in observed_graphs:
            duplicate_graphs += 1
        else:
            accepted.append(index)
            observed_graphs.add(graph_key)
        observed_sequences.add(sequence_key)
    return (accepted, duplicate_sequences, duplicate_graphs, cache_hits)


def _candidate_summary(path: Path) -> dict[str, int]:
    candidates = 0
    actual_fes = 0
    failures = 0
    screened = 0
    completed_candidate_tasks = 0
    representations = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            candidates += 1
            actual_fes += int(record.get("actual_design_fes", 0))
            failures += record.get("status") not in {"ok", "screened_out"}
            screened += record.get("status") == "screened_out"
            completed_candidate_tasks += len(record.get("evaluation_ledger") or [])
            encoded = json.dumps(
                {
                    "configuration": record.get("configuration"),
                    "pathways": record.get("representation"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            representations.add(hashlib.sha256(encoded).hexdigest())
    return {
        "generated_candidates": candidates,
        "different_candidates": len(representations),
        "actual_design_fes": actual_fes,
        "failed_candidates": failures,
        "screened_out_candidates": screened,
        "completed_candidate_tasks": completed_candidate_tasks,
    }


def _require_complete_exact_ledger(
    summary: dict[str, int], *, method: str, candidate_budget: int, training_tasks: int
) -> None:
    expected = int(candidate_budget) * int(training_tasks)
    completed = int(summary["completed_candidate_tasks"])
    failures = int(summary.get("failed_candidates", 0))
    if failures:
        raise RuntimeError(
            f"{method} produced {failures} failed or non-finite candidates."
        )
    if completed != expected:
        raise RuntimeError(
            f"{method} exact evaluation completed {completed} candidate-task evaluations; expected {expected}."
        )


def _checkpoint_candidate_counts(
    protocol: ExperimentProtocol, suite: str | None = None
) -> list[tuple[int, int]]:
    """Return registered nominal FE checkpoints and their candidate prefixes."""
    registered = protocol.validation_checkpoint_candidates()
    if registered:
        if suite is None:
            raise ValueError(
                "suite is required for dimension-dependent candidate checkpoints."
            )
        per_candidate = protocol.candidate_training_fe(suite)
        return [(count * per_candidate, count) for count in registered]
    per_candidate = int(protocol.design["training_runs"]) * int(
        protocol.design["candidate_run_budget"]
    )
    return [
        (int(fe), int(fe) // per_candidate)
        for fe in protocol.design.get("budget_checkpoints_fe", [])
    ]


def _export_budget_checkpoints(
    protocol: ExperimentProtocol,
    suite: str,
    function_id: int,
    method: str,
    *,
    artifact_root: Path,
) -> None:
    """Materialize the best actually evaluated algorithm at each FE prefix."""
    if method not in protocol.validation_methods():
        return
    if method == "aol_learning":
        return
    checkpoints = _checkpoint_candidate_counts(protocol, suite)
    if not checkpoints:
        return
    output = _artifact_directory(artifact_root, suite, function_id)
    candidate_path = output / f"{method}-candidates.jsonl"
    design_path = output / f"{method}-design.json"
    if not candidate_path.exists() or not design_path.exists():
        raise FileNotFoundError(
            f"Cannot export budget checkpoints without {candidate_path} and {design_path}."
        )
    records = [
        json.loads(line)
        for line in candidate_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    design_document = json.loads(design_path.read_text(encoding="utf-8"))
    actual_fe_policy = protocol.design.get("candidate_budget_policy") == "actual_fe"
    if actual_fe_policy:
        checkpoints = [(protocol.design_fe_cap(suite), len(records))]
    expected_design_identity = {
        "method": method,
        "suite": suite,
        "function_id": function_id,
        "protocol_fingerprint": protocol.fingerprint(),
    }
    for field, expected_value in expected_design_identity.items():
        if design_document.get(field) != expected_value:
            raise RuntimeError(
                f"Checkpoint source design {field} does not match {expected_value!r}."
            )
    source_revision = design_document.get("search_hyperparameters", {}).get(
        "implementation_revision"
    )
    if method == "aol_search":
        expected_revision = _expected_search_revision(protocol)
        allowed_revisions = {expected_revision}
        if source_revision not in allowed_revisions:
            raise RuntimeError(
                f"Checkpoint source Search revision is not an approved source for {expected_revision}: {source_revision!r}."
            )
    candidate_ledger_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    source_design_sha256 = hashlib.sha256(design_path.read_bytes()).hexdigest()
    recorded_ledger_sha256 = design_document.get("candidate_ledger_sha256")
    if (
        recorded_ledger_sha256 is not None
        and recorded_ledger_sha256 != candidate_ledger_sha256
    ):
        raise RuntimeError("Checkpoint source candidate ledger hash does not match.")
    if source_revision == SEARCH_V6_REVISION and recorded_ledger_sha256 is None:
        raise RuntimeError(
            "Current Search design is missing its candidate ledger hash."
        )
    source_summary = _candidate_summary(candidate_path)
    recorded_summary = dict(design_document.get("design_ledger") or {})
    for field in (
        "generated_candidates",
        "different_candidates",
        "actual_design_fes",
        "failed_candidates",
        "completed_candidate_tasks",
    ):
        if int(recorded_summary.get(field, -1)) != int(source_summary[field]):
            raise RuntimeError(
                f"Checkpoint source candidate ledger disagrees on {field}."
            )
    curve = list(design_document.get("curve", []))
    validation = dict(protocol.design.get("validation", {}))
    elite_count = (
        int(validation.get("elites_per_checkpoint", 5)) if method == "aol_search" else 1
    )
    training_pool = int(validation.get("training_pool", 20))
    max_per_structure = int(validation.get("max_per_structure", 2))
    deduplicate_across_checkpoints = bool(
        validation.get("deduplicate_across_checkpoints", False)
    )
    exported_semantics: set[str] = set()
    manifest = []
    racing = (
        method == "aol_search"
        and protocol.design.get("autooptlib_evaluation_policy", {}).get(method)
        == "racing"
    )
    for nominal_fe, candidate_count in checkpoints:
        prefix = (
            [
                record
                for record in records
                if 0
                < int(
                    record.get("candidate_metadata", {}).get(
                        "training_completed_design_fes", 0
                    )
                )
                <= nominal_fe
            ]
            if racing
            else records[:candidate_count]
        )
        if not racing and len(prefix) != candidate_count:
            raise RuntimeError(
                f"{method} {suite} f{function_id} has only {len(prefix)} candidate records at the {candidate_count}-candidate checkpoint."
            )
        valid = [
            record
            for record in prefix
            if record.get("status") == "ok" and record.get("training_cost") is not None
        ]
        if not valid:
            raise RuntimeError(
                f"{method} {suite} f{function_id} has no valid candidate at {nominal_fe} FE."
            )
        ordered = sorted(
            valid,
            key=lambda record: (
                float(record["training_cost"]),
                int(record["candidate"]),
            ),
        )
        semantic_keys = {
            int(record["candidate"]): json.dumps(
                {
                    "configuration": record.get("configuration"),
                    "pathways": record["representation"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for record in ordered
        }
        eligible = [
            record
            for record in ordered
            if not deduplicate_across_checkpoints
            or semantic_keys[int(record["candidate"])] not in exported_semantics
        ]
        pool = eligible[: max(elite_count, training_pool)]
        selected_records: list[dict[str, Any]] = []
        checkpoint_semantics: set[str] = set()
        structure_counts: dict[str, int] = {}
        for record in pool:
            semantic_key = semantic_keys[int(record["candidate"])]
            if deduplicate_across_checkpoints and semantic_key in checkpoint_semantics:
                continue
            structure_key = selection_structure_key(
                {"pathways": record["representation"]}
            )
            if structure_counts.get(structure_key, 0) >= max_per_structure:
                continue
            selected_records.append(record)
            checkpoint_semantics.add(semantic_key)
            structure_counts[structure_key] = structure_counts.get(structure_key, 0) + 1
            if len(selected_records) >= elite_count:
                break
        for record in pool:
            if len(selected_records) >= elite_count:
                break
            semantic_key = semantic_keys[int(record["candidate"])]
            if record not in selected_records and (
                not deduplicate_across_checkpoints
                or semantic_key not in checkpoint_semantics
            ):
                selected_records.append(record)
                checkpoint_semantics.add(semantic_key)
        for record in eligible[len(pool) :]:
            if len(selected_records) >= elite_count:
                break
            semantic_key = semantic_keys[int(record["candidate"])]
            if semantic_key not in checkpoint_semantics:
                selected_records.append(record)
                checkpoint_semantics.add(semantic_key)
        if deduplicate_across_checkpoints and len(selected_records) != elite_count:
            raise RuntimeError(
                f"{method} {suite} f{function_id} cannot export {elite_count} globally distinct checkpoint candidates."
            )
        if deduplicate_across_checkpoints:
            exported_semantics.update(checkpoint_semantics)
        selected = selected_records[0]
        checkpoint_root = (
            artifact_root.parent
            / "checkpoints"
            / (
                f"fe-{nominal_fe}"
                if actual_fe_policy
                else f"candidates-{candidate_count:04d}"
            )
            / "artifacts"
        )
        checkpoint_output = _artifact_directory(checkpoint_root, suite, function_id)
        elite_manifest = []
        artifact = checkpoint_output / f"{method}.json"
        for elite_rank, elite in enumerate(selected_records, 1):
            algorithm_schema_version = elite.get("algorithm_schema_version")
            execution_semantics = elite.get("execution_semantics")
            protocol_semantics = (
                str(
                    protocol.design.get("search", {}).get(
                        "graph_semantics", "legacy_pathway_v1"
                    )
                ).lower()
                if method == "aol_search"
                else "legacy_pathway_v1"
            )
            if algorithm_schema_version is None:
                if protocol_semantics == STREAM_GRAPH_SEMANTICS:
                    algorithm_schema_version = 2
                    execution_semantics = STREAM_GRAPH_SEMANTICS
                else:
                    algorithm_schema_version = 1
            elif method == "aol_search":
                expected_version = (
                    2 if protocol_semantics == STREAM_GRAPH_SEMANTICS else 1
                )
                expected_semantics = (
                    STREAM_GRAPH_SEMANTICS
                    if protocol_semantics == STREAM_GRAPH_SEMANTICS
                    else None
                )
                if (
                    int(algorithm_schema_version) != expected_version
                    or execution_semantics != expected_semantics
                ):
                    raise RuntimeError(
                        f"Checkpoint candidate {elite['candidate']} algorithm schema does not match the Search execution semantics in the protocol."
                    )
            algorithm_document = {
                "schema": "autooptlib.algorithm",
                "schema_version": int(algorithm_schema_version),
                "metadata": {},
                "configuration": elite.get("configuration"),
                "pathways": elite["representation"],
            }
            if execution_semantics is not None:
                algorithm_document["execution_semantics"] = execution_semantics
            algorithm = algorithm_from_dict(algorithm_document)
            round_trip = algorithm_to_dict(algorithm)
            for field in (
                "schema_version",
                "execution_semantics",
                "configuration",
                "pathways",
            ):
                if round_trip.get(field) != algorithm_document.get(field):
                    raise RuntimeError(
                        f"Checkpoint candidate {elite['candidate']} changed {field} during algorithm round-trip."
                    )
            elite_artifact = (
                checkpoint_output / f"{method}-elite-{elite_rank:02d}.json"
                if method == "aol_search"
                else artifact
            )
            metadata = {
                "designer": method.removeprefix("aol_"),
                "suite": suite,
                "function_id": function_id,
                "protocol_fingerprint": protocol.fingerprint(),
                "budget_checkpoint_fe": nominal_fe,
                "budget_checkpoint_candidates": candidate_count,
                "checkpoint_candidate_semantics": "actual_candidates_at_fe_cap"
                if actual_fe_policy
                else "nominal_full_candidate_equivalents"
                if racing
                else "candidates",
                "completed_candidates_at_checkpoint": len(prefix),
                "selected_candidate": int(elite["candidate"]),
                "selected_training_cost": float(elite["training_cost"]),
                "elite_rank": elite_rank,
                "selection": "training_pool_diverse_elite",
                "deduplicate_across_checkpoints": deduplicate_across_checkpoints,
                "source_candidate_metadata": elite.get("candidate_metadata", {}),
                "implementation_revision": source_revision,
                "checkpoint_export_revision": CHECKPOINT_EXPORT_REVISION,
                "source_candidate_ledger_sha256": candidate_ledger_sha256,
                "source_design_sha256": source_design_sha256,
            }
            save_algorithm(algorithm, elite_artifact, metadata=metadata)
            if elite_rank == 1 and elite_artifact != artifact:
                save_algorithm(algorithm, artifact, metadata=metadata)
            elite_manifest.append(
                {
                    "rank": elite_rank,
                    "artifact": str(elite_artifact.resolve()),
                    "selected_candidate": int(elite["candidate"]),
                    "selected_training_cost": float(elite["training_cost"]),
                }
            )
        prefix_curve = [
            point
            for point in curve
            if (
                int(point.get("design_fes", 0)) <= nominal_fe
                if racing
                else int(point.get("candidates", 0)) <= candidate_count
            )
        ]
        summary = _candidate_summary_from_records(prefix)
        if racing:
            summary["actual_design_fes"] = max(
                (
                    int(record["candidate_metadata"]["training_completed_design_fes"])
                    for record in prefix
                )
            )
        checkpoint_design = {
            **design_document,
            "created_utc": utc_now(),
            "candidate_budget": candidate_count,
            "full_protocol_candidate_budget": int(protocol.design["candidate_budget"]),
            "budget_checkpoint_fe": nominal_fe,
            "artifact": str(artifact.resolve()),
            "selected_candidate": int(selected["candidate"]),
            "selected_training_cost": float(selected["training_cost"]),
            "selection": "training_pool_diverse_elites",
            "validation_elites": elite_manifest,
            "design_ledger": summary,
            "curve": prefix_curve,
        }
        atomic_json(checkpoint_output / f"{method}-design.json", checkpoint_design)
        manifest.append(
            {
                "nominal_fe": nominal_fe,
                "candidate_count": candidate_count,
                "artifact": str(artifact.resolve()),
                "selected_candidate": int(selected["candidate"]),
                "selected_training_cost": float(selected["training_cost"]),
                "validation_elites": elite_manifest,
                "actual_design_fes": summary["actual_design_fes"],
            }
        )
    atomic_json(
        output / f"{method}-budget-checkpoints.json",
        {
            "schema": "autooptlib.paper-budget-checkpoints",
            "schema_version": 1,
            "created_utc": utc_now(),
            "method": method,
            "suite": suite,
            "function_id": function_id,
            "protocol_fingerprint": protocol.fingerprint(),
            "checkpoints": manifest,
        },
    )


def _candidate_summary_from_records(records: list[dict[str, Any]]) -> dict[str, int]:
    representations = {
        hashlib.sha256(
            json.dumps(
                {
                    "configuration": record.get("configuration"),
                    "pathways": record.get("representation"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        for record in records
    }
    return {
        "generated_candidates": len(records),
        "different_candidates": len(representations),
        "actual_design_fes": sum(
            (int(record.get("actual_design_fes", 0)) for record in records)
        ),
        "failed_candidates": sum(
            (record.get("status") not in {"ok", "screened_out"} for record in records)
        ),
        "completed_candidate_tasks": sum(
            (len(record.get("evaluation_ledger") or []) for record in records)
        ),
    }


def _training_normalizer(suite: str, function_id: int, instances: list[Any]):
    import ioh

    problem_class = ioh.ProblemClass.BBOB if suite == "bbob" else ioh.ProblemClass.PBO
    optimum_costs = []
    for instance in instances:
        problem = ioh.get_problem(
            function_id,
            instance=instance.instance,
            dimension=instance.dimension,
            problem_class=problem_class,
        )
        optimum = known_ioh_optimum_raw(
            suite,
            function_id,
            IOHInstance(
                dimension=instance.dimension,
                instance=instance.instance,
                repeat=getattr(instance, "repeat", 0),
                budget=getattr(instance, "budget", None),
            ),
            objective=problem,
        )
        if not np.isfinite(optimum):
            return (lambda _value: None, "unavailable_unknown_optimum")
        optimum_costs.append(optimum if suite == "bbob" else -optimum)
    reference = float(np.mean(optimum_costs))
    if suite == "bbob":
        return (
            lambda value: float(np.log10(1.0 + max(0.0, float(value) - reference))),
            "log10(1 + mean_training_delta_f)",
        )
    scale = max(1.0, float(np.mean(np.abs(optimum_costs))))
    return (
        lambda value: max(0.0, float(value) - reference) / scale,
        "mean_training_gap / max(1, mean_abs_optimum)",
    )


def _curve_from_trace(
    trace: list[Any],
    *,
    initial_candidates: int,
    batch_size: int,
    training_tasks: int,
    run_budget: int,
    candidate_budget: int | None,
    scorer=None,
    normalize=None,
) -> list[dict[str, float | int]]:
    result = []
    for index, design in enumerate(trace, 1):
        metadata = getattr(design, "metadata", {}) or {}
        candidates = int(
            metadata.get(
                "cumulative_candidates",
                metadata.get("candidate", initial_candidates + index * batch_size),
            )
        )
        if candidate_budget is not None:
            candidates = min(candidate_budget, candidates)
        values = np.asarray(design.performance[:training_tasks, :], dtype=float)
        training_cost = float(np.mean(values))
        metadata_score = metadata.get("training_score")
        if metadata_score is not None:
            normalized_score = float(metadata_score)
        elif scorer is not None:
            normalized_score = float(scorer(values, list(range(training_tasks))))
        elif normalize is not None:
            normalized_score = float(normalize(training_cost))
        else:
            normalized_score = training_cost
        result.append(
            {
                "generation": int(metadata.get("generation", index)),
                "candidates": candidates,
                "design_fes": int(
                    metadata.get(
                        "cumulative_design_fes",
                        candidates * training_tasks * run_budget,
                    )
                ),
                "generation_training_cost": training_cost,
                "generation_normalized_training_gap": normalized_score,
            }
        )
    return result


def design_search(
    protocol: ExperimentProtocol,
    suite: str,
    function_id: int,
    *,
    artifact_root: Path,
    resume: bool,
) -> Path:
    from methods.search_settings import require_search_protocol

    require_search_protocol(protocol)
    method = "aol_search"
    output = _artifact_directory(artifact_root, suite, function_id)
    problem = make_ioh_problem(suite, function_id)
    instances = protocol.training_instances(suite, function_id)
    maximum_run_budget = max((int(item.budget) for item in instances))
    design = protocol.design
    candidate_budget = int(design["candidate_budget"])
    actual_fe_policy = design.get("candidate_budget_policy") == "actual_fe"
    search = design["search"]
    graph_semantics = str(search.get("graph_semantics", "legacy_pathway_v1")).lower()
    warm_start_config = dict(design.get("search_warm_start"))
    warm_start_entries = (
        warm_start_config.get("configurations", {})
        .get(suite, {})
        .get(str(function_id), [])
    )
    if (
        warm_start_entries
        and warm_start_config.get("mode") != "sparkle_paradiseo_incumbents"
    ):
        raise ValueError("Unknown Search v6.0 warm-start mode.")
    initial_designs = [
        fastga_initial_spec(
            entry["configuration"], suite, source_method=str(entry["method"])
        )
        for entry in warm_start_entries
    ]
    revision = _expected_search_revision(protocol)
    racing_alpha = float(search.get("racing_alpha", 0.05))
    racing_initial_blocks = int(
        search.get("racing_initial_blocks", max(1, round(len(instances) * 0.2)))
    )
    structure_racing_min_fraction = float(
        search.get("structure_racing_min_fraction", 0.5)
    )
    evaluation_workers = _search_evaluation_workers(design["workers"])
    training_scorer = make_taskwise_scorer(
        suite,
        function_id,
        instances,
        aggregation=str(design.get("training_score_aggregation", "task_mean")),
    )
    normalization = training_scorer.name
    outer_population = min(10, max(1, candidate_budget // 2))
    proposal_budget = candidate_budget - outer_population
    if proposal_budget < outer_population:
        outer_population = 1
        proposal_budget = candidate_budget - 1
    alg_fe = max(1, proposal_budget)
    designer = "search"
    alg_n = outer_population
    checkpoint = output / f"{method}-checkpoint"
    candidate_path = output / f"{method}-candidates.jsonl"
    started = time.perf_counter()
    (algorithms, trace) = autoopt(
        Mode="design",
        Designer=designer,
        Problem=problem,
        InstanceTrain=instances,
        InstanceTest=[],
        GraphSemantics=graph_semantics,
        AlgP=2,
        AlgQ=min(2, int(design["components_max"])),
        AlgN=alg_n,
        AlgFE=alg_fe,
        DesignFEs=protocol.design_fe_cap(suite) if actual_fe_policy else None,
        AlgRuns=1,
        Alpha=racing_alpha,
        RacingK=racing_initial_blocks,
        StructureRacingMinFraction=structure_racing_min_fraction,
        ProbN=int(protocol.suites[suite]["population_size"]),
        ProbFE=maximum_run_budget,
        InnerFE=min(500, maximum_run_budget),
        PopulationSizeSpace=list(
            range(4, int(search.get("population_size_max", 100)) + 1)
        ),
        OffspringSizeSpace=list(
            range(1, int(search.get("offspring_size_max", 100)) + 1)
        ),
        StructureMutationRate=float(search.get("structure_mutation_rate", 0.3)),
        SearchArchiveSize=1,
        SearchStagnation=int(search.get("stagnation_generations", 3)),
        SearchRestartFraction=float(search.get("restart_fraction", 0.5)),
        SearchImprovementRate=float(search.get("improvement_rate", 0.05)),
        SearchParameterActionProbability=float(
            search.get("parameter_action_probability", 0.5)
        ),
        SearchActionProbabilityGain=float(search.get("action_probability_gain", 0.3)),
        SearchActionRewardEWMA=float(search.get("action_reward_ewma", 0.2)),
        ParameterCMAInitialSigma=float(search.get("parameter_cma_initial_sigma", 1.0)),
        StreamEventBudgetMultiplier=search.get("stream_event_budget_multiplier", None),
        ParameterCMAOffspring=int(search.get("parameter_cma_offspring", 5)),
        StructureCandidatesPerAction=int(
            search.get("structure_candidates_per_action", 3)
        ),
        ParameterCMABlockGenerations=int(
            search.get("parameter_cma_block_generations", 3)
        ),
        PostStructureCMAGenerations=int(
            search.get("post_structure_cma_generations", 1)
        ),
        SearchMaxAttempts=int(search["max_attempts"]),
        SearchInitialDesigns=initial_designs,
        IncRate=float(search["improvement_rate"]),
        Evaluate=str(design["autooptlib_evaluation_policy"][method]),
        Compare=str(design["compare"]),
        Seed=protocol.seed + function_id + (0 if suite == "bbob" else 10000),
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
        EvalAffinity=False,
        EvalAdaptive=False,
        EvalConcurrencyProfilePath=artifact_root / "runtime-concurrency-profiles.json",
        CheckpointDir=checkpoint,
        CheckpointEvery=1,
        CheckpointSignature=f"{protocol.fingerprint()}:{revision}:{IOH_OPTIMUM_METADATA_REVISION}",
        Resume=resume,
        ResumeBudgetExtension=False,
        ResumeBudgetExtensionSourceAlgFE=None,
        ResumeBudgetExtensionSourceCheckpointSignature=None,
        CandidateLogPath=candidate_path,
        OutputDir=output / f"{method}-autoopt-output",
    )
    elapsed_seconds = time.perf_counter() - started
    candidate_ledger_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    artifact = output / f"{method}.json"
    save_algorithm(
        algorithms[0],
        artifact,
        metadata={
            "designer": designer,
            "suite": suite,
            "function_id": function_id,
            "protocol_fingerprint": protocol.fingerprint(),
            "candidate_budget": candidate_budget,
            "candidate_ledger_sha256": candidate_ledger_sha256,
            "training_tasks": len(instances),
            "search_hyperparameters": {
                **search,
                "outer_population": alg_n,
                "offspring_per_generation": alg_n,
                "proposal_budget": alg_fe,
                "selection": "novel_structure_mu_plus_lambda_with_parameter_local_replacement",
                "archive_selection_role": "record_only",
                "plateau_tie_break": "maximum_minimum_graph_distance",
                "forced_action_rules": "six_cma_generations_without_strict_improvement_force_structure;three_outer_stagnant_generations_restart_worst_half",
                "structure_search": "iterated_local_search",
                "implementation_revision": revision,
                "known_optimum_revision": IOH_OPTIMUM_METADATA_REVISION,
                "graph_semantics": graph_semantics,
                "racing_alpha": racing_alpha,
                "racing_initial_blocks": racing_initial_blocks,
                "structure_racing_min_fraction": structure_racing_min_fraction,
                "evaluation_policy": "exact",
                "staged_exact_selection": dict(None)
                if isinstance(None, dict)
                else None,
                "cma_structure_transfer_semantics": "semantic_coordinate_mean_covariance_paths_sigma_generation;new_coordinates_independent;stagnation_reset",
                "structure_novelty_semantics": "global_history_key_five_immediate_attempts_then_parameter_fallback",
                "action_budget_semantics": "structure_six_candidates_parameter_fifteen_candidates"
                if int(search.get("structure_candidates_per_action", 3)) == 1
                else "structure_eight_candidates_parameter_fifteen_candidates",
                "controller_credit_semantics": "generation_mean_relative_improvement_per_full_candidate_fe_equivalent"
                if actual_fe_policy
                else "generation_mean_relative_improvement_per_candidate",
                "probability_update_semantics": "frozen_within_generation_updated_after_generation",
                "candidate_diagnostic_semantics": "structure_baseline_and_post_structure_cma_split_no_control_effect",
                "structure_bootstrap_semantics": "one_structure_baseline_then_five_cma_offspring"
                if int(search.get("structure_candidates_per_action", 3)) == 1
                else "three_structure_baselines_then_five_cma_offspring",
            },
        },
    )
    curve = _curve_from_trace(
        list(trace),
        initial_candidates=outer_population if method == "aol_search" else 0,
        batch_size=outer_population if method == "aol_search" else 1,
        training_tasks=len(instances),
        run_budget=protocol.candidate_training_fe(suite) // len(instances),
        candidate_budget=None if actual_fe_policy else candidate_budget,
        scorer=training_scorer,
    )
    manifest = json.loads(
        (output / f"{method}-autoopt-output" / "experiment.json").read_text(
            encoding="utf-8"
        )
    )
    actual_design_fes = int(
        manifest["options"].get(
            "ActualDesignFEs", _candidate_summary(candidate_path)["actual_design_fes"]
        )
    )
    candidate_summary = _candidate_summary(candidate_path)
    actual_candidates = candidate_summary["different_candidates"]
    if actual_fe_policy:
        unused = protocol.design_fe_cap(suite) - actual_design_fes
        if not 0 <= unused < protocol.candidate_training_fe(suite):
            raise RuntimeError("FE search stopped early or exceeded its cap")
    elif actual_candidates != candidate_budget:
        raise RuntimeError(
            f"{method} produced {candidate_summary['different_candidates']} different algorithms; expected {candidate_budget}."
        )
    _require_complete_exact_ledger(
        candidate_summary,
        method=method,
        candidate_budget=actual_candidates if actual_fe_policy else candidate_budget,
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
            "budget_stop_policy": "actual_fe"
            if actual_fe_policy
            else "candidate_count",
            "training_fe_cap": protocol.design_fe_cap(suite),
            "unused_training_fes": protocol.design_fe_cap(suite) - actual_design_fes,
            "candidate_ledger_sha256": candidate_ledger_sha256,
            "budget_extension": None,
            "training_tasks": [vars(item) for item in instances],
            "search_hyperparameters": {
                **search,
                "outer_population": alg_n,
                "offspring_per_generation": alg_n,
                "proposal_budget": alg_fe,
                "selection": "novel_structure_mu_plus_lambda_with_parameter_local_replacement",
                "archive_selection_role": "record_only",
                "plateau_tie_break": "maximum_minimum_graph_distance",
                "forced_action_rules": "six_cma_generations_without_strict_improvement_force_structure;three_outer_stagnant_generations_restart_worst_half",
                "structure_search": "iterated_local_search",
                "implementation_revision": revision,
                "known_optimum_revision": IOH_OPTIMUM_METADATA_REVISION,
                "racing_alpha": racing_alpha,
                "racing_initial_blocks": racing_initial_blocks,
                "structure_racing_min_fraction": structure_racing_min_fraction,
                "evaluation_policy": "exact",
                "staged_exact_selection": dict(None)
                if isinstance(None, dict)
                else None,
                "cma_structure_transfer_semantics": "semantic_coordinate_mean_covariance_paths_sigma_generation;new_coordinates_independent;stagnation_reset",
                "structure_novelty_semantics": "global_history_key_five_immediate_attempts_then_parameter_fallback",
                "action_budget_semantics": "structure_six_candidates_parameter_fifteen_candidates"
                if int(search.get("structure_candidates_per_action", 3)) == 1
                else "structure_eight_candidates_parameter_fifteen_candidates",
                "controller_credit_semantics": "generation_mean_relative_improvement_per_full_candidate_fe_equivalent"
                if actual_fe_policy
                else "generation_mean_relative_improvement_per_candidate",
                "probability_update_semantics": "frozen_within_generation_updated_after_generation",
                "candidate_diagnostic_semantics": "structure_baseline_and_post_structure_cma_split_no_control_effect",
                "structure_bootstrap_semantics": "one_structure_baseline_then_five_cma_offspring"
                if int(search.get("structure_candidates_per_action", 3)) == 1
                else "three_structure_baselines_then_five_cma_offspring",
            },
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


def _design_learning_simple(
    protocol: ExperimentProtocol,
    suite: str,
    function_id: int,
    *,
    artifact_root: Path,
    resume: bool,
    source_model: Path | None = None,
) -> Path:
    """Train a grammar-masked policy and decode it once greedily."""
    if protocol.design["learning"].get(
        "update_method"
    ) != "archive_imitation" or protocol.design["learning"].get(
        "dual_origin_sampling", False
    ):
        raise ValueError(
            "Only the current single-policy archive-imitation Learning method is supported."
        )
    import torch
    from autooptlib.learning._checkpoint import atomic_torch_save
    from autooptlib.learning.archive_imitation import (
        ARCHIVE_IMITATION_IMPLEMENTATION,
        ArchiveImitationConfig,
        ArchiveImitationTrainer,
    )
    from autooptlib.learning.evaluator import AutoOptEvaluator, EvaluationConfig
    from autooptlib.learning.model import GeneratorConfig, LearningGenerator
    from autooptlib.utils.design._helpers import get_problem_type
    from autooptlib.utils.design._search_control import representation_key

    source_sha256 = (
        None
        if source_model is None
        else hashlib.sha256(Path(source_model).read_bytes()).hexdigest()
    )

    def verify_source_model() -> None:
        if (
            source_model is not None
            and hashlib.sha256(Path(source_model).read_bytes()).hexdigest()
            != source_sha256
        ):
            raise RuntimeError("Source model changed during target fine-tuning.")

    method = "aol_learning"
    output = _artifact_directory(artifact_root, suite, function_id)
    problem = make_ioh_problem(suite, function_id)
    instances = protocol.training_instances(suite, function_id)
    maximum_run_budget = max((int(item.budget) for item in instances))
    design = protocol.design
    candidate_budget = int(design["candidate_budget"])
    learning = design["learning"]
    if type(False) is not bool:
        raise ValueError("Learning dual_origin_sampling must be boolean.")
    float(learning.get("dual_origin_transfer_fraction", 0.5))
    training_greedy_candidate = learning.get("training_greedy_candidate", False)
    if type(training_greedy_candidate) is not bool:
        raise ValueError("Learning training_greedy_candidate must be boolean.")
    if training_greedy_candidate and False:
        raise ValueError(
            "Greedy training candidates require single-policy archive imitation."
        )
    training_scorer = make_taskwise_scorer(suite, function_id, instances)
    learning_graph_semantics = str(
        learning.get("graph_semantics", "legacy_pathway_v1")
    ).lower()
    if learning_graph_semantics not in {"legacy_pathway_v1", "stream_graph_v2"}:
        raise ValueError("Unsupported Learning graph semantics.")
    learning_pathways = int(
        learning.get(
            "max_pathways", 2 if learning_graph_semantics == "stream_graph_v2" else 1
        )
    )
    learning_search_stages = int(
        learning.get(
            "max_search_stages",
            2
            if learning_graph_semantics == "stream_graph_v2"
            else design["components_max"],
        )
    )
    if learning_graph_semantics == "stream_graph_v2" and (
        learning_pathways != 2 or learning_search_stages != 2
    ):
        raise ValueError(
            "Learning stream_graph_v2 experiments must use max_pathways=2 and max_search_stages=2 so their structural search space matches Search."
        )
    normalization = training_scorer.name
    seed = protocol.seed + 20000 + function_id + (0 if suite == "bbob" else 10000)
    started = time.perf_counter()
    torch.manual_seed(seed)
    evaluation = EvaluationConfig(
        population_size=int(protocol.suites[suite]["population_size"]),
        evaluations=maximum_run_budget,
        runs=1,
        inner_evaluations=min(500, maximum_run_budget),
        improvement_rate=float(design["search"]["improvement_rate"]),
        seed=protocol.task_seed(
            suite, function_id, 0, 0, phase="training", dimension=0
        ),
        coordinate_seeds=tuple(
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
        graph_semantics=learning_graph_semantics,
    )
    workers = design["workers"]
    worker_override = os.environ.get("AOL_LEARNING_EVAL_WORKERS")
    if worker_override is not None:
        workers = int(worker_override)
        if workers <= 0:
            raise ValueError("AOL_LEARNING_EVAL_WORKERS must be positive.")
    with AutoOptEvaluator(
        problem,
        instances,
        config=evaluation,
        pathways=learning_pathways,
        search_components=learning_search_stages,
        workers=workers,
        affinity=False,
        adaptive=worker_override is None,
        concurrency_profile_path=artifact_root / "runtime-concurrency-profiles.json",
        deduplicate=True,
        training_scorer=training_scorer,
    ) as evaluator:
        grammar = evaluator.codec.grammar
        generator_config = GeneratorConfig(
            model_dim=int(learning["model_dimension"]),
            heads=int(learning["attention_heads"]),
            layers=int(learning["layers"]),
            feedforward_dim=int(learning["feedforward_dimension"]),
            dropout=float(learning["dropout"]),
            max_length=int(learning["max_length"]),
        )
        problem_type = get_problem_type(evaluator.problems)

        def make_model(*, device: Any = "auto") -> Any:
            fresh = LearningGenerator(
                evaluator.codec.vocabulary,
                grammar,
                generator_config,
                device=os.environ.get("AOL_LEARNING_DEVICE", device),
            )
            fresh.set_problem_type(problem_type)
            return fresh

        def make_trainer(generator: Any) -> Any:
            return ArchiveImitationTrainer(
                generator,
                ArchiveImitationConfig(
                    learning_rate=learning["learning_rate"],
                    final_learning_rate=learning.get(
                        "final_learning_rate", learning["learning_rate"]
                    ),
                    anneal_steps=max(1, len(batch_schedule)),
                    adam_epsilon=learning["adam_epsilon"],
                    weight_decay=learning["weight_decay"],
                    gradient_norm=learning["gradient_norm"],
                    candidates=learning["batch_size"],
                    archive_size=learning["archive_size"],
                    update_epochs=learning.get("archive_updates", 1),
                    pairwise_weight=learning.get("pairwise_weight", 0.0),
                    entropy_coefficient=learning.get("entropy_coefficient", 0.01),
                    final_entropy_coefficient=learning.get(
                        "final_entropy_coefficient", 0.0
                    ),
                ),
            )

        model = make_model()
        if source_model is not None:
            rng_after_initialization = _torch_rng_payload(torch)
            (source, _) = LearningGenerator.load_checkpoint(
                source_model, map_location=next(model.parameters()).device
            )
            verify_source_model()
            if (
                source.vocabulary != model.vocabulary
                or source.grammar != model.grammar
                or source.config != model.config
            ):
                raise ValueError(
                    "Source model vocabulary, grammar or configuration is incompatible."
                )
            model.load_state_dict(source.state_dict())
            del source
            _restore_torch_rng(torch, rng_after_initialization)
        initial_policy_state = {
            name: value.detach().clone() for (name, value) in model.state_dict().items()
        }
        nominal_candidates = int(learning["batch_size"]) * int(
            learning["full_batches"]
        ) + int(learning["last_batch_size"])
        if nominal_candidates != candidate_budget:
            raise ValueError(
                "The registered learning batches must span the unique-candidate budget."
            )
        if int(learning["full_batches"]) < 1 or not 0 <= int(
            learning["last_batch_size"]
        ) < int(learning["batch_size"]):
            raise ValueError(
                "Archive imitation requires at least one full initialization batch, and a remainder smaller than batch_size."
            )
        checkpoints = set(protocol.validation_checkpoint_candidates())
        boundaries = sorted(checkpoints | {int(design["candidate_budget"])})
        batch_schedule = _learning_batch_schedule(
            candidate_budget, int(learning["batch_size"]), boundaries
        )
        trainer = make_trainer(model)

        def model_state_is_finite(candidate_model) -> bool:
            """Cover parameters and persistent buffers in checkpoint validation."""
            return all(
                (
                    isinstance(value, torch.Tensor)
                    and bool(torch.isfinite(value).all())
                    for value in candidate_model.state_dict().values()
                )
            )

        checkpoint = output / f"{method}-training.pt"
        history_path = output / f"{method}-history.jsonl"
        candidate_path = output / f"{method}-candidates.jsonl"
        curve: list[dict[str, Any]] = []
        completed_steps = 0
        sampled_sequences: set[tuple[int, ...]] = set()
        seen_graphs: set[str] = set()
        graph_costs: dict[str, float] = {}
        duplicate_sequence_samples = 0
        duplicate_graph_samples = 0
        evaluation_cache_hits = 0
        sampled_candidates = 0
        candidate_count = 0
        policy_restart_count = 0
        raw_sample_limit = max(1000, int(learning["batch_size"]) * 100)
        rolling_window_size = max(256, int(learning["batch_size"]) * 32)
        minimum_unique_rate = max(0.01, int(learning["batch_size"]) / raw_sample_limit)
        unique_outcomes: deque[int] = deque(maxlen=rolling_window_size)

        def export_policy_checkpoint(
            count: int,
            *,
            policy_updates_through_candidates: int,
            pending_candidates_at_checkpoint: int = 0,
        ) -> None:
            if count not in checkpoints:
                return
            model.eval()
            with torch.no_grad():
                generated = model.generate(candidates=1, greedy=True)
            checkpoint_sequence = np.asarray(
                grammar.normalize(generated.sequences[0].detach().cpu().numpy()),
                dtype=int,
            )
            checkpoint_algorithm = evaluator.codec.decode(checkpoint_sequence)
            checkpoint_root = (
                artifact_root.parent
                / "checkpoints"
                / f"candidates-{count:04d}"
                / "artifacts"
            )
            checkpoint_output = _artifact_directory(checkpoint_root, suite, function_id)
            checkpoint_artifact = checkpoint_output / f"{method}.json"
            training_score = (
                float(
                    curve[-1].get(
                        "mean_cost",
                        curve[-1].get("archive_mean_task_cost", float("inf")),
                    )
                )
                if curve
                else float("inf")
            )
            save_algorithm(
                checkpoint_algorithm,
                checkpoint_artifact,
                metadata={
                    "designer": "learning",
                    "suite": suite,
                    "function_id": function_id,
                    "protocol_fingerprint": protocol.fingerprint(),
                    "source_model_sha256": source_sha256,
                    "budget_checkpoint_candidates": count,
                    "budget_checkpoint_fe": count
                    * protocol.candidate_training_fe(suite),
                    "checkpoint_training_score": training_score,
                    "checkpoint_training_score_kind": "archive_mean_taskwise_normalized_gap",
                    "policy_updates_through_candidates": policy_updates_through_candidates,
                    "pending_candidates_at_checkpoint": pending_candidates_at_checkpoint,
                    "checkpoint_update_semantics": "carry_partial_batch_without_checkpoint_update",
                    "inference_mode": "greedy_policy",
                    "learning_sequence": [int(value) for value in checkpoint_sequence],
                },
            )
            model_path = checkpoint_output / f"{method}-model.pt"
            temporary_model = model_path.with_suffix(model_path.suffix + ".tmp")
            model.save_checkpoint(
                temporary_model,
                suite=suite,
                function_id=function_id,
                protocol_fingerprint=protocol.fingerprint(),
                source_model_sha256=source_sha256,
                budget_checkpoint_candidates=count,
                policy_updates_through_candidates=policy_updates_through_candidates,
                pending_candidates_at_checkpoint=pending_candidates_at_checkpoint,
                checkpoint_update_semantics="carry_partial_batch_without_checkpoint_update",
                checkpoint_training_score=training_score,
                improvement_rate=evaluation.improvement_rate,
                learning_update_method="archive_imitation",
                learning_update_config=asdict(trainer.config),
                ppo_config=None,
                archive_imitation_implementation=ARCHIVE_IMITATION_IMPLEMENTATION,
                dual_origin_sampling=False,
                dual_origin_transfer_fraction=None,
                dual_origin_fresh_state_dict=None,
                dual_origin_inference=None,
                risk_seeking_implementation=None,
            )
            temporary_model.replace(model_path)

        if resume and checkpoint.exists():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            expected_schema_version = 13
            if not isinstance(payload, dict):
                raise ValueError("Learning checkpoint payload must be a dictionary.")
            schema_version = payload.get("schema_version", -1)
            schema_version_matches = (
                type(schema_version) is int
                and schema_version == expected_schema_version
            )
            if (
                payload.get("schema") != "autooptlib.paper-learning-checkpoint"
                or not schema_version_matches
            ):
                raise ValueError(
                    "Learning checkpoint uses a different training-state schema; restart this design task from candidate 0."
                )
            if payload.get("update_method", "ppo") != "archive_imitation":
                raise ValueError(
                    "Learning checkpoint update method does not match the protocol."
                )
            if payload.get("protocol_fingerprint") != protocol.fingerprint():
                raise ValueError(
                    "Refusing to resume a learning checkpoint created under a different experiment protocol."
                )
            if payload.get("source_model_sha256") != source_sha256:
                raise ValueError("Target checkpoint source model does not match.")
            if payload.get("dual_origin_sampling", False) is not False:
                raise ValueError(
                    "Learning checkpoint dual-origin mode does not match the protocol."
                )

            def checkpoint_integer(name: str, *, minimum: int = 0) -> int:
                value = payload.get(name)
                if type(value) is not int or value < minimum:
                    raise ValueError(
                        f"Archive-imitation checkpoint {name} must be an integer at least {minimum}."
                    )
                return value

            steps = checkpoint_integer("steps")
            completed_steps = checkpoint_integer("completed_steps")
            candidate_count = checkpoint_integer("candidate_count")
            duplicate_sequence_samples = checkpoint_integer(
                "duplicate_sequence_samples"
            )
            duplicate_graph_samples = checkpoint_integer("duplicate_graph_samples")
            evaluation_cache_hits = checkpoint_integer("evaluation_cache_hits")
            sampled_candidates = checkpoint_integer("sampled_candidates")
            policy_restart_count = checkpoint_integer("policy_restart_count")
            actual_design_fes = checkpoint_integer("actual_design_fes")
            if completed_steps > len(batch_schedule):
                raise ValueError(
                    "Archive-imitation checkpoint exceeds the batch schedule."
                )
            expected_candidate_count = sum(batch_schedule[:completed_steps])
            if candidate_count != expected_candidate_count:
                raise ValueError(
                    "Learning checkpoint candidate count does not match its successful update count."
                )
            if sampled_candidates < candidate_count:
                raise ValueError(
                    "Archive-imitation checkpoint sampled fewer candidates than it evaluated."
                )
            raw_curve = payload.get("curve")
            if (
                not isinstance(raw_curve, list)
                or len(raw_curve) != completed_steps
                or any((not isinstance(row, dict) for row in raw_curve))
            ):
                raise ValueError(
                    "Archive-imitation checkpoint curve does not match its completed batch count."
                )
            curve = list(raw_curve)
            cumulative = 0
            for index, (row, batch) in enumerate(zip(curve, batch_schedule), 1):
                cumulative += batch
                if (
                    type(row.get("generation")) is not int
                    or row["generation"] != index
                    or type(row.get("candidates")) is not int
                    or (row["candidates"] != cumulative)
                ):
                    raise ValueError(
                        "Archive-imitation checkpoint curve is not contiguous."
                    )
            raw_sequences = payload.get("sampled_sequences")
            if not isinstance(raw_sequences, list):
                raise ValueError(
                    "Archive-imitation sampled-sequence history must be a list."
                )
            sampled_sequences = set()
            for raw_sequence in raw_sequences:
                if not isinstance(raw_sequence, list) or any(
                    (type(token) is not int for token in raw_sequence)
                ):
                    raise ValueError(
                        "Archive-imitation sampled sequences must contain integer token lists."
                    )
                sequence = tuple(grammar.validate(raw_sequence))
                if sequence != tuple(raw_sequence):
                    raise ValueError(
                        "Archive-imitation sampled sequence is not normalized."
                    )
                if sequence in sampled_sequences:
                    raise ValueError(
                        "Archive-imitation sampled-sequence history has duplicates."
                    )
                sampled_sequences.add(sequence)
            raw_graph_costs = payload.get("graph_costs")
            if not isinstance(raw_graph_costs, dict):
                raise ValueError(
                    "Archive-imitation graph-cost history must be a dictionary."
                )
            graph_costs = {}
            for key, value in raw_graph_costs.items():
                if (
                    not isinstance(key, str)
                    or not key
                    or isinstance(value, (bool, np.bool_))
                    or (not isinstance(value, (int, float, np.integer, np.floating)))
                    or (not bool(np.isfinite(value)))
                ):
                    raise ValueError(
                        "Archive-imitation graph-cost history is malformed."
                    )
                graph_costs[key] = float(value)
            if len(graph_costs) != candidate_count:
                raise ValueError(
                    "Archive-imitation graph-cost history does not match its candidate count."
                )
            if len(sampled_sequences) < candidate_count:
                raise ValueError(
                    "Archive-imitation sequence history cannot cover its evaluated candidates."
                )
            raw_outcomes = payload.get("unique_outcomes")
            if (
                not isinstance(raw_outcomes, list)
                or len(raw_outcomes) > rolling_window_size
                or any(
                    (
                        type(value) is not int or value not in {0, 1}
                        for value in raw_outcomes
                    )
                )
            ):
                raise ValueError(
                    "Archive-imitation rolling uniqueness history is malformed."
                )
            unique_outcomes = deque(raw_outcomes, maxlen=rolling_window_size)
            rng_payload = payload.get("torch_rng_state")
            if (
                not isinstance(rng_payload, dict)
                or not isinstance(rng_payload.get("cpu"), torch.Tensor)
                or rng_payload["cpu"].dtype != torch.uint8
                or (rng_payload["cpu"].ndim != 1)
                or (rng_payload["cpu"].numel() != torch.get_rng_state().numel())
            ):
                raise ValueError("Archive-imitation checkpoint RNG state is malformed.")
            runtime_statistics = payload.get("runtime_statistics")
            if runtime_statistics is not None and (
                not isinstance(runtime_statistics, dict)
            ):
                raise ValueError(
                    "Archive-imitation runtime statistics must be a dictionary or null."
                )
            model.load_state_dict(payload["model_state_dict"])
            trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
            trainer.steps = steps
            trainer.load_archive_state(payload.get("archive_state", []))
            restored_archive = trainer.archive_state()
            if any(
                (
                    record["structure_key"] not in graph_costs
                    or representation_key(evaluator.codec.decode(record["sequence"]))
                    != record["structure_key"]
                    for record in restored_archive
                )
            ):
                raise ValueError(
                    "Archive-imitation checkpoint archive identity disagrees with its sequence or evaluated graph history."
                )
            if not model_state_is_finite(model):
                raise ValueError(
                    "Archive-imitation checkpoint model contains non-finite values."
                )
            if not trainer._optimizer_state_is_finite():
                raise ValueError(
                    "Archive-imitation checkpoint optimizer contains non-finite values."
                )
            if trainer.steps != completed_steps:
                raise ValueError(
                    "Learning checkpoint trainer step count does not match its completed batch count."
                )
            if trainer.archive_size != min(
                trainer.config.archive_size, candidate_count
            ):
                raise ValueError(
                    "Archive-imitation checkpoint does not contain a complete elite archive."
                )
            seen_graphs = set(graph_costs)
            expected_candidate_count = sum(batch_schedule[:completed_steps])
            if candidate_count != expected_candidate_count:
                raise ValueError(
                    "Learning checkpoint candidate count does not match its successful update count."
                )
            if not history_path.exists():
                raise ValueError(
                    "Archive-imitation checkpoint has no append-only history."
                )
            history_lines = [
                line
                for line in history_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            if len(history_lines) < completed_steps:
                raise ValueError(
                    "Archive-imitation checkpoint history is shorter than its completed batch count."
                )
            try:
                persisted_curve = [
                    json.loads(line) for line in history_lines[:completed_steps]
                ]
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Archive-imitation append-only history is malformed."
                ) from exc
            if persisted_curve != curve:
                raise ValueError(
                    "Archive-imitation checkpoint curve disagrees with its append-only history."
                )
            trim_candidate_records(candidate_path, candidate_count)
            _trim_jsonl_records(history_path, completed_steps)
            evaluator.restore_runtime_state(
                actual_design_fes, payload.get("runtime_statistics")
            )
            _restore_torch_rng(torch, rng_payload)
            export_policy_checkpoint(
                candidate_count, policy_updates_through_candidates=candidate_count
            )
        elif candidate_path.exists():
            candidate_path.unlink()
            if history_path.exists():
                history_path.unlink()
        while candidate_count < candidate_budget:
            if completed_steps >= len(batch_schedule):
                raise RuntimeError(
                    "Learning exhausted its update schedule before its unique-candidate budget."
                )
            batch_size = batch_schedule[completed_steps]
            if candidate_count + batch_size > candidate_budget:
                raise RuntimeError("Learning batch exceeds the candidate budget.")
            sampled_sequences_before_batch = set(sampled_sequences)
            accepted_tensors: list[Any] = []
            accepted_sequences: list[tuple[int, ...]] = []
            accepted_graphs: list[str] = []
            accepted_origins: list[str] = []
            batch_sequence_keys: set[tuple[int, ...]] = set()
            batch_graph_keys: set[str] = set()
            raw_samples_this_update = 0
            raw_samples_since_restart = 0
            restarts_this_update = 0
            greedy_pending = training_greedy_candidate
            greedy_graph_key = None
            greedy_cache_hit = False
            while len(accepted_sequences) < batch_size:
                sampling_model = model
                sampling_origin = "policy"
                remaining = batch_size - len(accepted_sequences)
                sampling_model.eval()
                with torch.no_grad():
                    is_greedy = greedy_pending
                    if is_greedy:
                        generated = sampling_model.generate(candidates=1, greedy=True)
                        sampling_origin = "greedy"
                        greedy_pending = False
                    else:
                        generated = sampling_model.generate(candidates=remaining)
                    generated_tensor = generated.sequences
                    sequence_keys = [
                        tuple(grammar.normalize(row))
                        for row in generated_tensor.detach().cpu().numpy()
                    ]
                    graph_keys = [
                        representation_key(evaluator.codec.decode(sequence_key))
                        for sequence_key in sequence_keys
                    ]
                    if is_greedy:
                        greedy_graph_key = graph_keys[0]
                        greedy_cache_hit = greedy_graph_key in graph_costs
                (accepted_indices, sequence_duplicates, graph_duplicates, hits) = (
                    _unique_graph_sample_indices(
                        sequence_keys,
                        graph_keys,
                        historical_sequences=sampled_sequences,
                        historical_graphs=set(graph_costs),
                        batch_sequences=batch_sequence_keys,
                        batch_graphs=batch_graph_keys,
                    )
                )
                accepted_index_set = set(accepted_indices)
                duplicate_sequence_samples += sequence_duplicates
                duplicate_graph_samples += graph_duplicates
                evaluation_cache_hits += hits
                sampled_candidates += len(sequence_keys)
                raw_samples_this_update += len(sequence_keys)
                raw_samples_since_restart += len(sequence_keys)
                unique_outcomes.extend(
                    (
                        1 if index in accepted_index_set else 0
                        for index in range(len(sequence_keys))
                    )
                )
                sampled_sequences.update(sequence_keys)
                batch_sequence_keys.update(sequence_keys)
                for index in accepted_indices:
                    accepted_tensors.append(generated_tensor[index : index + 1])
                    accepted_sequences.append(sequence_keys[index])
                    accepted_graphs.append(graph_keys[index])
                    accepted_origins.append(sampling_origin)
                    batch_graph_keys.add(graph_keys[index])
                batch_complete = len(accepted_sequences) == batch_size
                rolling_collapsed = (
                    len(unique_outcomes) == rolling_window_size
                    and _rolling_unique_rate(unique_outcomes) < minimum_unique_rate
                )
                hit_hard_limit = raw_samples_since_restart >= raw_sample_limit
                if not batch_complete and (rolling_collapsed or hit_hard_limit):
                    device = next(model.parameters()).device
                    if restarts_this_update >= 10:
                        raise RuntimeError(
                            "Learning cannot fill a unique batch after 10 policy recoveries."
                        )
                    policy_restart_count += 1
                    torch.manual_seed(seed + policy_restart_count)
                    model = make_model(device=device)
                    model.load_state_dict(initial_policy_state)
                    retained_training_state = trainer.archive_state()
                    trainer = make_trainer(model)
                    trainer.load_archive_state(retained_training_state)
                    trainer.steps = completed_steps
                    sampled_sequences = set(sampled_sequences_before_batch)
                    accepted_tensors.clear()
                    accepted_sequences.clear()
                    accepted_graphs.clear()
                    accepted_origins.clear()
                    batch_sequence_keys.clear()
                    batch_graph_keys.clear()
                    unique_outcomes.clear()
                    raw_samples_since_restart = 0
                    restarts_this_update += 1
                    greedy_pending = training_greedy_candidate
                    greedy_graph_key = None
                    greedy_cache_hit = False
            unique_tensor = _pad_learning_sequence_tensors(
                torch, accepted_tensors, end_index=model.end_index
            )
            if unique_tensor.shape[0] != batch_size:
                raise RuntimeError("Learning did not fill a unique-graph batch.")
            with torch.no_grad():
                policy_entropy = float(model.entropy(unique_tensor).mean().cpu())
            (new_costs, task_performances) = evaluator.evaluate_many(
                accepted_sequences, workers=workers
            )
            if not np.all(np.isfinite(new_costs)):
                raise FloatingPointError(
                    "Learning candidate evaluation produced a non-finite score."
                )
            new_designs = evaluator.last_designs(accepted_sequences)
            if len(new_designs) != batch_size:
                raise RuntimeError(
                    "Learning evaluator did not return one design per unique graph."
                )
            for sequence_key, graph_key, origin, cost, candidate in zip(
                accepted_sequences,
                accepted_graphs,
                accepted_origins,
                new_costs,
                new_designs,
            ):
                metadata = dict(getattr(candidate, "metadata", {}))
                metadata.update(
                    {
                        "learning_sequence": list(sequence_key),
                        "learning_stage": "training",
                        "learning_graph_key": graph_key,
                        "training_score": float(cost),
                        "evaluation_cache_hit": False,
                        "learning_sampling_origin": origin,
                    }
                )
                candidate.metadata = metadata
                graph_costs[graph_key] = float(cost)
                seen_graphs.add(graph_key)

            def evaluate_batch(_sequences):
                return new_costs

            grouped_task_costs = _learning_grouped_task_costs(
                training_scorer, task_performances, instances
            )
            for checkpoint_count in sorted(checkpoints):
                if candidate_count < checkpoint_count < candidate_count + batch_size:
                    export_policy_checkpoint(
                        checkpoint_count,
                        policy_updates_through_candidates=candidate_count,
                        pending_candidates_at_checkpoint=checkpoint_count
                        - candidate_count,
                    )
            progress = (
                float(completed_steps > 0)
                if len(batch_schedule) <= 2
                else min(max(completed_steps - 1, 0) / (len(batch_schedule) - 2), 1.0)
            )
            if grouped_task_costs is None:
                raise RuntimeError("Archive imitation requires grouped task costs.")
            metrics = trainer.step(
                unique_tensor,
                grouped_task_costs,
                accepted_graphs,
                annealing_progress=progress,
            )
            append_candidate_records(
                candidate_path,
                new_designs,
                start=candidate_count + 1,
                designer="learning",
            )
            candidate_count += batch_size
            completed_steps += 1
            best_new = int(np.argmin(new_costs))
            raw_training_cost = float(
                np.mean(np.asarray(new_designs[best_new].performance, dtype=float))
            )
            generation_score = float(new_costs[best_new])
            greedy_metrics = (
                {
                    "training_greedy_graph_key": greedy_graph_key,
                    "training_greedy_cache_hit": greedy_cache_hit,
                    "training_greedy_cost": graph_costs[greedy_graph_key],
                    "greedy_new_candidates": accepted_origins.count("greedy"),
                }
                if training_greedy_candidate
                else {}
            )
            record = {
                **greedy_metrics,
                "generation": completed_steps,
                "batch": completed_steps,
                "batch_size": batch_size,
                "new_candidates": batch_size,
                "candidates": candidate_count,
                "sampled_candidates": sampled_candidates,
                "raw_samples_this_update": raw_samples_this_update,
                "dual_origin_sampling": False,
                "transfer_candidates_this_update": 0,
                "fresh_candidates_this_update": 0,
                "transfer_raw_samples_this_update": 0,
                "fresh_raw_samples_this_update": 0,
                "batch_unique_rate": batch_size / raw_samples_this_update,
                "rolling_unique_rate": _rolling_unique_rate(unique_outcomes),
                "rolling_unique_window": len(unique_outcomes),
                "minimum_unique_rate": minimum_unique_rate,
                "design_fes": evaluator.actual_design_fes,
                "generation_training_cost": raw_training_cost,
                "generation_normalized_training_gap": generation_score,
                "batch_unique_sequences": len(set(accepted_sequences)),
                "batch_unique_graphs": len(set(accepted_graphs)),
                "cumulative_unique_sequences": len(sampled_sequences),
                "cumulative_unique_graphs": len(seen_graphs),
                "duplicate_sequence_samples": duplicate_sequence_samples,
                "duplicate_graph_samples": duplicate_graph_samples,
                "policy_entropy": policy_entropy,
                "policy_restarted": bool(restarts_this_update),
                "policy_restarts_this_update": restarts_this_update,
                "policy_restart_count": policy_restart_count,
                "restart_initialization": "task_initial_weights",
                "raw_sample_limit": raw_sample_limit,
                "stage": "training",
                **metrics,
            }
            curve.append(record)
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            atomic_torch_save(
                {
                    "schema": "autooptlib.paper-learning-checkpoint",
                    "schema_version": 13,
                    "update_method": "archive_imitation",
                    "dual_origin_sampling": False,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "auxiliary_model_state_dict": None,
                    "auxiliary_optimizer_state_dict": None,
                    "auxiliary_steps": None,
                    "auxiliary_archive_state": [],
                    "steps": trainer.steps,
                    "baseline": None,
                    "elite_replay_state": [],
                    "archive_state": trainer.archive_state(),
                    "completed_steps": completed_steps,
                    "candidate_count": candidate_count,
                    "protocol_fingerprint": protocol.fingerprint(),
                    "source_model_sha256": source_sha256,
                    "improvement_rate": evaluation.improvement_rate,
                    "curve": curve,
                    "sampled_sequences": [
                        list(sequence) for sequence in sampled_sequences
                    ],
                    "graph_costs": graph_costs,
                    "duplicate_sequence_samples": duplicate_sequence_samples,
                    "duplicate_graph_samples": duplicate_graph_samples,
                    "evaluation_cache_hits": evaluation_cache_hits,
                    "sampled_candidates": sampled_candidates,
                    "unique_outcomes": list(unique_outcomes),
                    "policy_restart_count": policy_restart_count,
                    "actual_design_fes": evaluator.actual_design_fes,
                    "runtime_statistics": evaluator.runtime_statistics,
                    "torch_rng_state": _torch_rng_payload(torch),
                },
                checkpoint,
            )
            export_policy_checkpoint(
                candidate_count, policy_updates_through_candidates=candidate_count
            )
        model.eval()
        with torch.no_grad():
            inferred = model.generate(candidates=1, greedy=True)
        sequence = np.asarray(
            grammar.normalize(inferred.sequences[0].detach().cpu().numpy()), dtype=int
        )
        algorithm = evaluator.codec.decode(sequence)
        artifact = output / f"{method}.json"
        save_algorithm(
            algorithm,
            artifact,
            metadata={
                "designer": "learning",
                "learning_sequence": [int(value) for value in sequence],
                "suite": suite,
                "function_id": function_id,
                "inference_mode": "greedy_policy",
                "protocol_fingerprint": protocol.fingerprint(),
                "source_model_sha256": source_sha256,
            },
        )
        final_model = output / f"{method}-model.pt"
        temporary_model = final_model.with_suffix(final_model.suffix + ".tmp")
        model.save_checkpoint(
            temporary_model,
            suite=suite,
            function_id=function_id,
            protocol_fingerprint=protocol.fingerprint(),
            source_model_sha256=source_sha256,
            improvement_rate=evaluation.improvement_rate,
            learning_update_method="archive_imitation",
            learning_update_config=asdict(trainer.config),
            ppo_config=None,
            archive_imitation_implementation=ARCHIVE_IMITATION_IMPLEMENTATION,
            dual_origin_sampling=False,
            dual_origin_transfer_fraction=None,
            dual_origin_fresh_state_dict=None,
            dual_origin_inference=None,
            risk_seeking_implementation=None,
        )
        temporary_model.replace(final_model)
        model.set_problem_type(None)
        verify_source_model()
        runtime_statistics = evaluator.runtime_statistics
        actual_design_fes = evaluator.actual_design_fes
    elapsed_seconds = time.perf_counter() - started
    candidate_summary = _candidate_summary(candidate_path)
    if candidate_summary["different_candidates"] != candidate_budget:
        raise RuntimeError(
            f"{method} produced {candidate_summary['different_candidates']} different algorithms; expected {candidate_budget}."
        )
    _require_complete_exact_ledger(
        candidate_summary,
        method=method,
        candidate_budget=int(design["candidate_budget"]),
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
            "source_model_sha256": source_sha256,
            "candidate_budget": int(design["candidate_budget"]),
            "training_tasks": [vars(item) for item in instances],
            "learning_hyperparameters": {
                **learning,
                "implementation_revision": LEARNING_IMPLEMENTATION_REVISION,
                "final_learning_rate": trainer.config.final_learning_rate,
                "network_implementation": "compact_aldes_decoder",
                "parameter_initialization": "kaiming_uniform",
                "parameter_encoding": "flat_10",
                "parameter_levels": 10,
                "max_search_components": int(design["components_max"]),
                "improvement_rate": evaluation.improvement_rate,
                "inference_mode": "greedy_policy",
                "update_method": "archive_imitation",
                "batch_unit": "unique_graph",
                "duplicates_in_update_loss": False,
                "learning_rate_schedule": "successful_learning_updates",
                "collapse_detection": "rolling_unique_rate",
                "archive_imitation_implementation": ARCHIVE_IMITATION_IMPLEMENTATION,
                "archive_size": trainer.config.archive_size,
                "archive_selection": "global_mu_plus_lambda_mean_taskwise_normalized_cost",
                "archive_loss": "equal_weight_mean_per_token_negative_log_likelihood",
                "dual_origin_active": False,
                "dual_origin_transfer_fraction": None,
                "dual_origin_batch_allocation": None,
                "dual_origin_archive": None,
                "dual_origin_policy_updates": None,
                "first_batch_semantics": "initialize_archive_without_gradient",
                "advantage_method": None,
                "risk_seeking_fraction": None,
                "risk_seeking_objective": None,
                "risk_seeking_implementation": None,
                "entropy_coefficient": trainer.config.entropy_coefficient,
                "final_entropy_coefficient": trainer.config.final_entropy_coefficient,
                "elite_replay_weight": None,
                "elite_replay_fraction": None,
                "elite_replay_capacity": None,
                "elite_replay_min_candidates": None,
                "elite_replay_max_per_structure": None,
                "elite_replay_ranking": None,
                "elite_replay_loss": None,
                "advantage_task_grouping": None,
                "advantage_confidence_level": None,
            },
            "artifact": str(artifact.resolve()),
            "inference_mode": "greedy_policy",
            "design_ledger": {
                **candidate_summary,
                "actual_design_fes": actual_design_fes,
                "evaluation_cache_hits": evaluation_cache_hits,
                "sampled_candidates": sampled_candidates,
                "duplicate_sequence_samples": duplicate_sequence_samples,
                "duplicate_graph_samples": duplicate_graph_samples,
                "policy_restart_count": policy_restart_count,
                "rolling_unique_window_size": rolling_window_size,
                "minimum_unique_rate": minimum_unique_rate,
                "raw_sample_limit": raw_sample_limit,
                "elapsed_seconds_this_invocation": elapsed_seconds,
            },
            "normalization": normalization,
            "runtime_statistics": runtime_statistics,
            "curve": curve,
        },
    )
    return artifact


def design_learning(
    protocol: ExperimentProtocol,
    suite: str,
    function_id: int,
    *,
    artifact_root: Path,
    resume: bool,
    source_model: Path | None = None,
) -> Path:
    """Run the retained Learning-V1 designer."""
    return _design_learning_simple(
        protocol,
        suite,
        function_id,
        artifact_root=artifact_root,
        resume=resume,
        source_model=source_model,
    )


def design_one(
    protocol: ExperimentProtocol,
    suite: str,
    function_id: int,
    method: str,
    *,
    artifact_root: Path,
    resume: bool,
    source_model: Path | None = None,
) -> Path:
    if method == "aol_search":
        from methods.search_settings import require_search_protocol

        require_search_protocol(protocol)
    if method not in {"aol_search", "aol_learning", "aol_random"}:
        raise ValueError("Expected aol_search, aol_learning, or aol_random.")
    if source_model is not None:
        if method != "aol_learning":
            raise ValueError("source_model is only supported for Learning.")
        if Path(source_model).resolve().is_relative_to(artifact_root.resolve()) or Path(
            source_model
        ).resolve().is_relative_to(artifact_root.parent.resolve() / "checkpoints"):
            raise ValueError("Source model must be outside the target run directory.")
    source_sha256 = (
        None
        if source_model is None
        else hashlib.sha256(Path(source_model).read_bytes()).hexdigest()
    )
    output = artifact_root / suite / f"f{function_id:02d}"
    artifact = output / f"{method}.json"
    design_record = output / f"{method}-design.json"
    if resume and artifact.exists() and design_record.exists():
        payload = json.loads(design_record.read_text(encoding="utf-8"))
        expected = {
            "schema": "autooptlib.paper-design",
            "method": method,
            "suite": suite,
            "function_id": function_id,
            "protocol_fingerprint": protocol.fingerprint(),
            "candidate_budget": int(protocol.design["candidate_budget"]),
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError(
                    f"Refusing to resume stale design artifact {design_record}: field {name!r} is not {value!r}."
                )
        if (
            method == "aol_learning"
            and payload.get("learning_hyperparameters", {}).get(
                "implementation_revision"
            )
            != LEARNING_IMPLEMENTATION_REVISION
        ):
            raise ValueError(
                "Refusing to reuse a stale Learning implementation; retrain with the audited codec."
            )
        if method == "aol_search":
            expected_revision = _expected_search_revision(protocol)
            if (
                payload.get("search_hyperparameters", {}).get("implementation_revision")
                != expected_revision
            ):
                raise ValueError(
                    "Refusing to reuse a stale Search implementation; rerun design."
                )
        if (
            method == "aol_learning"
            and payload.get("source_model_sha256") != source_sha256
        ):
            raise ValueError("Target artifact source model does not match.")
        ledger = payload.get("design_ledger")
        curve = payload.get("curve")
        actual_fes = (
            int(ledger.get("actual_design_fes", -1)) if isinstance(ledger, dict) else -1
        )
        exact_task_count = int(protocol.design["candidate_budget"]) * len(
            protocol.training_instances(suite, function_id)
        )
        racing = (
            method == "aol_search"
            and protocol.design.get("autooptlib_evaluation_policy", {}).get(method)
            == "racing"
        )
        actual_fe_policy = (
            method == "aol_search"
            and protocol.design.get("candidate_budget_policy") == "actual_fe"
        )
        expected_candidates = (
            int(ledger.get("generated_candidates", -1))
            if (racing or actual_fe_policy) and isinstance(ledger, dict)
            else int(protocol.design["candidate_budget"])
        )
        if actual_fe_policy:
            exact_task_count = expected_candidates * len(
                protocol.training_instances(suite, function_id)
            )
            if (
                not 0
                <= protocol.design_fe_cap(suite) - actual_fes
                < protocol.candidate_training_fe(suite)
            ):
                raise ValueError(
                    "Cannot resume a terminal FE checkpoint with unspent candidate budget"
                )
        incomplete_exact_ledger = (
            not racing
            and method in {"aol_search", "aol_learning", "aol_random"}
            and isinstance(ledger, dict)
            and (int(ledger.get("completed_candidate_tasks", -1)) != exact_task_count)
        )
        if (
            not isinstance(ledger, dict)
            or expected_candidates <= 0
            or int(ledger.get("generated_candidates", -1)) != expected_candidates
            or (int(ledger.get("different_candidates", -1)) != expected_candidates)
            or (int(ledger.get("failed_candidates", -1)) != 0)
            or incomplete_exact_ledger
            or (not 0 < actual_fes <= protocol.design_fe_cap(suite))
            or (not isinstance(curve, list))
            or (not curve)
            or (int(curve[-1].get("candidates", -1)) != expected_candidates)
            or (int(curve[-1].get("design_fes", -1)) != actual_fes)
        ):
            raise ValueError(
                f"Refusing to resume incomplete design artifact: {design_record}."
            )
        _export_budget_checkpoints(
            protocol, suite, function_id, method, artifact_root=artifact_root
        )
        return artifact
    if method == "aol_learning":
        result = design_learning(
            protocol,
            suite,
            function_id,
            artifact_root=artifact_root,
            resume=resume,
            source_model=source_model,
        )
    elif method == "aol_random":
        from comparisons.random_design.workflow import design_random

        result = design_random(
            protocol, suite, function_id, artifact_root=artifact_root, resume=resume
        )
    else:
        result = design_search(
            protocol, suite, function_id, artifact_root=artifact_root, resume=resume
        )
    _export_budget_checkpoints(
        protocol, suite, function_id, method, artifact_root=artifact_root
    )
    return result


__all__ = [
    "_checkpoint_candidate_counts",
    "_export_budget_checkpoints",
    "design_one",
]
