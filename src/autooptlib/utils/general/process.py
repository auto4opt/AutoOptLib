"""Python translation of Utilities/General/Process.m."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import pickle
import warnings
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List, Sequence, Tuple

import numpy as np

from autooptlib.problems.base import validate_constructed_problems

from ...components import get_component
from ...runtime import DesignEvaluationRuntime
from .._constraint_order import CONSTRAINT_ORDER_REVISION
from ..design import Design
from ..design._helpers import ensure_rng, get_flex
from ..design._population import (
    configuration_is_valid,
    sample_configuration,
    set_configuration,
)
from ..design._search_actions import (
    PARAMETER,
    STRUCTURE,
    STRUCTURE_NOVELTY_ATTEMPTS,
    ActionSelector,
    StructureNoveltyExhausted,
    controller_action_credit,
    execute_action,
    parameter_action_evaluation_cost,
    semantic_action_credits,
    structure_action_evaluation_cost,
    supports_parameter_action,
    tune_structure_candidates,
)
from ..design._search_control import (
    active_operator_indices,
    performance_plateau_select,
    representation_key,
    update_global_archive,
)
from ..design._stream_graph import (
    STREAM_IMPLEMENTATION_REVISION,
    named_initial_genotype,
)
from ..select import select as _select_alg
from ..solve import (
    _write_checkpoint as _write_design_checkpoint,
)
from ..solve import (
    input_algorithm,
    run_algorithm,
)
from ..space import space
from ._search_batch import SearchActionBatch
from .candidate_log import append_candidate_records, trim_candidate_records

_DESIGN_CHECKPOINT_SCHEMA_VERSION = 6


def _progress(app: Any, message: str, *, done: bool = False) -> None:
    if app is not None:
        if hasattr(app, "TextArea"):
            try:
                app.TextArea.Value = message
            except Exception:
                pass
        elif callable(app):
            app(message)
        return
    end = "\n" if done else ""
    prefix = "" if done else "\r"
    print(f"{prefix}{message}", end=end, flush=True)


def _normalize_setting(setting: Any) -> SimpleNamespace:
    if isinstance(setting, SimpleNamespace):
        ns = setting
    elif isinstance(setting, dict):
        ns = SimpleNamespace(**setting)
    else:
        data = {
            name: getattr(setting, name)
            for name in dir(setting)
            if not name.startswith("__") and (not callable(getattr(setting, name)))
        }
        ns = SimpleNamespace(**data)
    for name in list(vars(ns)):
        if name[0].isupper():
            setattr(ns, name.lower(), getattr(ns, name))
    return ns


def _configure_rng(setting: SimpleNamespace) -> np.random.Generator:
    """Attach one seeded generator to all stages of an AutoOptLib run."""
    existing = getattr(setting, "rng", None)
    if isinstance(existing, np.random.Generator):
        return existing
    seed = getattr(
        setting,
        "seed",
        getattr(setting, "random_seed", getattr(setting, "random_state", None)),
    )
    rng = np.random.default_rng(seed)
    setting.rng = rng
    setting.random_seed = seed
    return rng


def _resolve_problem_callable(
    descriptor: Any,
) -> Callable[[Sequence[Any], Sequence[Any], str], Tuple[Any, Any, Any]]:
    if callable(descriptor):
        return descriptor
    if isinstance(descriptor, str):
        if ":" in descriptor:
            (module_name, func_name) = descriptor.split(":", 1)
            module = __import__(module_name, fromlist=[func_name])
            func = getattr(module, func_name)
            if callable(func):
                return func
        raise ValueError(
            "Problem descriptor must be a callable or 'module:function' string in Python translation."
        )
    raise TypeError("Unsupported problem descriptor format")


def _build_problem_struct(
    problem_descriptor: Any, instances: Sequence[Any], setting: SimpleNamespace
):
    problems = []
    for _ in instances:
        problems.append(
            SimpleNamespace(
                name=problem_descriptor,
                setting="",
                N=int(getattr(setting, "ProbN", getattr(setting, "prob_n", 20))),
                Gmax=int(
                    math.ceil(
                        getattr(setting, "ProbFE", getattr(setting, "prob_fe", 5000))
                        / max(
                            1, getattr(setting, "ProbN", getattr(setting, "prob_n", 20))
                        )
                    )
                ),
            )
        )
    return problems


def _mean_performance(algs: Sequence[Design]) -> np.ndarray:
    values = []
    for alg in algs:
        arr = alg.ave_perform_all()
        values.append(float(np.mean(arr)) if arr.size else float("inf"))
    return np.asarray(values, dtype=float)


def _training_cost(candidate: Design, setting: Any, seeds: Sequence[int]) -> float:
    values = np.asarray(candidate.get_performance(setting, seeds), dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return float("inf")
    scorer = getattr(
        setting, "EvalTrainingScorer", getattr(setting, "evaltrainingscorer", None)
    )
    if scorer is not None:
        value = float(scorer(values.reshape(len(seeds), -1), seeds))
        return value if np.isfinite(value) else float("inf")
    return float(np.mean(values))


def _annotate_training_scores(
    candidates: Sequence[Design],
    setting: Any,
    seeds: Sequence[int],
) -> None:
    """Persist the exact selection score used by the candidate ledger."""
    scores = []
    for candidate in candidates:
        metadata = dict(getattr(candidate, "metadata", {}) or {})
        score = _training_cost(candidate, setting, seeds)
        metadata["training_score"] = score
        candidate.metadata = metadata
        scores.append(score)
    if scores and not np.all(np.isfinite(scores)):
        raise FloatingPointError(
            "Exact candidate evaluation produced a non-finite training score."
        )


def _population_quality(
    candidates: Sequence[Design], setting: Any, seeds: Sequence[int]
) -> float:
    """Mean cost of the better half, used for population-level stagnation."""
    if not candidates:
        return float("inf")
    costs = np.sort(
        np.asarray(
            [_training_cost(candidate, setting, seeds) for candidate in candidates],
            dtype=float,
        )
    )
    count = max(1, int(math.ceil(len(costs) / 2)))
    return float(np.mean(costs[:count]))


def _relative_population_improvement(previous: float, current: float) -> float:
    if not np.isfinite(previous) or not np.isfinite(current):
        return 0.0
    scale = max(abs(previous), abs(current), float(np.finfo(float).eps))
    return float(previous - current) / scale


def _unique_designs(
    candidates: Sequence[Design], setting: Any, seeds: Sequence[int]
) -> list[Design]:
    """Deduplicate an elitist pool without letting archive copies occupy slots."""
    unique: dict[str, Design] = {}
    for candidate in candidates:
        key = representation_key(candidate)
        incumbent = unique.get(key)
        if incumbent is None or _training_cost(
            candidate, setting, seeds
        ) < _training_cost(incumbent, setting, seeds):
            unique[key] = candidate
    return list(unique.values())


def _unique_structure_designs(
    candidates: Sequence[Design], setting: Any, seeds: Sequence[int]
) -> list[Design]:
    """Keep the best evaluated algorithm for each decoded graph structure."""
    unique: dict[str, Design] = {}
    for candidate in candidates:
        key = representation_key(candidate, include_continuous_parameters=False)
        incumbent = unique.get(key)
        if incumbent is None or _training_cost(
            candidate, setting, seeds
        ) < _training_cost(incumbent, setting, seeds):
            unique[key] = candidate
    return list(unique.values())


def _sample_unique_design(
    problems: Sequence[Any],
    setting: Any,
    seen: set[str],
    *,
    maximum_attempts: int,
    structure_only: bool = False,
    parameter_bank: Any = None,
) -> Design:
    for _ in range(maximum_attempts):
        candidate = Design(problems, setting)
        configuration = sample_configuration(setting, ensure_rng(setting), candidate)
        if not configuration_is_valid(candidate, configuration):
            continue
        set_configuration(candidate, configuration)
        if parameter_bank is not None:
            candidate = Design.from_genotype(
                candidate.operator,
                deepcopy(parameter_bank),
                problems,
                setting,
                design_aux=deepcopy(candidate.design_aux),
                configuration=candidate.configuration,
            )
        key = representation_key(
            candidate, include_continuous_parameters=not structure_only
        )
        if key not in seen:
            seen.add(key)
            return candidate
    raise RuntimeError(
        f"Search could not sample a new algorithm representation within {maximum_attempts} attempts. Increase SearchMaxAttempts or inspect whether the configured design space is exhausted."
    )


def _sample_unique_structure_design(
    problems: Sequence[Any],
    setting: Any,
    seen_structures: set[str],
    seen_representations: set[str],
    *,
    maximum_attempts: int,
    parameter_bank: Any = None,
) -> Design:
    """Sample a graph that is unique both before and after decoding."""
    for _ in range(maximum_attempts):
        candidate = Design(problems, setting)
        configuration = sample_configuration(setting, ensure_rng(setting), candidate)
        if not configuration_is_valid(candidate, configuration):
            continue
        set_configuration(candidate, configuration)
        if parameter_bank is not None:
            candidate = Design.from_genotype(
                candidate.operator,
                deepcopy(parameter_bank),
                problems,
                setting,
                design_aux=deepcopy(candidate.design_aux),
                configuration=candidate.configuration,
            )
        structure_key = representation_key(
            candidate, include_continuous_parameters=False
        )
        if structure_key in seen_structures:
            continue
        seen_structures.add(structure_key)
        if representation_key(candidate) not in seen_representations:
            return candidate
    raise StructureNoveltyExhausted(maximum_attempts)


def _checkpoint_value(value: Any) -> Any:
    """Convert search semantics to a deterministic checkpoint fingerprint value."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _checkpoint_value(item)
            for (key, item) in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_checkpoint_value(item) for item in value]
    if callable(value):
        try:
            implementation = inspect.getsource(value).encode("utf-8")
        except (OSError, TypeError):
            code = getattr(value, "__code__", None)
            implementation = repr(
                (
                    getattr(code, "co_code", b""),
                    getattr(code, "co_consts", ()),
                    getattr(code, "co_names", ()),
                )
            ).encode("utf-8")
        result = {
            "module": getattr(value, "__module__", type(value).__module__),
            "qualname": getattr(value, "__qualname__", type(value).__qualname__),
            "implementation_sha256": hashlib.sha256(implementation).hexdigest(),
        }
        defaults = getattr(value, "__defaults__", None)
        if defaults:
            result["defaults"] = _checkpoint_value(defaults)
        keyword_defaults = getattr(value, "__kwdefaults__", None)
        if keyword_defaults:
            result["keyword_defaults"] = _checkpoint_value(keyword_defaults)
        closure = getattr(value, "__closure__", None)
        if closure:
            stable_closure: list[Any] = []
            for cell in closure:
                try:
                    item = cell.cell_contents
                except ValueError:
                    stable_closure.append({"empty_cell": True})
                    continue
                if (
                    item is None
                    or isinstance(
                        item,
                        (str, int, float, bool, np.generic, Path, tuple, frozenset),
                    )
                    or callable(item)
                ):
                    stable_closure.append(_checkpoint_value(item))
                else:
                    stable_closure.append(
                        {
                            "mutable_type": f"{type(item).__module__}.{type(item).__qualname__}"
                        }
                    )
            result["stable_closure"] = stable_closure
        explicit_semantics = getattr(value, "__autoopt_checkpoint_semantics__", None)
        if explicit_semantics is not None:
            result["semantics"] = _checkpoint_value(explicit_semantics)
        if hasattr(value, "__dict__"):
            # Execution plans are rebuildable runtime metadata, with a back
            # reference to this function. They are neither scientific state
            # nor checkpoint payload and must not recurse into the signature.
            from ...components._execution import Computation
            state = {
                key: item
                for (key, item) in vars(value).items()
                if key != "__autoopt_checkpoint_semantics__"
                and not (key == "__execution_computation__"
                         and type(item) is Computation and item.reference is value)
            }
            if state:
                result["state"] = _checkpoint_value(state)
        return result
    if hasattr(value, "__dict__"):
        return _checkpoint_value(vars(value))
    return repr(value)


def _design_checkpoint_signature(
    problem_descriptor: Any,
    setting: Any,
    *,
    setting_overrides: dict[str, Any] | None = None,
) -> str:
    """Fingerprint every setting that changes exact Search semantics."""
    names: tuple[str, ...] = (
        "AlgP",
        "AlgQ",
        "AlgN",
        "AlgFE",
        "AlgRuns",
        "ProbN",
        "CrossoverRate",
        "MutationRate",
        "PopulationSizeSpace",
        "OffspringSizeSpace",
        "ProbFE",
        "InnerFE",
        "IncRate",
        "Archive",
        "InitialPopulations",
        "Tmax",
        "Thres",
        "Metric",
        "Compare",
        "Evaluate",
        "Alpha",
        "Data",
        "Seed",
        "EvalBackend",
        "EvalWorkers",
        "EvalRetries",
        "EvalTimeoutSec",
        "EvalFailure",
        "EvalPenalty",
        "StreamEventBudgetMultiplier",
        "EvalWorkerConfig",
        "EvalWorkerInitializer",
        "EvalWorkerFinalizer",
        "EvalCommonRandomSeed",
        "EvalCoordinateSeeds",
        "EvalTrainingScorer",
        "StructureMutationRate",
        "StructureCandidatesPerAction",
        "SearchStagnation",
        "SearchRestartFraction",
        "SearchImprovementRate",
        "SearchParameterActionProbability",
        "SearchActionProbabilityGain",
        "SearchActionRewardEWMA",
        "ParameterCMAOffspring",
        "ParameterCMABlockGenerations",
        "PostStructureCMAGenerations",
        "SearchMaxAttempts",
        "SearchInitialDesigns",
        "GraphSemantics",
        "CheckpointSignature",
        "op_space",
        "all_op",
        "behav_space",
        "para_space",
        "para_type_space",
    )
    component_names = {str(name) for name in get_flex(setting, "all_op", ()) or ()}
    component_names.update(
        (str(name) for name in get_flex(setting, "Archive", ()) or ())
    )
    component_names.update(("para_cma", "para_pso"))
    components = {
        name: _checkpoint_value(get_component(name)) for name in sorted(component_names)
    }
    overrides = dict(setting_overrides or {})
    settings_payload = {
        name: _checkpoint_value(
            overrides.get(
                name, getattr(setting, name, getattr(setting, name.lower(), None))
            )
        )
        for name in names
    }
    if getattr(setting, "DesignFEs", None) is not None:
        settings_payload["DesignFEs"] = int(setting.DesignFEs)
    payload = {
        "implementation": STREAM_IMPLEMENTATION_REVISION,
        "constraint_order": CONSTRAINT_ORDER_REVISION,
        "problem": _checkpoint_value(problem_descriptor),
        "components": components,
        "settings": settings_payload,
    }
    declared_space = getattr(setting, "DesignSpace", None)
    if declared_space is not None:
        payload["design_space"] = declared_space.to_dict()
    if getattr(setting, "ParameterScope", "component") == "node":
        payload["parameter_scope"] = "node_v1"
    if (
        declared_space is not None
        or getattr(setting, "ParameterScope", "component") == "node"
    ):
        from ...components.contracts import get_component_contract
        from ...model_semantics import execution_spec

        payload["program_contract"] = execution_spec(
            get_flex(setting, "ComponentBackend", "python"),
            get_flex(setting, "StreamBackend", None),
        ).to_dict()
        payload["component_contracts"] = {
            name: get_component_contract(name).to_dict()
            for name in sorted(component_names)
        }
    if str(getattr(setting, "StreamBackend", "batch" if getattr(setting, "ComponentBackend", "python") == "native" else "python")).lower() == "batch":
        payload["population_execution"] = "population_batch_v1"
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_design_checkpoint(
    path: Path,
    *,
    alg_n: int,
    alg_fe: int,
    instance_train: Sequence[Any],
    instance_test: Sequence[Any],
    checkpoint_signature: str,
    budget_extension_source_alg_fe: int | None = None,
    budget_extension_source_signature: str | None = None,
) -> dict[str, Any]:
    warnings.warn(
        "Loading an AutoOptLib checkpoint uses pickle and can execute arbitrary code. Resume only checkpoints produced by a trusted run.",
        UserWarning,
        stacklevel=2,
    )
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "autooptlib.design-checkpoint"
    ):
        raise ValueError(f"Invalid AutoOptLib design checkpoint: {path}")
    if payload.get("schema_version") != _DESIGN_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported AutoOptLib design checkpoint schema version.")
    expected = {
        "alg_n": alg_n,
        "instance_train": list(instance_train),
        "instance_test": list(instance_test),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(
                f"Design checkpoint {name} does not match the current run."
            )
    current_checkpoint = (
        payload.get("alg_fe") == alg_fe
        and payload.get("checkpoint_signature") == checkpoint_signature
    )
    source_checkpoint = (
        budget_extension_source_alg_fe is not None
        and budget_extension_source_signature is not None
        and (payload.get("alg_fe") == budget_extension_source_alg_fe)
        and (payload.get("checkpoint_signature") == budget_extension_source_signature)
    )
    if current_checkpoint:
        payload["budget_extension_source_checkpoint"] = False
        return payload
    if not source_checkpoint:
        mismatch = (
            "alg_fe"
            if payload.get("alg_fe") not in {alg_fe, budget_extension_source_alg_fe}
            else "checkpoint_signature"
        )
        raise ValueError(
            f"Design checkpoint {mismatch} does not match the current run."
        )
    if budget_extension_source_alg_fe is None:
        raise ValueError("Budget-extension source AlgFE is missing.")
    if alg_fe <= int(budget_extension_source_alg_fe):
        raise ValueError(
            "Budget-extension resume requires a strictly larger target AlgFE."
        )
    if not payload.get("complete", False):
        raise ValueError(
            "Budget-extension resume requires a completed source checkpoint."
        )
    if int(payload.get("evaluated_count", -1)) != int(budget_extension_source_alg_fe):
        raise ValueError(
            "Budget-extension source progress does not equal its registered AlgFE."
        )
    payload["budget_extension_source_checkpoint"] = True
    return payload


def _process_design(
    problem_descriptor: Any,
    instance_train: Sequence[Any],
    instance_test: Sequence[Any],
    setting: Any,
    app: Any,
    runtime_holder: dict[str, DesignEvaluationRuntime] | None = None,
) -> Tuple[List[Design], List[Design]]:
    if str(getattr(setting, "Evaluate", "exact")).lower() != "exact":
        raise ValueError("Search V6 supports exact evaluation only.")
    setting_ns = setting
    if getattr(setting_ns, "GraphSemantics", "stream_graph_v2") != "stream_graph_v2":
        raise ValueError("Only Search V6 stream_graph_v2 can design new algorithms.")
    rng = setting_ns.rng
    instance_train = list(instance_train)
    instance_test = list(instance_test)
    instances = instance_train + instance_test
    problems = _build_problem_struct(problem_descriptor, instances, setting_ns)
    construct_fn = _resolve_problem_callable(problem_descriptor)
    (problems, data, _) = construct_fn(problems, instances, "construct")
    validate_constructed_problems(problems, data)
    task_budgets = [
        int(budget)
        for item in data
        if (budget := getattr(getattr(item, "task", None), "budget", None)) is not None
    ]
    if task_budgets:
        minimum_budget = min(task_budgets)
        for name, fallback in (
            ("PopulationSizeSpace", range(4, 101)),
            ("OffspringSizeSpace", range(1, 101)),
        ):
            values = [
                int(value)
                for value in getattr(setting_ns, name, fallback)
                if int(value) <= minimum_budget
            ]
            if not values:
                raise ValueError(
                    f"{name} has no value within the smallest task budget."
                )
            setattr(setting_ns, name, values)
    setting_ns = space(problems, setting_ns)
    setting_ns.SearchEvaluationEngine = "search_v6"
    checkpoint_signature = (
        _design_checkpoint_signature(problem_descriptor, setting_ns)
        if getattr(setting_ns, "CheckpointDir", None) is not None
        else None
    )
    design_runtime = DesignEvaluationRuntime(problems, data, setting_ns)
    if runtime_holder is not None:
        runtime_holder["runtime"] = design_runtime
    alg_n = int(getattr(setting_ns, "AlgN", getattr(setting_ns, "alg_n", 1)))
    alg_fe = int(getattr(setting_ns, "AlgFE", getattr(setting_ns, "alg_fe", alg_n)))
    seed_train = rng.permutation(len(instance_train)).tolist()
    seed_test = (rng.permutation(len(instance_test)) + len(instance_train)).tolist()
    design_fe_limit = getattr(setting_ns, "DesignFEs", None)
    full_candidate_fes = int(setting_ns.AlgRuns) * sum(
        (
            int(
                get_flex(getattr(data[index], "task", None), "budget", None)
                or setting_ns.ProbFE
            )
            for index in seed_train
        )
    )
    if design_fe_limit is not None:
        if any(
            (
                str(get_flex(problems[index], "type", ["", "static"])[1]) != "static"
                for index in seed_train
            )
        ):
            raise ValueError(
                "DesignFEs requires static tasks with bounded per-run FE costs."
            )
        if design_fe_limit < alg_n * full_candidate_fes:
            raise ValueError(
                "DesignFEs must cover full evaluation of the initial population."
            )

    def remaining_candidates() -> int:
        if design_fe_limit is None:
            return max(0, alg_fe - evaluated_count)
        remaining = max(
            0,
            (int(design_fe_limit) - design_runtime.total_evaluations)
            // full_candidate_fes,
        )
        # Reserve a whole macro-action, including a possible pre-evaluation
        # fallback to the other action. FE mode never truncates a CMA block.
        complete_reserve = max(
            structure_action_evaluation_cost(setting_ns),
            parameter_action_evaluation_cost(setting_ns),
        )
        return remaining if remaining >= complete_reserve else 0

    checkpoint_dir = getattr(setting_ns, "CheckpointDir", None)
    checkpoint_path = (
        Path(checkpoint_dir) / "design.pkl" if checkpoint_dir is not None else None
    )
    checkpoint_every = int(getattr(setting_ns, "CheckpointEvery", 1))
    resume = bool(getattr(setting_ns, "Resume", False))
    budget_extension = bool(getattr(setting_ns, "ResumeBudgetExtension", False))
    extension_source_alg_fe = (
        int(setting_ns.ResumeBudgetExtensionSourceAlgFE) if budget_extension else None
    )
    extension_source_signature = (
        str(setting_ns.ResumeBudgetExtensionSourceCheckpointSignature)
        if budget_extension
        else None
    )
    resume_state = None
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        resume_state = _load_design_checkpoint(
            checkpoint_path,
            alg_n=alg_n,
            alg_fe=alg_fe,
            instance_train=instance_train,
            instance_test=instance_test,
            checkpoint_signature=checkpoint_signature,
            budget_extension_source_alg_fe=extension_source_alg_fe,
            budget_extension_source_signature=extension_source_signature,
        )
    _progress(app, "Initializing...")
    algs: List[Design] = []
    evaluated_count = 0
    alg_trace: List[Design] = []
    search_archive: list[Design] = []
    seen_keys: set[str] = set()
    seen_structure_keys: set[str] = set()
    stagnation_generations = 0
    population_quality = float("inf")
    component_parameter_bank: Any = None
    initial_evaluated: list[Design] = []
    controller_stats: dict[str, Any] = {
        "duplicate_rejections": 0,
        "structure_novelty_rejections": 0,
        "structure_to_parameter_fallbacks": 0,
        "parameter_to_structure_fallbacks": 0,
        "restart_candidates": 0,
        "action_history": [],
        "population_quality_history": [],
        "generation_probability_history": [],
        "initial_structure_backfills": 0,
    }
    maximum_attempts = int(getattr(setting_ns, "SearchMaxAttempts", 50))
    action_selector = ActionSelector.from_setting(setting_ns)

    def controller_summary() -> dict[str, Any]:
        summary = deepcopy(controller_stats)
        summary["generations"] = len(alg_trace)
        summary["action_selector"] = action_selector.payload()
        blocks = {STRUCTURE: 0, PARAMETER: 0}
        evaluations = {STRUCTURE: 0, PARAMETER: 0}
        semantic = {STRUCTURE: 0, PARAMETER: 0}
        for entry in controller_stats["action_history"]:
            blocks[entry["action"]] += 1
            evaluations[entry["action"]] += entry["evaluation_cost"]
            for action, count in entry["semantic_candidate_evaluations"].items():
                semantic[action] += count
        summary.update(
            outer_action_blocks=blocks,
            outer_action_evaluations=evaluations,
            semantic_candidate_evaluations=semantic,
        )
        return summary

    if resume_state is not None:
        algs = list(resume_state["algs"])
        alg_trace = list(resume_state["alg_trace"])
        evaluated_count = int(resume_state["evaluated_count"])
        seed_train = list(resume_state["seed_train"])
        seed_test = list(resume_state["seed_test"])
        rng.bit_generator.state = resume_state["rng_state"]
        search_state = dict(resume_state.get("search_state"))
        search_archive = list(search_state.get("archive", []))
        seen_keys = set(search_state.get("seen_keys", []))
        seen_structure_keys = set(search_state.get("seen_structure_keys", []))
        stagnation_generations = int(search_state.get("stagnation_generations", 0))
        controller_stats.update(search_state.get("controller_stats", {}))
        action_selector = ActionSelector.from_setting(
            setting_ns, search_state.get("action_selector")
        )
        component_parameter_bank = deepcopy(
            search_state.get("component_parameter_bank")
        )
        design_runtime.total_evaluations = int(resume_state["actual_design_fes"])
        trim_candidate_records(
            getattr(setting_ns, "CandidateLogPath", None), alg_n + evaluated_count
        )
        if resume_state.get("complete", False) and (
            not resume_state.get("budget_extension_source_checkpoint", False)
        ):
            setting_ns.SearchArchive = list(search_archive)
            setting_ns.SearchControllerStats = controller_summary()
            setting_ns.EvalActualDesignFEs = design_runtime.total_evaluations
            setting_ns.SearchEvaluatedCandidates = alg_n + evaluated_count
            setting_ns.SearchTrainingFEs = int(
                controller_stats.get(
                    "training_design_fes", design_runtime.total_evaluations
                )
            )
            if design_fe_limit is not None:
                setting_ns.SearchUnusedDesignFEs = (
                    int(design_fe_limit) - setting_ns.SearchTrainingFEs
                )
            design_runtime.close()
            _progress(app, "Complete", done=True)
            return (algs[:alg_n], alg_trace)
    else:
        initial_specifications = list(getattr(setting_ns, "SearchInitialDesigns", []))
        initial_designs: list[Design] = []
        for warm_index, specification in enumerate(initial_specifications, 1):
            (operators, parameters, configuration) = named_initial_genotype(
                specification, setting_ns
            )
            candidate = Design.from_genotype(
                operators, parameters, problems, setting_ns, configuration=configuration
            )
            if not configuration_is_valid(candidate, candidate.configuration):
                raise ValueError(
                    f"Search initial design {warm_index} violates its component population/offspring-size contract."
                )
            candidate.metadata = {
                "initialization": "warm_start",
                "warm_start_index": warm_index,
                **dict(specification.get("metadata")),
            }
            initial_designs.append(candidate)
        if len(initial_designs) > alg_n:
            raise ValueError("SearchInitialDesigns cannot exceed AlgN.")
        prototype = Design(problems, setting_ns)
        component_parameter_bank = deepcopy(prototype.parameter)
        setting_ns._ComponentParameterBank = component_parameter_bank
        algs = list(initial_designs)
        for candidate in algs:
            key = representation_key(candidate)
            if key in seen_keys:
                raise ValueError("SearchInitialDesigns contains duplicate algorithms.")
            seen_keys.add(key)
            seen_structure_keys.add(
                representation_key(candidate, include_continuous_parameters=False)
            )
        if len(algs) < alg_n:
            prototype = Design.from_genotype(
                prototype.operator,
                deepcopy(component_parameter_bank),
                problems,
                setting_ns,
                design_aux=deepcopy(prototype.design_aux),
            )
            first_structure = representation_key(
                prototype, include_continuous_parameters=False
            )
            prototype_key = representation_key(prototype)
            if prototype_key in seen_keys:
                prototype = _sample_unique_structure_design(
                    problems,
                    setting_ns,
                    seen_structure_keys,
                    seen_keys,
                    maximum_attempts=maximum_attempts,
                    parameter_bank=component_parameter_bank,
                )
                first_structure = representation_key(
                    prototype, include_continuous_parameters=False
                )
                prototype_key = representation_key(prototype)
            seen_structure_keys.add(first_structure)
            seen_keys.add(prototype_key)
            prototype.metadata = {"initialization": "cold_start"}
            algs.append(prototype)
        for _ in range(max(0, alg_n - len(algs))):
            candidate = _sample_unique_structure_design(
                problems,
                setting_ns,
                seen_structure_keys,
                seen_keys,
                maximum_attempts=maximum_attempts,
                parameter_bank=component_parameter_bank,
            )
            candidate.metadata = {"initialization": "cold_start"}
            seen_keys.add(representation_key(candidate))
            algs.append(candidate)
        design_runtime.evaluate(algs, setting_ns, seed_train)
        initial_evaluated = list(algs)
        algs = _unique_structure_designs(algs, setting_ns, seed_train)
        structure_deficit = alg_n - len(algs)
        if structure_deficit > 0:
            if structure_deficit > alg_fe:
                raise RuntimeError(
                    "Search cannot backfill a structure-unique initial parent population within the candidate budget."
                )
            replacements: list[Design] = []
            for _ in range(structure_deficit):
                replacement = _sample_unique_structure_design(
                    problems,
                    setting_ns,
                    seen_structure_keys,
                    seen_keys,
                    maximum_attempts=maximum_attempts,
                    parameter_bank=component_parameter_bank,
                )
                replacement.metadata = {
                    "initialization": "cold_start_structure_backfill"
                }
                seen_keys.add(representation_key(replacement))
                replacements.append(replacement)
            design_runtime.evaluate(replacements, setting_ns, seed_train)
            initial_evaluated.extend(replacements)
            algs.extend(replacements)
            evaluated_count += len(replacements)
            controller_stats["initial_structure_backfills"] += len(replacements)
        if len(algs) != alg_n:
            raise RuntimeError(
                "Search failed to construct a full structure-unique parent population."
            )
    if resume_state is None:
        if not initial_evaluated:
            initial_evaluated = list(algs)

        if component_parameter_bank is None:
            component_parameter_bank = deepcopy(algs[0].parameter) if algs else None
        seen_structure_keys.update(
            (
                representation_key(candidate, include_continuous_parameters=False)
                for candidate in algs
            )
        )
        search_archive = update_global_archive(
            search_archive,
            algs,
            size=1,
            cost=lambda candidate: _training_cost(candidate, setting_ns, seed_train),
        )
        population_quality = _population_quality(algs, setting_ns, seed_train)
    else:
        population_quality = _population_quality(algs, setting_ns, seed_train)
        if not seen_structure_keys:
            seen_structure_keys.update(
                (
                    representation_key(candidate, include_continuous_parameters=False)
                    for candidate in algs
                )
            )
        if component_parameter_bank is None and algs:
            component_parameter_bank = deepcopy(algs[0].parameter)
    setting_ns._ComponentParameterBank = component_parameter_bank
    if resume_state is None:
        candidate_log = getattr(setting_ns, "CandidateLogPath", None)
        if candidate_log is not None:
            Path(candidate_log).unlink(missing_ok=True)
        _annotate_training_scores(
            initial_evaluated,
            setting_ns,
            seed_train,
        )
        append_candidate_records(
            candidate_log, initial_evaluated, start=1, designer="search"
        )

    def save_checkpoint(*, complete: bool) -> None:
        _write_design_checkpoint(
            checkpoint_path,
            {
                "schema": "autooptlib.design-checkpoint",
                "schema_version": _DESIGN_CHECKPOINT_SCHEMA_VERSION,
                "alg_n": alg_n,
                "alg_fe": alg_fe,
                "instance_train": instance_train,
                "instance_test": instance_test,
                "algs": list(algs),
                "alg_trace": list(alg_trace),
                "evaluated_count": evaluated_count,
                "seed_train": seed_train,
                "seed_test": seed_test,
                "rng_state": rng.bit_generator.state,
                "complete": complete,
                "actual_design_fes": design_runtime.total_evaluations,
                "checkpoint_signature": checkpoint_signature,
                "search_state": {
                    "archive": list(search_archive),
                    "seen_keys": sorted(seen_keys),
                    "seen_structure_keys": sorted(seen_structure_keys),
                    "stagnation_generations": stagnation_generations,
                    "controller_stats": deepcopy(controller_stats),
                    "action_selector": action_selector.payload(),
                    "component_parameter_bank": deepcopy(component_parameter_bank),
                },
            },
        )

    if checkpoint_path is not None and resume_state is None:
        save_checkpoint(complete=False)
    generation_representatives: list[Design] = []
    generation_evaluated: list[Design] = []
    generation_parent_queue: list[Design] | None = None
    generation_action_count = 0
    generation_best_reference = _training_cost(
        search_archive[0], setting_ns, seed_train
    )
    generation_action_history_start = len(controller_stats["action_history"])
    generation_parameter_budget_fraction = 0.5

    def controller_parameter_budget_fraction() -> float:
        return action_selector.parameter_probability(
            float(getattr(setting_ns, "SearchParameterActionProbability", 0.5))
        )

    def evaluate_action_batch(candidates: Sequence[Design]) -> None:
        design_runtime.evaluate(candidates, setting_ns, seed_train)

    batch_evaluator = SearchActionBatch(
        problems,
        setting_ns,
        evaluate_action_batch,
        lambda candidate: _training_cost(candidate, setting_ns, seed_train),
        seen_structure_keys,
        seen_keys,
    )

    def record_action_result(
        action: str,
        parent: Design,
        result: Any,
        *,
        restart: bool = False,
        parameter_budget_fraction: float | None = None,
    ) -> None:
        nonlocal evaluated_count, search_archive
        nonlocal algs, component_parameter_bank
        if (
            not int(result.evaluation_cost)
            == len(result.evaluated)
            == len(result.candidate_actions)
        ):
            raise RuntimeError(
                "A Search action reported inconsistent exact candidate accounting before ledger recording."
            )
        start = alg_n + evaluated_count + 1
        evaluated_count += int(result.evaluation_cost)
        candidate_actions = tuple(result.candidate_actions)
        if len(candidate_actions) != len(result.evaluated):
            raise ValueError(
                "Semantic candidate actions must align with evaluated candidates."
            )
        for candidate, candidate_action in zip(result.evaluated, candidate_actions):
            metadata = dict(getattr(candidate, "metadata", {}))
            metadata["search_action"] = candidate_action
            metadata["search_action_block"] = action
            metadata["search_action_context"] = (
                "structure_proposal"
                if candidate_action == STRUCTURE
                else "post_structure_bootstrap"
                if action == STRUCTURE
                else "optional_parameter_block"
            )
            if action == STRUCTURE and candidate_action == STRUCTURE:
                metadata["search_structure_batch_selected"] = (
                    candidate is result.structure_baseline
                )
                if result.bootstrap_skip_reason is not None:
                    metadata["post_structure_cma_skip_reason"] = (
                        result.bootstrap_skip_reason
                    )
            if restart:
                metadata["search_restart"] = True
            candidate.metadata = metadata
            seen_keys.add(representation_key(candidate))
        _annotate_training_scores(result.evaluated, setting_ns, seed_train)
        append_candidate_records(
            getattr(setting_ns, "CandidateLogPath", None),
            result.evaluated,
            start=start,
            designer="search",
        )
        control_credit = controller_action_credit(result, parent)
        control_parent_cost = _training_cost(
            control_credit.parent, setting_ns, seed_train
        )
        control_child_cost = _training_cost(
            control_credit.child, setting_ns, seed_train
        )
        actual_action_fes = sum(
            (
                int(entry.get("evaluations", 0))
                for candidate in result.evaluated
                for entry in candidate.metadata.get("evaluation_ledger", [])
            )
        )
        reward_cost = (
            actual_action_fes / full_candidate_fes
            if design_fe_limit is not None
            else control_credit.evaluation_cost
        )
        attribution = action_selector.observe(
            control_credit.action,
            parent_cost=control_parent_cost,
            child_cost=control_child_cost,
            evaluation_cost=reward_cost,
        )
        semantic_diagnostics = []
        for diagnostic in semantic_action_credits(result, parent):
            diagnostic_parent_cost = _training_cost(
                diagnostic.parent, setting_ns, seed_train
            )
            diagnostic_child_cost = _training_cost(
                diagnostic.child, setting_ns, seed_train
            )
            semantic_diagnostics.append(
                {
                    "action": diagnostic.action,
                    "context": diagnostic.context,
                    "evaluation_cost": int(diagnostic.evaluation_cost),
                    "parent_cost": float(diagnostic_parent_cost),
                    "child_cost": float(diagnostic_child_cost),
                }
            )
        if _training_cost(
            result.representative, setting_ns, seed_train
        ) < _training_cost(parent, setting_ns, seed_train):
            if component_parameter_bank is None:
                component_parameter_bank = deepcopy(result.representative.parameter)
            else:
                for operator_index in active_operator_indices(result.representative):
                    if operator_index <= len(
                        component_parameter_bank
                    ) and operator_index <= len(result.representative.parameter):
                        component_parameter_bank[operator_index - 1] = deepcopy(
                            result.representative.parameter[operator_index - 1]
                        )
            setting_ns._ComponentParameterBank = component_parameter_bank
        search_archive = update_global_archive(
            search_archive,
            [*result.evaluated, *algs],
            size=1,
            cost=lambda candidate: _training_cost(candidate, setting_ns, seed_train),
        )
        controller_stats["action_history"].append(
            {
                "action": action,
                "action_block": action,
                "evaluation_cost": int(result.evaluation_cost),
                "actual_design_fes": actual_action_fes,
                "post_structure_cma_skip_reason": result.bootstrap_skip_reason,
                "semantic_candidate_evaluations": {
                    candidate_action: candidate_actions.count(candidate_action)
                    for candidate_action in (STRUCTURE, PARAMETER)
                },
                "controller_credit": {
                    "action": control_credit.action,
                    "context": control_credit.context,
                    "evaluation_cost": int(control_credit.evaluation_cost),
                    "reward_cost_full_candidate_equivalents": float(reward_cost),
                    "parent_cost": float(control_parent_cost),
                    "child_cost": float(control_child_cost),
                    "relative_improvement": attribution["relative_improvement"],
                    "reward": attribution["reward"],
                },
                "semantic_credit_diagnostics": semantic_diagnostics,
                "restart": bool(restart),
                "parameter_budget_fraction": float(
                    controller_parameter_budget_fraction()
                    if parameter_budget_fraction is None
                    else parameter_budget_fraction
                ),
            }
        )

    def replace_parameter_parent(parent: Design, representative: Design) -> None:
        """Update one lineage in place instead of duplicating its graph."""
        for index, candidate in enumerate(algs):
            if candidate is parent:
                algs[index] = representative
                return
        parent_key = representation_key(parent)
        for index, candidate in enumerate(algs):
            if representation_key(candidate) == parent_key:
                algs[index] = representative
                return
        raise RuntimeError("The parameter-action parent left the active population.")

    while remaining_candidates() > 0:
        if generation_parent_queue is None:
            generation_action_history_start = len(controller_stats["action_history"])
            generation_parameter_budget_fraction = (
                controller_parameter_budget_fraction()
            )
            if algs and stagnation_generations >= int(
                getattr(setting_ns, "SearchStagnation", 3)
            ):
                restart_count = min(
                    len(algs),
                    max(
                        1,
                        int(
                            math.ceil(
                                len(algs)
                                * float(
                                    getattr(setting_ns, "SearchRestartFraction", 0.5)
                                )
                            )
                        ),
                    ),
                )
                ordered = sorted(
                    range(len(algs)),
                    key=lambda index: _training_cost(
                        algs[index], setting_ns, seed_train
                    ),
                    reverse=True,
                )
                set(ordered[:restart_count])
                completed_restart_indices: set[int] = set()
                for index in ordered[:restart_count]:
                    if remaining_candidates() <= 0:
                        break
                    parent = algs[index]
                    remaining = remaining_candidates()
                    restart_cost = 1 + int(
                        getattr(setting_ns, "ParameterCMAOffspring", 5)
                    ) * int(getattr(setting_ns, "PostStructureCMAGenerations", 1))
                    if remaining < restart_cost:
                        break
                    try:
                        fresh = _sample_unique_structure_design(
                            problems,
                            setting_ns,
                            seen_structure_keys,
                            seen_keys,
                            maximum_attempts=STRUCTURE_NOVELTY_ATTEMPTS,
                            parameter_bank=component_parameter_bank,
                        )
                    except StructureNoveltyExhausted:
                        controller_stats["structure_novelty_rejections"] += (
                            STRUCTURE_NOVELTY_ATTEMPTS
                        )
                        controller_stats["structure_to_parameter_fallbacks"] += 1
                        if not supports_parameter_action(parent, setting_ns):
                            break
                        result = execute_action(
                            PARAMETER,
                            parent,
                            problems,
                            setting_ns,
                            evaluate_action_batch,
                            lambda candidate: _training_cost(
                                candidate, setting_ns, seed_train
                            ),
                            seen_structures=seen_structure_keys,
                            seen_representations=seen_keys,
                            maximum_attempts=maximum_attempts,
                            remaining_evaluations=remaining_candidates(),
                        )
                        record_action_result(
                            PARAMETER,
                            parent,
                            result,
                            parameter_budget_fraction=generation_parameter_budget_fraction,
                        )
                        replace_parameter_parent(parent, result.representative)
                        generation_evaluated.extend(result.evaluated)
                        generation_action_count += 1
                        completed_restart_indices.add(index)
                        continue
                    fresh_state = dict(fresh.design_aux or {})
                    fresh_state["conditional_operator_indices"] = list(
                        active_operator_indices(fresh)
                    )
                    fresh.design_aux = fresh_state
                    result = tune_structure_candidates(
                        [fresh],
                        problems,
                        setting_ns,
                        evaluate_action_batch,
                        lambda candidate: _training_cost(
                            candidate, setting_ns, seed_train
                        ),
                        seen_representations=seen_keys,
                        remaining_evaluations=remaining_candidates(),
                    )
                    record_action_result(
                        STRUCTURE,
                        parent,
                        result,
                        restart=True,
                        parameter_budget_fraction=generation_parameter_budget_fraction,
                    )
                    algs[index] = result.representative
                    generation_representatives.append(result.representative)
                    generation_evaluated.extend(result.evaluated)
                    generation_action_count += 1
                    completed_restart_indices.add(index)
                    controller_stats["restart_candidates"] += 1
                stagnation_generations = 0
                population_quality = _population_quality(algs, setting_ns, seed_train)
                generation_parent_queue = [
                    candidate
                    for (index, candidate) in enumerate(algs)
                    if index not in completed_restart_indices
                ]
            else:
                generation_parent_queue = list(algs)
        if remaining_candidates() > 0 and generation_parent_queue:
            (plans, consumed_parents) = batch_evaluator.execute(
                generation_parent_queue,
                candidate_budget=remaining_candidates(),
                maximum_actions=alg_n - generation_action_count,
                parameter_budget_fraction=generation_parameter_budget_fraction,
            )
            generation_parent_queue = generation_parent_queue[consumed_parents:]
            for plan in plans:
                if plan.get("exhausted"):
                    controller_stats["stalled_parents"] = controller_stats.get("stalled_parents",0)+1
                    continue
                parent = plan["parent"]
                action = plan["action"]
                result = plan["result"]
                novelty_attempts = int(plan.get("novelty_attempts", 0))
                if novelty_attempts:
                    controller_stats["structure_novelty_rejections"] += novelty_attempts
                    controller_stats["structure_to_parameter_fallbacks"] += 1
                if plan.get("parameter_sampling_fallback", False):
                    controller_stats["parameter_to_structure_fallbacks"] += 1
                record_action_result(
                    action,
                    parent,
                    result,
                    parameter_budget_fraction=generation_parameter_budget_fraction,
                )
                if action == PARAMETER:
                    replace_parameter_parent(parent, result.representative)
                else:
                    generation_representatives.append(result.representative)
                generation_evaluated.extend(result.evaluated)
                generation_action_count += 1
        if not generation_parent_queue and not generation_evaluated and controller_stats.get("stalled_parents",0):
            controller_stats["termination_reason"] = "no_admissible_action_within_attempt_limit"
            break
        generation_complete = (
            generation_action_count >= alg_n or remaining_candidates() <= 0
            or (not generation_parent_queue and controller_stats.get("stalled_parents",0)>0)
        )
        if not generation_complete:
            continue
        selection_pool = _unique_designs(
            [*algs, *generation_representatives, *search_archive],
            setting_ns,
            seed_train,
        )
        selection_pool = _unique_structure_designs(
            selection_pool, setting_ns, seed_train
        )
        if len(selection_pool) < alg_n:
            raise RuntimeError(
                "Exact Search has fewer structure-unique evaluated candidates than required parent slots."
            )
        if str(get_flex(setting_ns, "Compare", "average")).lower() == "average":
            algs = performance_plateau_select(
                selection_pool,
                count=alg_n,
                costs=[
                    _training_cost(candidate, setting_ns, seed_train)
                    for candidate in selection_pool
                ],
                global_best=search_archive[0],
            )
        else:
            algs = _select_alg(selection_pool, problems, data, setting_ns, seed_train)
        selected_structure_keys = {
            representation_key(candidate, include_continuous_parameters=False)
            for candidate in algs
        }
        if len(selected_structure_keys) != len(algs):
            raise RuntimeError(
                "Exact Search parent population contains duplicate structures."
            )
        current_quality = _population_quality(algs, setting_ns, seed_train)
        population_improvement = _relative_population_improvement(
            population_quality, current_quality
        )
        current_global_best = _training_cost(search_archive[0], setting_ns, seed_train)
        strict_global_improvement = current_global_best < generation_best_reference
        significant_population_improvement = population_improvement >= float(
            getattr(setting_ns, "SearchImprovementRate", 0.05)
        )
        if significant_population_improvement:
            stagnation_generations = 0
        else:
            stagnation_generations += 1
        generation_best_reference = current_global_best
        population_quality = current_quality
        controller_stats["population_quality_history"].append(
            {
                "quality": float(current_quality),
                "relative_improvement": float(population_improvement),
                "global_best": float(current_global_best),
                "strict_global_improvement": bool(strict_global_improvement),
                "significant_population_improvement": bool(
                    significant_population_improvement
                ),
            }
        )
        generation_actions = controller_stats["action_history"][
            generation_action_history_start:
        ]
        action_counts = {
            action: sum((item["action"] == action for item in generation_actions))
            for action in (STRUCTURE, PARAMETER)
        }
        generation_rewards = {}
        for action in (STRUCTURE, PARAMETER):
            generation_rewards[action] = [
                float(item["controller_credit"]["reward"])
                for item in generation_actions
                if item["action"] == action
            ]
        mean_rewards = action_selector.update_generation(generation_rewards)
        next_parameter_budget_fraction = controller_parameter_budget_fraction()
        controller_stats["generation_probability_history"].append(
            {
                "generation": len(alg_trace) + 1,
                "parameter_budget_fraction": float(
                    generation_parameter_budget_fraction
                ),
                "next_parameter_budget_fraction": float(next_parameter_budget_fraction),
                "action_counts": action_counts,
                "mean_rewards": mean_rewards,
                "quality_after_update": dict(action_selector.quality),
            }
        )
        costs = np.asarray(
            [
                _training_cost(candidate, setting_ns, seed_train)
                for candidate in generation_evaluated
            ],
            dtype=float,
        )
        best_index = int(np.argmin(costs))
        generation_best = deepcopy(generation_evaluated[best_index])
        metadata = dict(getattr(generation_best, "metadata", {}))
        metadata["generation"] = len(alg_trace) + 1
        metadata["cumulative_candidates"] = alg_n + evaluated_count
        metadata["cumulative_design_fes"] = design_runtime.total_evaluations
        generation_best.metadata = metadata
        alg_trace.append(generation_best)
        generation_representatives = []
        generation_evaluated = []
        generation_parent_queue = None
        generation_action_count = 0
        if checkpoint_path is not None and (
            len(alg_trace) % checkpoint_every == 0 or remaining_candidates() <= 0
        ):
            save_checkpoint(complete=False)
    controller_stats["training_design_fes"] = int(design_runtime.total_evaluations)
    setting_ns.SearchTrainingFEs = int(design_runtime.total_evaluations)
    setting_ns.SearchEvaluatedCandidates = int(alg_n + evaluated_count)
    if design_fe_limit is not None:
        setting_ns.SearchUnusedDesignFEs = (
            int(design_fe_limit) - design_runtime.total_evaluations
        )
    if seed_test:
        _progress(app, "Testing...")
        setting_ns.Evaluate = "exact"
        setting_ns.evaluate = "exact"
        design_runtime.evaluate(algs, setting_ns, seed_test)
    setting_ns.SearchArchive = list(search_archive)
    setting_ns.SearchControllerStats = controller_summary()
    if checkpoint_path is not None:
        save_checkpoint(complete=True)
    _progress(app, "Complete", done=True)
    design_runtime.close()
    return (algs[:alg_n], alg_trace)


def process(problem_descriptor: Any, *args, setting: Any, app: Any | None = None):
    """Main entry replicating MATLAB Process.m behaviour."""
    setting_ns = _normalize_setting(setting)
    _configure_rng(setting_ns)
    mode = str(
        getattr(setting_ns, "Mode", getattr(setting_ns, "mode", "design"))
    ).lower()
    if mode == "design":
        if len(args) < 2:
            raise ValueError(
                "Design mode requires instance_train and instance_test arguments"
            )
        (instance_train, instance_test) = args[:2]
        runtime_holder: dict[str, DesignEvaluationRuntime] = {}
        try:
            return _process_design(
                problem_descriptor,
                instance_train,
                instance_test,
                setting_ns,
                app,
                runtime_holder,
            )
        finally:
            runtime = runtime_holder.get("runtime")
            if runtime is not None:
                runtime.close()
    if mode == "solve":
        if len(args) < 1:
            raise ValueError("Solve mode requires instance list argument")
        instance_solve = args[0]
        problems = _build_problem_struct(problem_descriptor, instance_solve, setting_ns)
        construct_fn = _resolve_problem_callable(problem_descriptor)
        (problems, data, _) = construct_fn(problems, instance_solve, "construct")
        validate_constructed_problems(problems, data)
        (alg, setting_ns) = input_algorithm(setting_ns)
        if getattr(alg, "program_contract", None) is not None:
            setting_ns.ModelContractVersion = 1
            if alg.program_contract.get("execution") == "population_batch_v1":
                explicit = getattr(setting_ns, "_ExplicitOptions", frozenset())
                for name, value in (
                    ("ComponentBackend", "native"),
                    ("StreamBackend", "batch"),
                ):
                    if name not in explicit:
                        setattr(setting_ns,name,value)
                        setattr(setting_ns,name.lower(),value)
            from ...model_semantics import validate_design

            validate_design(
                alg,
                problems,
                backend=setting_ns.ComponentBackend,
                stream_backend=setting_ns.StreamBackend,
                require_installed=True,
            )
        _progress(app, "Solving...")
        (best_solutions, all_solutions) = run_algorithm(
            alg, problems, data, app, setting_ns
        )
        _progress(app, "Complete", done=True)
        return (best_solutions, all_solutions)
    raise ValueError("Mode must be 'design' or 'solve'.")
