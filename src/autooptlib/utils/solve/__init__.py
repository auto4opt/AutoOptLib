from __future__ import annotations

import hashlib
import json
import multiprocessing
import pickle
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ...components import get_component
from ..design._helpers import (
    Pathway,
    PathwayParam,
    SearchParam,
    SearchStep,
    ensure_rng,
    get_flex,
    get_problem_type,
)
from ..general.improve_rate import improve_rate

if TYPE_CHECKING:
    from ..design import Design


@dataclass
class Solution:
    dec: np.ndarray
    obj: float
    con: float
    fit: float
    acc: Any = None


class SolutionSet(Sequence[Solution]):
    def __init__(self, items: Iterable[Solution]):
        self._items: List[Solution] = list(items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    # Aggregations compatible with components
    def decs(self) -> np.ndarray:
        if not self._items:
            return np.zeros((0, 0))
        return np.vstack([s.dec for s in self._items])

    def objs(self) -> np.ndarray:
        return np.asarray([s.obj for s in self._items]).reshape(-1, 1)

    def cons(self) -> np.ndarray:
        return np.asarray([s.con for s in self._items]).reshape(-1, 1)

    def fits(self) -> np.ndarray:
        return np.asarray([s.fit for s in self._items]).reshape(-1, 1)


class ObjectiveEvaluationError(RuntimeError):
    """Raised when a user objective fails or returns an unsupported value."""


def _evaluation_cache_key(dec: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
    contiguous = np.ascontiguousarray(dec)
    return contiguous.dtype.str, contiguous.shape, contiguous.tobytes()


def repair_sol(dec: np.ndarray, problem: Any) -> np.ndarray:
    ptype = get_problem_type(problem)
    bound = getattr(problem, "bound", None)
    if bound is None:
        return dec
    bound = np.asarray(bound)
    if ptype == "continuous":
        lower = bound[0]
        upper = bound[1]
        return np.minimum(np.maximum(dec, lower), upper)
    if ptype == "discrete":
        lower = bound[0]
        upper = bound[1]
        repaired = np.clip(np.rint(dec), lower, upper).astype(int)
        setting = get_flex(problem, "setting", "")
        if isinstance(setting, str) and "dec_diff" in setting:
            n, d = repaired.shape
            for i in range(n):
                seen = set()
                duplicates = []
                for j, val in enumerate(repaired[i]):
                    if val in seen:
                        duplicates.append(j)
                    else:
                        seen.add(val)
                if not duplicates:
                    continue
                for idx in duplicates:
                    choices = [
                        v
                        for v in range(int(lower[idx]), int(upper[idx]) + 1)
                        if v not in repaired[i]
                    ]
                    if choices:
                        repaired[i, idx] = int(ensure_rng(problem).choice(choices))
            return repaired
        return repaired
    if ptype == "permutation":
        repaired = np.rint(dec).astype(int)
        if repaired.ndim == 1:
            rows = [repaired.copy()]
        else:
            rows = [row.copy() for row in repaired]
        d = len(rows[0])
        domain = list(range(1, d + 1))
        for row in rows:
            missing = [v for v in domain if v not in row]
            counts: dict[int, int] = {}
            duplicate_indices: list[int] = []
            for j, val in enumerate(row):
                counts[val] = counts.get(val, 0) + 1
                if counts[val] > 1:
                    duplicate_indices.append(j)
            for j in duplicate_indices:
                if missing:
                    row[j] = missing.pop(0)
        if repaired.ndim == 1:
            return rows[0]
        return np.vstack(rows)
    return dec


def _call_evaluator(problem: Any, data: Any, dec: np.ndarray):
    preview = np.asarray(dec).reshape(-1)[:8].tolist()
    cache = getattr(problem, "_autoopt_eval_cache", None)
    cache_key = None
    if isinstance(cache, dict):
        cache_key = _evaluation_cache_key(dec)
        lock = getattr(problem, "_autoopt_eval_lock", None)
        with lock if lock is not None else nullcontext():
            result = cache.get(cache_key)
            cache_hit = cache_key in cache
        if cache_hit:
            _log_evaluation(problem, dec, attempt=0, status="cache_hit", elapsed=0.0)
            return result

    retries = int(getattr(problem, "_autoopt_eval_retries", 0))
    last_error: ObjectiveEvaluationError | None = None
    for attempt in range(1, retries + 2):
        started = time.perf_counter()
        try:
            result = _evaluate_once(problem, data, dec, preview)
        except ObjectiveEvaluationError as exc:
            last_error = exc
            _log_evaluation(
                problem,
                dec,
                attempt=attempt,
                status="failure",
                elapsed=time.perf_counter() - started,
                error=str(exc),
            )
            continue
        _log_evaluation(
            problem,
            dec,
            attempt=attempt,
            status="success",
            elapsed=time.perf_counter() - started,
        )
        if isinstance(cache, dict) and cache_key is not None:
            lock = getattr(problem, "_autoopt_eval_lock", None)
            with lock if lock is not None else nullcontext():
                cache[cache_key] = result
        return result

    failure = getattr(problem, "_autoopt_eval_failure", "raise")
    if failure == "penalize":
        return float(getattr(problem, "_autoopt_eval_penalty", 1e30)), 0.0, None
    assert last_error is not None
    raise last_error


def _invoke_evaluator(problem: Any, data: Any, dec: np.ndarray):
    """Invoke the user callable without policy handling or normalization."""
    eval_fn = getattr(problem, "evaluate", None)
    if callable(eval_fn):
        return eval_fn(data, dec)
    name = getattr(problem, "name", None)
    if callable(name):
        return name(data, dec)
    raise NotImplementedError("Problem must provide an evaluate(data, dec) callable")


def _normalize_evaluator_output(out: Any, preview: list[Any]):
    """Normalize one evaluator result to objective, violation, and auxiliary."""
    # Normalize outputs: obj, con, acc(optional)
    if isinstance(out, tuple):
        if len(out) == 3:
            obj, con, acc = out
        elif len(out) == 2:
            obj, con = out
            acc = None
        elif len(out) == 1:
            obj, con, acc = out[0], 0.0, None
        else:
            raise ObjectiveEvaluationError(
                "Objective result tuples must contain one to three values "
                "(objective, optional constraint, optional auxiliary output)."
            )
    else:
        obj, con, acc = out, 0.0, None
    try:
        objective = np.asarray(obj, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ObjectiveEvaluationError(
            f"Objective did not return a numeric scalar for decision {preview}."
        ) from exc
    if objective.size != 1:
        raise ObjectiveEvaluationError(
            "AutoOptLib currently supports one scalar objective; "
            f"received shape {objective.shape} for decision {preview}."
        )
    objective_value = float(objective.reshape(-1)[0])
    if not np.isfinite(objective_value):
        raise ObjectiveEvaluationError(
            f"Objective returned a non-finite value for decision {preview}."
        )

    if con is None:
        constraint_value = 0.0
    else:
        try:
            constraints = np.asarray(con, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ObjectiveEvaluationError(
                f"Constraint did not return numeric values for decision {preview}."
            ) from exc
        if not np.all(np.isfinite(constraints)):
            raise ObjectiveEvaluationError(
                f"Constraint returned a non-finite value for decision {preview}."
            )
        constraint_value = float(np.sum(np.maximum(0.0, constraints)))
    return objective_value, constraint_value, acc


def _evaluate_payload(
    problem: Any, data: Any, dec: np.ndarray, preview: list[Any]
) -> tuple[float, float, Any]:
    """Evaluate certain or sampled-uncertain problem semantics."""
    behavior = get_flex(problem, "type", [])
    uncertainty = (
        str(behavior[2]).lower()
        if isinstance(behavior, Sequence) and len(behavior) > 2
        else "certain"
    )
    if uncertainty != "uncertain":
        return _normalize_evaluator_output(
            _invoke_evaluator(problem, data, dec), preview
        )

    sample_n = int(get_flex(problem, "sampleN", get_flex(problem, "sample_n", 0)))
    if sample_n <= 0:
        raise ObjectiveEvaluationError(
            "Uncertain problems must define a positive sampleN."
        )
    samples = [
        _normalize_evaluator_output(_invoke_evaluator(problem, data, dec), preview)
        for _ in range(sample_n)
    ]
    objectives = np.asarray([sample[0] for sample in samples], dtype=float)
    constraints = np.asarray([sample[1] for sample in samples], dtype=float)
    problem_setting = str(get_flex(problem, "setting", "")).lower()
    if "uncertain_average" in problem_setting:
        objective = float(np.mean(objectives))
        constraint = float(np.mean(constraints))
    elif "uncertain_worst" in problem_setting:
        objective = float(np.max(objectives))
        constraint = float(np.max(constraints))
    else:
        raise ObjectiveEvaluationError(
            "Uncertain problems must select 'uncertain_average' or "
            "'uncertain_worst' in problem.setting."
        )
    return objective, constraint, samples[-1][2]


def _timeout_worker(connection: Any, problem: Any, data: Any, dec: np.ndarray) -> None:
    """Evaluate and send a pickle-safe result from an isolated process."""
    try:
        preview = np.asarray(dec).reshape(-1)[:8].tolist()
        result = _evaluate_payload(problem, data, dec, preview)
        # Auxiliary values are not consumed by the execution engine and may be
        # unpickleable simulator handles. Do not transfer them across processes.
        connection.send((True, (result[0], result[1], None)))
    except BaseException as exc:  # child must report user failures to parent
        connection.send((False, f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def _evaluate_once(
    problem: Any, data: Any, dec: np.ndarray, preview: list[Any]
) -> tuple[float, float, Any]:
    timeout = getattr(problem, "_autoopt_eval_timeout", None)
    if timeout is None:
        try:
            return _evaluate_payload(problem, data, dec, preview)
        except ObjectiveEvaluationError:
            raise
        except Exception as exc:
            raise ObjectiveEvaluationError(
                f"Objective evaluation failed for decision {preview}: {exc}"
            ) from exc

    methods = multiprocessing.get_all_start_methods()
    method = "fork" if "fork" in methods else "spawn"
    context: Any = multiprocessing.get_context(method)
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_timeout_worker,
        args=(child, problem, data, np.asarray(dec)),
        daemon=True,
    )
    try:
        process.start()
    except Exception as exc:
        parent.close()
        child.close()
        raise ObjectiveEvaluationError(
            "Could not start the isolated objective worker. Use module-level, "
            "pickleable problem definitions on spawn-based platforms."
        ) from exc
    child.close()
    try:
        if not parent.poll(float(timeout)):
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=1.0)
            raise ObjectiveEvaluationError(
                f"Objective evaluation timed out after {float(timeout):g}s "
                f"for decision {preview}."
            )
        ok, payload = parent.recv()
        process.join(timeout=1.0)
        if not ok:
            raise ObjectiveEvaluationError(
                f"Objective evaluation failed for decision {preview}: {payload}"
            )
        return payload
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)


def _log_evaluation(
    problem: Any,
    dec: np.ndarray,
    *,
    attempt: int,
    status: str,
    elapsed: float,
    error: str | None = None,
) -> None:
    path = getattr(problem, "_autoopt_eval_log", None)
    if path is None:
        return
    array = np.ascontiguousarray(dec)
    event = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "problem": str(getattr(problem, "name", "problem")),
        "decision_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "decision_preview": array.reshape(-1)[:8].tolist(),
        "attempt": attempt,
        "status": status,
        "elapsed_seconds": elapsed,
    }
    if error is not None:
        event["error"] = error
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lock = getattr(problem, "_autoopt_eval_lock", None)
    with lock if lock is not None else nullcontext():
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def _configure_evaluation_runtime(problem: Any, setting: Any) -> None:
    """Attach validated, run-scoped evaluation controls to a problem record."""
    problem._autoopt_eval_retries = int(get_flex(setting, "EvalRetries", 0))
    problem._autoopt_eval_timeout = get_flex(setting, "EvalTimeoutSec", None)
    problem._autoopt_eval_failure = str(
        get_flex(setting, "EvalFailure", "raise")
    ).lower()
    problem._autoopt_eval_penalty = float(get_flex(setting, "EvalPenalty", 1e30))
    problem._autoopt_eval_log = get_flex(setting, "EvalLog", None)
    problem._autoopt_eval_workers = int(get_flex(setting, "EvalWorkers", 1))
    problem._autoopt_eval_lock = threading.RLock()
    if bool(get_flex(setting, "EvalCache", False)):
        if not isinstance(getattr(problem, "_autoopt_eval_cache", None), dict):
            problem._autoopt_eval_cache = {}
    elif hasattr(problem, "_autoopt_eval_cache"):
        delattr(problem, "_autoopt_eval_cache")


def _checkpoint_payload(
    *,
    solutions: SolutionSet,
    aux_cache: list[list[Any]],
    archives: list[Any],
    history: list[Optional[Solution]],
    fit_history: list[float],
    evaluations: int,
    elapsed_seconds: float,
    generation: int,
    rng: np.random.Generator,
    population_size: int,
    evaluation_budget: int,
    complete: bool,
) -> dict[str, Any]:
    return {
        "schema": "autooptlib.checkpoint",
        "schema_version": 1,
        "population_size": population_size,
        "evaluation_budget": evaluation_budget,
        "solutions": list(solutions),
        "aux_cache": aux_cache,
        "archives": archives,
        "history": history,
        "fit_history": fit_history,
        "evaluations": evaluations,
        "elapsed_seconds": elapsed_seconds,
        "generation": generation,
        "rng_state": rng.bit_generator.state,
        "complete": complete,
    }


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _load_checkpoint(
    path: Path, *, population_size: int, evaluation_budget: int
) -> dict[str, Any]:
    warnings.warn(
        "Loading an AutoOptLib checkpoint uses pickle and can execute arbitrary "
        "code. Resume only checkpoints produced by a trusted run.",
        UserWarning,
        stacklevel=2,
    )
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "autooptlib.checkpoint"
    ):
        raise ValueError(f"Invalid AutoOptLib checkpoint: {path}")
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported AutoOptLib checkpoint schema version.")
    if payload.get("population_size") != population_size:
        raise ValueError("Checkpoint population size does not match ProbN.")
    if payload.get("evaluation_budget") != evaluation_budget:
        raise ValueError("Checkpoint evaluation budget does not match ProbFE.")
    return payload


def _sequence_checkpoint_payload(
    *,
    solutions: Sequence[Solution],
    cumulative: float,
    remaining_budget: int,
    population_size: int,
    evaluation_budget: int,
    rng: np.random.Generator,
    complete: bool,
) -> dict[str, Any]:
    return {
        "schema": "autooptlib.sequence-checkpoint",
        "schema_version": 1,
        "population_size": population_size,
        "evaluation_budget": evaluation_budget,
        "solutions": list(solutions),
        "cumulative": cumulative,
        "remaining_budget": remaining_budget,
        "rng_state": rng.bit_generator.state,
        "complete": complete,
    }


def _load_sequence_checkpoint(
    path: Path, *, population_size: int, evaluation_budget: int
) -> dict[str, Any]:
    warnings.warn(
        "Loading an AutoOptLib checkpoint uses pickle and can execute arbitrary "
        "code. Resume only checkpoints produced by a trusted run.",
        UserWarning,
        stacklevel=2,
    )
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "autooptlib.sequence-checkpoint"
    ):
        raise ValueError(f"Invalid AutoOptLib sequential checkpoint: {path}")
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported AutoOptLib sequential checkpoint schema version.")
    if payload.get("population_size") != population_size:
        raise ValueError("Sequential checkpoint population size does not match ProbN.")
    if payload.get("evaluation_budget") != evaluation_budget:
        raise ValueError(
            "Sequential checkpoint evaluation budget does not match ProbFE."
        )
    return payload


def make_solutions(decs: np.ndarray, problem: Any, data: Any) -> SolutionSet:
    repaired: List[np.ndarray] = []
    for i in range(decs.shape[0]):
        d = np.array(decs[i], dtype=float)
        d = repair_sol(d, problem)
        repair_fn = getattr(problem, "repair", None)
        if callable(repair_fn):
            try:
                d = np.asarray(repair_fn(data, d))
            except TypeError:
                d = np.asarray(repair_fn(d))
        else:
            name = getattr(problem, "name", None)
            # Legacy MATLAB-style problem callables expose repair/evaluate
            # through ``name``.  Modern Python definitions expose ``evaluate``
            # separately and their construct callable must not be invoked as a
            # repair hook.
            if callable(name) and not callable(getattr(problem, "evaluate", None)):
                try:
                    repaired_output = name(data, d, "repair")
                except TypeError:
                    repaired_output = d
                if isinstance(repaired_output, tuple):
                    repaired_output = repaired_output[0]
                d = np.asarray(repaired_output)
        repaired.append(d)

    cache = getattr(problem, "_autoopt_eval_cache", None)
    unique_repaired = repaired
    original_keys: List[tuple[str, tuple[int, ...], bytes]] | None = None
    if isinstance(cache, dict):
        original_keys = [_evaluation_cache_key(decision) for decision in repaired]
        unique_by_key: dict[tuple[str, tuple[int, ...], bytes], np.ndarray] = {}
        for key, decision in zip(original_keys, repaired):
            unique_by_key.setdefault(key, decision)
        unique_repaired = list(unique_by_key.values())

    workers = int(getattr(problem, "_autoopt_eval_workers", 1))
    if workers == 1 or len(unique_repaired) <= 1:
        unique_results = [_call_evaluator(problem, data, d) for d in unique_repaired]
    else:
        with ThreadPoolExecutor(
            max_workers=min(workers, len(unique_repaired))
        ) as executor:
            unique_results = list(
                executor.map(
                    lambda d: _call_evaluator(problem, data, d), unique_repaired
                )
            )

    if original_keys is None:
        results = unique_results
    else:
        unique_results_by_key = {
            _evaluation_cache_key(decision): result
            for decision, result in zip(unique_repaired, unique_results)
        }
        seen: set[tuple[str, tuple[int, ...], bytes]] = set()
        results = []
        for key, decision in zip(original_keys, repaired):
            if key in seen:
                # The first in-flight duplicate has completed and populated the
                # cache, so this records a normal cache-hit event without a
                # second objective call.
                results.append(_call_evaluator(problem, data, decision))
            else:
                seen.add(key)
                results.append(unique_results_by_key[key])

    items: List[Solution] = []
    for d, (obj, con, acc) in zip(repaired, results):
        feasible = con <= 0.0
        fit = obj if feasible else (con + 1e8)
        items.append(Solution(dec=d, obj=obj, con=con, fit=fit, acc=acc))
    return SolutionSet(items)


def _init_population(
    rng: np.random.Generator, problem: Any, data: Any, setting: Any
) -> SolutionSet:
    ptype = get_problem_type(problem) or "continuous"
    pop_n = int(get_flex(problem, "N", get_flex(setting, "ProbN", 10)))
    if ptype == "continuous":
        bound = np.asarray(get_flex(problem, "bound"), dtype=float)
        lower = bound[0]
        upper = bound[1]
        decs = lower + (upper - lower) * rng.random((pop_n, lower.shape[-1]))
    elif ptype == "discrete":
        bound = np.asarray(get_flex(problem, "bound"), dtype=int)
        lower = bound[0]
        upper = bound[1]
        decs = np.vstack(
            [
                rng.integers(lower[j], upper[j] + 1, size=pop_n)
                for j in range(lower.shape[-1])
            ]
        ).T
    elif ptype == "permutation":
        bound = np.asarray(get_flex(problem, "bound"), dtype=int)
        if bound.ndim == 2 and bound.shape[1] > 0:
            d = bound.shape[1]
        else:
            d = int(get_flex(problem, "dimension", 0) or get_flex(problem, "D", 0))
            if d == 0:
                raise ValueError(
                    "Permutation problem requires explicit dimension or bound"
                )
        domain = np.arange(1, d + 1)
        decs = np.vstack([rng.permutation(domain) for _ in range(pop_n)])
    else:
        raise NotImplementedError(f"Unsupported problem type: {ptype}")
    return make_solutions(decs, problem, data)


def _split_indices(indices: Sequence[int], alg_p: int, total: int) -> List[List[int]]:
    array = np.asarray(indices, dtype=int).flatten()
    if array.size == 0:
        normalized_indices = list(range(total))
    else:
        normalized_indices = [int(value) for value in array.tolist()]
    if alg_p <= 1:
        return [normalized_indices]
    splits: List[List[int]] = []
    chunk = max(1, len(normalized_indices) // alg_p)
    start = 0
    for i in range(alg_p):
        if i == alg_p - 1:
            splits.append(normalized_indices[start:])
        else:
            splits.append(normalized_indices[start : start + chunk])
            start += chunk
    if not splits[-1]:
        splits[-1] = list(range(total))
    return splits


def _update_archives(
    solutions: SolutionSet,
    archive_names: Sequence[str],
    archives: List[Any],
    problem: Any,
) -> List[Any]:
    for index, name in enumerate(archive_names):
        archive_fn = get_component(name)
        if name in {"archive_best", "archive_statistic"}:
            archives[index], _ = archive_fn(solutions, archives[index], "execute")
        else:
            archives[index], _ = archive_fn(
                solutions, archives[index], problem, "execute"
            )
    return archives


def _execute_single_path(
    path: Any,
    params: Any,
    solutions: SolutionSet,
    aux_state: Any,
    problem: Any,
    data: Any,
    setting: Any,
    generation: int,
    remaining_evaluations: int,
    archive_names: Sequence[str],
    archives: List[Any],
) -> Tuple[SolutionSet, Any, int, int, List[Any], List[Solution]]:
    """Execute a serial pathway with an update after every search call.

    This mirrors the reference MATLAB execution loop: each inner search
    chooses from the current population, creates offspring, updates the
    population immediately, and lets the next search step see that update.
    """
    if not isinstance(aux_state, dict):
        aux_state = {}
    aux_state.setdefault("rng", ensure_rng(setting))
    current = solutions
    evaluations = 0
    iterations = 0
    update_fn = get_component(path.update)
    update_param = getattr(params, "update", None)
    iteration_bests: List[Solution] = []

    for step_index, step in enumerate(path.search):
        improve = None
        inner_generation = 1
        limit = int(step.termination[1]) if step.termination.size > 1 else 1
        threshold = float(step.termination[0]) if step.termination.size else -np.inf
        primary_param = params.search[step_index].primary if params.search else None
        secondary_param = params.search[step_index].secondary if params.search else None
        while (
            (improve is None or improve[0] >= threshold)
            and inner_generation <= limit
            and evaluations < remaining_evaluations
        ):
            component_generation = generation + iterations
            choose_fn = get_component(path.choose)
            choose_param = getattr(params, "choose", None)
            selected, choose_aux = choose_fn(
                current,
                problem,
                choose_param,
                aux_state,
                component_generation,
                inner_generation,
                data,
                "execute",
            )
            if isinstance(choose_aux, dict):
                aux_state = choose_aux
                aux_state.setdefault("rng", ensure_rng(setting))
            selected_indices = np.asarray(selected, dtype=int).reshape(-1)
            parent = SolutionSet([current[int(index)] for index in selected_indices])

            primary_fn = get_component(step.primary)
            new_dec, aux_state = primary_fn(
                parent,
                problem,
                primary_param,
                aux_state,
                component_generation,
                inner_generation,
                data,
                "execute",
            )
            if step.secondary:
                new_dec = repair_sol(np.asarray(new_dec), problem)
                secondary_fn = get_component(step.secondary)
                new_dec, aux_state = secondary_fn(
                    new_dec,
                    problem,
                    secondary_param,
                    aux_state,
                    component_generation,
                    inner_generation,
                    data,
                    "execute",
                )

            decisions = np.asarray(new_dec)
            if decisions.ndim == 1:
                decisions = decisions.reshape(1, -1)
            remaining = remaining_evaluations - evaluations
            if decisions.shape[0] > remaining:
                decisions = decisions[:remaining]
            new = make_solutions(decisions, problem, data)
            if step.primary == "search_cma":
                aux_state = get_component("para_cma")(
                    new, problem, aux_state, "solution"
                )
            elif step.primary == "search_pso":
                aux_state = get_component("para_pso")(new, problem, aux_state)

            if len(new) < len(current):
                updated = sorted(
                    list(current) + list(new), key=lambda solution: solution.fit
                )[: len(current)]
            else:
                updated, update_aux = update_fn(
                    list(current) + list(new),
                    problem,
                    update_param,
                    aux_state,
                    component_generation,
                    inner_generation,
                    data,
                    "execute",
                )
                if isinstance(update_aux, dict):
                    aux_state = update_aux
                    aux_state.setdefault("rng", ensure_rng(setting))
            current = SolutionSet(updated)
            evaluations += len(new)
            iterations += 1
            improve = improve_rate(current, improve, inner_generation, "solution")
            archives = _update_archives(current, archive_names, archives, problem)
            if len(current):
                iteration_bests.append(min(list(current), key=lambda sol: sol.fit))
            inner_generation += 1

    return current, aux_state, evaluations, iterations, archives, iteration_bests


def _execute_path(
    path: Any,
    params: Any,
    subset: SolutionSet,
    aux_state: Any,
    problem: Any,
    data: Any,
    setting: Any,
    generation: int,
    remaining_evaluations: int,
) -> Tuple[List[Any], Any, int]:
    if not isinstance(aux_state, dict):
        aux_state = {}
    aux_state.setdefault("rng", ensure_rng(setting))
    current_parent = subset
    evals = 0
    for step_idx, step in enumerate(path.search):
        if evals >= remaining_evaluations:
            break
        improve = None
        inner_g = 1
        limit = int(step.termination[1]) if step.termination.size > 1 else 1
        threshold = float(step.termination[0]) if step.termination.size > 0 else -np.inf
        primary_param = params.search[step_idx].primary if params.search else None
        secondary_param = params.search[step_idx].secondary if params.search else None
        while (improve is None or improve[0] >= threshold) and inner_g <= limit:
            remaining = remaining_evaluations - evals
            if remaining <= 0:
                break
            primary_fn = get_component(step.primary)
            new_dec, aux_state = primary_fn(
                current_parent,
                problem,
                primary_param,
                aux_state,
                generation,
                inner_g,
                data,
                "execute",
            )
            if step.secondary:
                new_dec = repair_sol(np.asarray(new_dec), problem)
                secondary_fn = get_component(step.secondary)
                new_dec, aux_state = secondary_fn(
                    new_dec,
                    problem,
                    secondary_param,
                    aux_state,
                    generation,
                    inner_g,
                    data,
                    "execute",
                )
            new_dec = np.asarray(new_dec)
            if new_dec.ndim == 1:
                new_dec = new_dec.reshape(1, -1)
            if new_dec.shape[0] > remaining:
                new_dec = new_dec[:remaining]
            new = make_solutions(new_dec, problem, data)
            if step.primary == "search_cma":
                aux_state = get_component("para_cma")(
                    new, problem, aux_state, "solution"
                )
            elif step.primary == "search_pso":
                aux_state = get_component("para_pso")(new, problem, aux_state)
            current_parent = new
            evals += len(new)
            improve = improve_rate(new, improve, inner_g, "solution")
            inner_g += 1
    return list(current_parent), aux_state, evals


def run_design(
    pathways: List[Any], params: List[Any], problem: Any, data: Any, setting: Any
) -> dict:
    rng = ensure_rng(setting)
    _configure_evaluation_runtime(problem, setting)
    try:
        problem.rng = rng
    except (AttributeError, TypeError):
        pass
    population_size = int(get_flex(problem, "N", get_flex(setting, "ProbN", 10)))
    evaluation_budget = int(
        get_flex(
            setting,
            "ProbFE",
            population_size * int(get_flex(problem, "Gmax", 10)),
        )
    )
    if evaluation_budget < population_size:
        raise ValueError(
            f"ProbFE ({evaluation_budget}) must be at least the initial population size "
            f"({population_size})."
        )

    checkpoint_value = get_flex(setting, "_checkpoint_path", None)
    checkpoint_path = Path(checkpoint_value) if checkpoint_value else None
    resume = bool(get_flex(setting, "Resume", False))
    checkpoint_every = int(get_flex(setting, "CheckpointEvery", 1))

    embedded_archives = getattr(pathways[0], "archive", []) if pathways else []
    archive_names = list(embedded_archives or get_flex(setting, "archive", []))
    resume_state = None
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        resume_state = _load_checkpoint(
            checkpoint_path,
            population_size=population_size,
            evaluation_budget=evaluation_budget,
        )
        solutions = SolutionSet(resume_state["solutions"])
        aux_cache = resume_state["aux_cache"]
        archives = resume_state["archives"]
        rng.bit_generator.state = resume_state["rng_state"]
    else:
        solutions = _init_population(rng, problem, data, setting)
        # MATLAB keeps one Aux structure per pathway and shares it among the
        # choose, search, parameter-update and population-update components.
        aux_cache = [{} for _ in pathways]
        archives = []
        for name in archive_names:
            if name == "archive_statistic":
                archives.append(np.empty((0, 2)))
            else:
                archives.append([])

    metric = get_flex(setting, "metric", "quality")
    tmax = get_flex(setting, "tmax", np.inf)
    if tmax is None:
        tmax = np.inf
    thres_val = get_flex(setting, "thres", -np.inf)
    thres = float(thres_val if thres_val is not None else -np.inf)
    gmax = int(get_flex(problem, "Gmax", 10))

    if metric == "quality":
        Tmax = np.inf
        threshold = thres
    elif metric == "auc":
        Tmax = np.inf
        threshold = -np.inf
    elif metric == "runtimeFE":
        requested_limit = evaluation_budget if not np.isfinite(tmax) else int(tmax)
        Tmax = min(evaluation_budget, requested_limit)
        threshold = thres
    elif metric == "runtimeSec":
        Tmax = float(tmax)
        threshold = thres
    else:
        Tmax = np.inf
        threshold = thres

    history: List[Optional[Solution]] = []
    fit_history: List[float] = []
    # Initial-population objective calls are part of the public ProbFE budget.
    evaluations = len(solutions)
    elapsed_seconds = 0.0
    generation = 1

    if resume_state is not None:
        history = resume_state["history"]
        fit_history = resume_state["fit_history"]
        evaluations = int(resume_state["evaluations"])
        elapsed_seconds = float(resume_state["elapsed_seconds"])
        generation = int(resume_state["generation"])
        if resume_state.get("complete", False):
            candidates = [item for item in history if item is not None]
            best = min(candidates, key=lambda item: item.fit) if candidates else None
            return {
                "solutions": solutions,
                "history": history,
                "fit_history": fit_history,
                "evaluations": evaluations,
                "elapsed": elapsed_seconds,
                "archives": archives,
                "best_solution": best,
            }

    if not history:
        if len(solutions):
            initial_best = min(list(solutions), key=lambda s: s.fit)
            history.append(initial_best)
            fit_history.append(float(initial_best.fit))
        else:
            history.append(None)
            fit_history.append(np.inf)

    while True:
        if generation > gmax:
            break
        if evaluations >= evaluation_budget:
            break
        if metric == "quality" and fit_history[-1] <= threshold:
            break
        if metric == "runtimeFE" and (
            evaluations >= Tmax or fit_history[-1] <= threshold
        ):
            break
        if metric == "runtimeSec" and elapsed_seconds >= Tmax:
            break

        iteration_start = time.perf_counter()

        iterations_used = 1
        if len(pathways) == 1:
            remaining = evaluation_budget - evaluations
            if metric == "runtimeFE":
                remaining = min(remaining, int(Tmax) - evaluations)
            (
                solutions,
                aux_cache[0],
                evals,
                iterations_used,
                archives,
                iteration_bests,
            ) = _execute_single_path(
                pathways[0],
                params[0],
                solutions,
                aux_cache[0],
                problem,
                data,
                setting,
                generation,
                remaining,
                archive_names,
                archives,
            )
            evaluations += evals
            for current_best in iteration_bests:
                previous = history[-1]
                best_so_far = (
                    current_best
                    if previous is None or current_best.fit < previous.fit
                    else previous
                )
                history.append(best_so_far)
                fit_history.append(float(best_so_far.fit))
            if evals == 0:
                break
        else:
            choose_fn = get_component(pathways[0].choose)
            choose_param = getattr(params[0], "choose", None)
            selected, _ = choose_fn(
                solutions,
                problem,
                choose_param,
                {"rng": rng},
                generation,
                1,
                data,
                "execute",
            )
            selected = np.asarray(selected, dtype=int).reshape(-1)
            index_groups = _split_indices(selected, len(pathways), len(solutions))

            new_items: List[Any] = []
            for p_idx, path in enumerate(pathways):
                remaining = evaluation_budget - evaluations
                if metric == "runtimeFE":
                    remaining = min(remaining, int(Tmax) - evaluations)
                if remaining <= 0:
                    break
                subset_idx = (
                    index_groups[p_idx]
                    if p_idx < len(index_groups)
                    else index_groups[-1]
                )
                subset = (
                    SolutionSet([solutions[i] for i in subset_idx])
                    if subset_idx
                    else solutions
                )
                produced, aux_state, evals = _execute_path(
                    path,
                    params[p_idx],
                    subset,
                    aux_cache[p_idx],
                    problem,
                    data,
                    setting,
                    generation,
                    remaining,
                )
                aux_cache[p_idx] = aux_state
                new_items.extend(produced)
                evaluations += evals

            if not new_items:
                break

            if len(new_items) < len(solutions):
                updated = sorted(
                    list(solutions) + new_items, key=lambda solution: solution.fit
                )[: len(solutions)]
            else:
                update_fn = get_component(pathways[0].update)
                update_param = getattr(params[0], "update", None)
                updated, _ = update_fn(
                    list(solutions) + new_items,
                    problem,
                    update_param,
                    {"rng": rng},
                    generation,
                    1,
                    data,
                    "execute",
                )
            solutions = SolutionSet(updated)
            archives = _update_archives(solutions, archive_names, archives, problem)
            current_best = min(list(solutions), key=lambda sol: sol.fit)
            previous = history[-1]
            best_so_far = (
                current_best
                if previous is None or current_best.fit < previous.fit
                else previous
            )
            history.append(best_so_far)
            fit_history.append(float(best_so_far.fit))

        if metric == "runtimeFE":
            pass
        elif metric == "runtimeSec":
            elapsed_seconds += time.perf_counter() - iteration_start

        generation += max(1, iterations_used)
        if checkpoint_path is not None and generation % checkpoint_every == 0:
            _write_checkpoint(
                checkpoint_path,
                _checkpoint_payload(
                    solutions=solutions,
                    aux_cache=aux_cache,
                    archives=archives,
                    history=history,
                    fit_history=fit_history,
                    evaluations=evaluations,
                    elapsed_seconds=elapsed_seconds,
                    generation=generation,
                    rng=rng,
                    population_size=population_size,
                    evaluation_budget=evaluation_budget,
                    complete=False,
                ),
            )

    candidates = [item for item in history if item is not None]
    final_solution = min(candidates, key=lambda item: item.fit) if candidates else None

    if checkpoint_path is not None:
        _write_checkpoint(
            checkpoint_path,
            _checkpoint_payload(
                solutions=solutions,
                aux_cache=aux_cache,
                archives=archives,
                history=history,
                fit_history=fit_history,
                evaluations=evaluations,
                elapsed_seconds=elapsed_seconds,
                generation=generation,
                rng=rng,
                population_size=population_size,
                evaluation_budget=evaluation_budget,
                complete=True,
            ),
        )

    return {
        "solutions": solutions,
        "history": history,
        "fit_history": fit_history,
        "evaluations": evaluations,
        "elapsed": elapsed_seconds,
        "archives": archives,
        "best_solution": final_solution,
    }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _placeholder_solution() -> Solution:
    return Solution(
        dec=np.zeros((1, 0)), obj=float("inf"), con=float("inf"), fit=float("inf")
    )


def _ensure_solution_instance(obj: Any) -> Solution:
    if isinstance(obj, Solution):
        return obj
    if obj is None:
        return _placeholder_solution()
    if hasattr(obj, "dec") and hasattr(obj, "fit"):
        return Solution(
            dec=np.asarray(getattr(obj, "dec")),
            obj=float(getattr(obj, "obj", np.inf)),
            con=float(getattr(obj, "con", np.inf)),
            fit=float(getattr(obj, "fit", np.inf)),
        )
    return _placeholder_solution()


# ---------------------------------------------------------------------------
# Solve-mode helpers (InputAlg / RunAlg translation)
# ---------------------------------------------------------------------------


def _normalize_setting(setting: Any) -> Any:
    if isinstance(setting, dict):
        return type("Setting", (), setting)()
    for attr in dir(setting):
        if attr[0].isupper():
            setattr(setting, attr.lower(), getattr(setting, attr))
    return setting


def _to_array(value: Any) -> Optional[np.ndarray]:
    if value is None or (isinstance(value, (list, tuple)) and len(value) == 0):
        return None
    arr = np.asarray(value, dtype=float)
    return arr.reshape(-1) if arr.ndim > 0 else arr


def _make_search_step(
    primary: str, secondary: Optional[str], termination: Sequence[float]
) -> SearchStep:
    term = np.asarray(termination, dtype=float).reshape(-1)
    return SearchStep(
        primary=primary, secondary=secondary if secondary else None, termination=term
    )


def _make_search_param(primary: Any, secondary: Any) -> SearchParam:
    return SearchParam(primary=_to_array(primary), secondary=_to_array(secondary))


def _build_default_algorithm(name: str, setting: Any) -> "Design":
    name = name.strip().lower()
    tmpl: dict[str, list[Any]] = {
        "continuous genetic algorithm": [
            (
                "choose_tournament",
                [
                    (
                        "cross_sim_binary",
                        "search_mu_polynomial",
                        (-np.inf, 1),
                        [20.0],
                        [0.2, 20.0],
                    ),
                ],
                "update_round_robin",
                [],
            ),
        ],
        "evolutionary programming": [
            (
                "choose_tournament",
                [
                    ("search_mu_gaussian", None, (-np.inf, 1), None, None),
                ],
                "update_round_robin",
                [],
            ),
        ],
        "fast evolutionary programming": [
            (
                "choose_tournament",
                [
                    ("search_mu_cauchy", None, (-np.inf, 1), None, None),
                ],
                "update_round_robin",
                [],
            ),
        ],
        "cma-es": [
            (
                "choose_traverse",
                [
                    ("search_cma", None, (-np.inf, 1), None, None),
                ],
                "update_greedy",
                [],
            ),
        ],
        "estimation of distribution": [
            (
                "choose_traverse",
                [
                    ("search_eda", None, (-np.inf, 1), None, None),
                ],
                "update_greedy",
                [],
            ),
        ],
        "particle swarm optimization": [
            (
                "choose_traverse",
                [
                    ("search_pso", None, (-np.inf, 1), [0.9], None),
                ],
                "update_pairwise",
                [],
            ),
        ],
        "differential evolution": [
            (
                "choose_traverse",
                [
                    ("search_de_current", None, (-np.inf, 1), [0.9, 0.1], None),
                ],
                "update_pairwise",
                [],
            ),
        ],
        "continuous random search": [
            (
                "choose_traverse",
                [
                    ("reinit_continuous", None, (-np.inf, 1), None, None),
                ],
                "update_greedy",
                [],
            ),
        ],
        "ica": [
            (
                "choose_ica",
                [
                    ("search_ica", None, (-np.inf, 1), [0.6, 0.5], None),
                ],
                "update_greedy",
                [],
            ),
        ],
        "discrete genetic algorithm": [
            (
                "choose_tournament",
                [
                    (
                        "cross_point_uniform",
                        "search_reset_rand",
                        (-np.inf, 1),
                        [0.2],
                        [0.2],
                    ),
                ],
                "update_round_robin",
                [],
            ),
        ],
        "discrete iterative local search": [
            (
                "choose_traverse",
                [
                    ("search_reset_one", None, (0.05, 10), None, None),
                    ("reinit_discrete", None, (-np.inf, 1), None, None),
                ],
                "update_greedy",
                [],
            ),
        ],
        "discrete simulated annealing": [
            (
                "choose_traverse",
                [
                    ("search_reset_one", None, (-np.inf, 1), None, None),
                ],
                "update_simulated_annealing",
                [],
                np.array([0.1]),
            ),
        ],
        "discrete random search": [
            (
                "choose_traverse",
                [
                    ("reinit_discrete", None, (-np.inf, 1), None, None),
                ],
                "update_greedy",
                [],
            ),
        ],
        "permutation genetic algorithm": [
            (
                "choose_tournament",
                [
                    ("cross_order_two", "search_swap", (-np.inf, 1), None, None),
                ],
                "update_round_robin",
                [],
            ),
        ],
        "permutation iterative local search": [
            (
                "choose_traverse",
                [
                    ("search_insert", None, (0.05, 10), None, None),
                    ("reinit_permutation", None, (-np.inf, 1), None, None),
                ],
                "update_greedy",
                [],
            ),
        ],
        "permutation simulated annealing": [
            (
                "choose_traverse",
                [
                    ("search_insert", None, (-np.inf, 1), None, None),
                ],
                "update_simulated_annealing",
                [],
                np.array([0.1]),
            ),
        ],
        "permutation variable neighborhood search": [
            (
                "choose_traverse",
                [
                    ("search_swap", None, (0.05, 10), None, None),
                    ("search_scramble", None, (0.05, 10), None, None),
                    ("search_insert", None, (0.05, 10), None, None),
                ],
                "update_greedy",
                [],
            ),
        ],
        "permutation random search": [
            (
                "choose_traverse",
                [
                    ("reinit_permutation", None, (-np.inf, 1), None, None),
                ],
                "update_greedy",
                [],
            ),
        ],
    }

    if name not in tmpl:
        raise NotImplementedError(
            f"Preset algorithm {name!r} is not available in the Python translation."
        )

    pathways_cfg = tmpl[name]
    pathways: List[Pathway] = []
    params: List[PathwayParam] = []
    for item in pathways_cfg:
        choose = item[0]
        search_entries = item[1]
        update = item[2]
        archive = item[3] if len(item) > 3 else []
        update_param = item[4] if len(item) > 4 else None

        search_steps = []
        search_params = []
        for entry in search_entries:
            primary, secondary, termination, primary_param, secondary_param = entry
            search_steps.append(_make_search_step(primary, secondary, termination))
            search_params.append(_make_search_param(primary_param, secondary_param))

        pathways.append(
            Pathway(choose=choose, search=search_steps, update=update, archive=archive)
        )
        params.append(
            PathwayParam(
                choose=None,
                search=search_params,
                update=_to_array(update_param),
            )
        )

    from ..design import Design

    design = Design()
    design.construct([pathways], [params])
    return design


def input_algorithm(setting: Any) -> Tuple["Design", Any]:
    setting = _normalize_setting(setting)
    alg_file = getattr(setting, "AlgFile", getattr(setting, "alg_file", ""))
    if alg_file and str(alg_file).lower() != "none":
        path = Path(alg_file)
        if not path.exists():
            raise FileNotFoundError(f"Algorithm file {alg_file} not found.")
        if path.suffix.lower() == ".json":
            from ...serialization import load_algorithm

            alg = load_algorithm(path)
        else:
            warnings.warn(
                "Loading pickle algorithm files can execute arbitrary code. Only load trusted "
                "files; use AutoOptLib JSON algorithm files for portable exchange.",
                UserWarning,
                stacklevel=2,
            )
            with path.open("rb") as handle:
                data = pickle.load(handle)
            if isinstance(data, dict) and "algs" in data:
                alg = data["algs"][0]
            else:
                alg = data[0] if isinstance(data, (list, tuple)) else data
        from ..design import Design

        if not isinstance(alg, Design):
            raise TypeError("Loaded algorithm is not a Design instance.")
        setting.AlgP = len(getattr(alg, "operator_pheno", []) or [])
        return alg, setting

    alg_name = getattr(setting, "AlgName", getattr(setting, "alg_name", "")).strip()
    if not alg_name:
        raise ValueError("Please specify AlgFile or AlgName in solve mode.")
    design = _build_default_algorithm(alg_name.lower(), setting)
    setting.AlgP = len(design.operator_pheno or [])
    return design, setting


def _extract_mode(problem: Any) -> str:
    behavior = get_flex(problem, "type", ["continuous", "static"])
    if isinstance(behavior, (list, tuple)) and len(behavior) > 1:
        return str(behavior[1])
    return "static"


def _setting_with_budget(setting: Any, budget: int) -> Any:
    """Return a shallow setting copy with synchronized ProbFE aliases."""
    cloned = copy(setting)
    if isinstance(cloned, dict):
        cloned["ProbFE"] = int(budget)
        cloned["prob_fe"] = int(budget)
    else:
        setattr(cloned, "ProbFE", int(budget))
        setattr(cloned, "prob_fe", int(budget))
    return cloned


def _advance_sequential_problem(
    current_problem: Any, current_data: Any, best: Solution
) -> tuple[Any, Any] | None:
    next_fn = getattr(current_problem, "advance_sequence", None)
    if callable(next_fn):
        return next_fn(best, current_data)
    name = getattr(current_problem, "name", None)
    if callable(name):
        next_problem, next_data, _ = name(
            current_problem, current_data, best, "sequence"
        )
        return next_problem, next_data
    return None


def run_algorithm(
    alg: "Design", problems: Sequence[Any], data: Sequence[Any], app: Any, setting: Any
):
    if not alg.operator_pheno or not alg.parameter_pheno:
        raise ValueError("The algorithm must be decoded before solve execution.")
    pathways = alg.operator_pheno[0]
    params = alg.parameter_pheno[0]
    alg_runs = int(get_flex(setting, "AlgRuns", 1))
    best_solutions: List[List[Solution]] = []
    all_solutions: List[List[Solution]] = []

    for idx, problem in enumerate(problems):
        data_obj = data[idx] if idx < len(data) else None
        mode = _extract_mode(problem)
        run_histories: List[List[Solution]] = []
        run_best: List[Solution] = []
        fitness_sequence: List[float] = []

        for run in range(alg_runs):
            run_setting = setting
            checkpoint_dir = get_flex(setting, "CheckpointDir", None)
            if checkpoint_dir is not None:
                run_setting = copy(setting)
                if mode == "static":
                    run_setting._checkpoint_path = str(
                        Path(checkpoint_dir) / f"instance_{idx + 1}_run_{run + 1}.pkl"
                    )
                elif mode == "sequential":
                    run_setting._sequence_checkpoint_path = str(
                        Path(checkpoint_dir)
                        / f"instance_{idx + 1}_run_{run + 1}_sequence.pkl"
                    )
            if mode == "static":
                result = run_design(pathways, params, problem, data_obj, run_setting)
                history = [
                    sol for sol in result["history"] if isinstance(sol, Solution)
                ]
                best = result["best_solution"]
                if best is None and history:
                    best = history[-1]
                run_histories.append(history)
                run_best.append(_ensure_solution_instance(best))
            elif mode == "sequential":
                current_problem = problem
                current_data = data_obj
                sequence_solutions: List[Solution] = []
                cumulative = 0.0
                total_budget = int(get_flex(setting, "ProbFE", 0))
                remaining_budget = total_budget
                population_size = int(
                    get_flex(current_problem, "N", get_flex(setting, "ProbN", 1))
                )
                sequence_path_value = get_flex(
                    run_setting, "_sequence_checkpoint_path", None
                )
                sequence_path = (
                    Path(sequence_path_value) if sequence_path_value else None
                )
                sequence_state = None
                sequence_rng = ensure_rng(run_setting)
                if (
                    bool(get_flex(run_setting, "Resume", False))
                    and sequence_path is not None
                    and sequence_path.exists()
                ):
                    sequence_state = _load_sequence_checkpoint(
                        sequence_path,
                        population_size=population_size,
                        evaluation_budget=total_budget,
                    )
                    sequence_solutions = list(sequence_state["solutions"])
                    cumulative = float(sequence_state["cumulative"])
                    remaining_budget = int(sequence_state["remaining_budget"])
                    sequence_rng.bit_generator.state = sequence_state["rng_state"]

                if sequence_state is not None and sequence_state.get("complete", False):
                    run_histories.append(sequence_solutions)
                    run_best.append(
                        sequence_solutions[-1]
                        if sequence_solutions
                        else _placeholder_solution()
                    )
                    fitness_sequence.append(cumulative)
                    continue

                for completed_solution in sequence_solutions:
                    advanced = _advance_sequential_problem(
                        current_problem, current_data, completed_solution
                    )
                    if advanced is None:
                        break
                    current_problem, current_data = advanced

                while getattr(current_data, "continue", False):
                    # Each sequential instance receives the full configured
                    # budget, matching the original sequence protocol.
                    stage_setting = run_setting
                    if checkpoint_dir is not None:
                        stage_number = len(sequence_solutions) + 1
                        stage_path = (
                            Path(checkpoint_dir)
                            / f"instance_{idx + 1}_run_{run + 1}_stage_{stage_number}.pkl"
                        )
                        if isinstance(stage_setting, dict):
                            stage_setting["_checkpoint_path"] = str(stage_path)
                        else:
                            stage_setting._checkpoint_path = str(stage_path)
                    result = run_design(
                        pathways, params, current_problem, current_data, stage_setting
                    )
                    best = result["best_solution"]
                    if best is None and result["history"]:
                        history = [
                            sol
                            for sol in result["history"]
                            if isinstance(sol, Solution)
                        ]
                        best = history[-1] if history else None
                    if best is None:
                        break
                    sequence_solutions.append(best)
                    cumulative += float(best.fit)
                    if sequence_path is not None:
                        _write_checkpoint(
                            sequence_path,
                            _sequence_checkpoint_payload(
                                solutions=sequence_solutions,
                                cumulative=cumulative,
                                remaining_budget=remaining_budget,
                                population_size=population_size,
                                evaluation_budget=total_budget,
                                rng=sequence_rng,
                                complete=False,
                            ),
                        )
                    advanced = _advance_sequential_problem(
                        current_problem, current_data, best
                    )
                    if advanced is None:
                        break
                    current_problem, current_data = advanced
                    if current_problem is problem and current_data is data_obj:
                        break
                if sequence_path is not None:
                    _write_checkpoint(
                        sequence_path,
                        _sequence_checkpoint_payload(
                            solutions=sequence_solutions,
                            cumulative=cumulative,
                            remaining_budget=remaining_budget,
                            population_size=population_size,
                            evaluation_budget=total_budget,
                            rng=sequence_rng,
                            complete=True,
                        ),
                    )
                run_histories.append(sequence_solutions)
                run_best.append(
                    sequence_solutions[-1]
                    if sequence_solutions
                    else _placeholder_solution()
                )
                fitness_sequence.append(cumulative)
            else:
                raise NotImplementedError(f"Unsupported problem mode: {mode}")

            if app is not None and hasattr(app, "TextArea"):
                pct = (idx * alg_runs + run + 1) / (len(problems) * alg_runs)
                app.TextArea.Value = f"Solving... {pct * 100:.1f}%"

        if mode == "static":
            best_idx = int(np.argmin([b.fit for b in run_best]))
            all_solutions.append(run_histories[best_idx])
        else:
            best_idx = int(np.argmin(fitness_sequence)) if fitness_sequence else 0
            all_solutions.append(run_histories[best_idx])
        best_solutions.append(run_best)

    return best_solutions, all_solutions


__all__ = [
    "Solution",
    "SolutionSet",
    "repair_sol",
    "make_solutions",
    "run_design",
    "input_algorithm",
    "run_algorithm",
]
