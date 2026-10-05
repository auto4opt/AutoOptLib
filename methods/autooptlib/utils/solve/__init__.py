from __future__ import annotations

import hashlib
import inspect
import json
import multiprocessing
import pickle
import time
import warnings
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ...components import (
    _custom_component_snapshot,
    _restore_custom_components,
    component_category,
    component_parent_arity,
    get_component,
)
from ...runtime import (
    DesignEvaluationRuntime,
    EvaluationRuntimeConfig,
    PersistentProcessExecutor,
    RuntimeStatistics,
    ScheduledTask,
    TaskResources,
)
from ...runtime.lifecycle import (
    finalize_evaluation_session,
    initialize_evaluation_session,
)
from ...runtime.resources import discover_cpu_topology
from ..design._helpers import (
    ensure_rng,
    get_flex,
    get_problem_type,
)
from ..design._population import (
    copy_configuration,
    is_mutation_component,
    normalize_configuration,
)
from ..design._stream_graph import (
    STREAM_IMPLEMENTATION_REVISION,
    StreamPathway,
    StreamPathwayParam,
)
from ..design._stream_graph import (
    validate_phenotype as validate_stream_phenotype,
)
from ..general.improve_rate import improve_rate

if TYPE_CHECKING:
    from ..design import Design


STREAM_MAX_ZERO_EVALUATION_GENERATIONS = 1000


@dataclass
class Solution:
    dec: np.ndarray
    obj: float
    con: float
    fit: float
    acc: Any = None


@dataclass
class _PendingStreamOffspring:
    """One invalid final FastGA offspring awaiting generation-end evaluation."""

    decisions: np.ndarray
    sources: "SolutionSet"
    invalidated: np.ndarray


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
        strategy = str(get_flex(problem, "_AlgorithmBoundaryHandling", "clip")).lower()
        finite = np.isfinite(dec)
        midpoint = lower + 0.5 * (upper - lower)
        if strategy == "clip":
            sanitized = np.where(
                np.isnan(dec),
                midpoint,
                np.where(
                    np.isposinf(dec), upper, np.where(np.isneginf(dec), lower, dec)
                ),
            )
            return np.minimum(np.maximum(sanitized, lower), upper)
        if strategy == "reflect":
            span = upper - lower
            fixed = span == 0
            safe_span = np.where(fixed, 1.0, span)
            sanitized = np.where(finite, dec, midpoint)
            phase = np.mod(sanitized - lower, 2.0 * safe_span)
            reflected = lower + np.where(
                phase <= safe_span, phase, 2.0 * safe_span - phase
            )
            return np.where(fixed, lower, reflected)
        if strategy == "resample":
            repaired = np.array(dec, dtype=float, copy=True)
            outside = (repaired < lower) | (repaired > upper) | ~np.isfinite(repaired)
            if np.any(outside):
                rng = ensure_rng(problem)
                sampled = lower + (upper - lower) * rng.random(repaired.shape)
                repaired[outside] = sampled[outside]
            return repaired
        raise ValueError(f"Unsupported continuous boundary strategy: {strategy!r}.")
    if ptype == "discrete":
        lower = bound[0]
        upper = bound[1]
        repaired = np.clip(np.rint(dec), lower, upper).astype(int)
        setting = get_flex(problem, "setting", "")
        if isinstance(setting, str) and "dec_diff" in setting:
            was_vector = repaired.ndim == 1
            rows = np.atleast_2d(repaired).copy()
            if rows.ndim != 2:
                raise ValueError("Discrete decisions must be a vector or matrix.")
            n, d = rows.shape
            lower_values = np.broadcast_to(np.asarray(lower), (d,))
            upper_values = np.broadcast_to(np.asarray(upper), (d,))
            rng = ensure_rng(problem)
            for i in range(n):
                # Solve the all-different constraint as a bipartite matching.
                # Greedily replacing only repeated positions is incorrect for
                # heterogeneous domains: an early choice can consume the sole
                # feasible value of a later position even when a repair exists.
                allowed: list[list[int]] = []
                for j in range(d):
                    values = np.arange(
                        int(lower_values[j]), int(upper_values[j]) + 1, dtype=int
                    )
                    values = rng.permutation(values).tolist()
                    current = int(rows[i, j])
                    if current in values:
                        values.remove(current)
                        values.insert(0, current)
                    allowed.append(values)

                value_to_position: dict[int, int] = {}

                def assign(position: int, visited: set[int]) -> bool:
                    for value in allowed[position]:
                        if value in visited:
                            continue
                        visited.add(value)
                        incumbent = value_to_position.get(value)
                        if incumbent is None or assign(incumbent, visited):
                            value_to_position[value] = position
                            return True
                    return False

                for position in range(d):
                    if not assign(position, set()):
                        raise ValueError(
                            "The discrete dec_diff bounds cannot represent an "
                            "all-different decision vector."
                        )
                for value, position in value_to_position.items():
                    rows[i, position] = value
            return rows[0] if was_vector else rows
        return repaired
    if ptype == "permutation":
        repaired = np.rint(dec).astype(int)
        if repaired.ndim not in (1, 2):
            raise ValueError("Permutation decisions must be a vector or matrix.")
        if repaired.size == 0:
            return repaired.copy()
        if repaired.ndim == 1:
            rows = [repaired.copy()]
        else:
            rows = [row.copy() for row in repaired]
        d = len(rows[0])
        domain = list(range(1, d + 1))
        domain_set = set(domain)
        for row in rows:
            seen: set[int] = set()
            invalid_indices: list[int] = []
            for j, val in enumerate(row):
                value = int(val)
                if value not in domain_set or value in seen:
                    invalid_indices.append(j)
                else:
                    seen.add(value)
            missing = [value for value in domain if value not in seen]
            for j, value in zip(invalid_indices, missing):
                row[j] = value
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


def _serialized_timeout_worker(connection: Any, payload: bytes) -> None:
    """Spawn-safe timeout entry point for closures and local problem classes."""

    import cloudpickle

    problem, data, dec = cloudpickle.loads(payload)
    _timeout_worker(connection, problem, data, dec)


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
    if method == "fork":
        worker_target: Any = _timeout_worker
        worker_arguments: tuple[Any, ...] = (child, problem, data, np.asarray(dec))
    else:
        import cloudpickle

        transferable_problem = copy(problem)
        # The per-process cache lock is an execution detail and is not
        # serializable on spawn-only platforms. The isolated child performs
        # one call and does not need it.
        try:
            transferable_problem._autoopt_eval_lock = None
        except (AttributeError, TypeError):
            pass
        try:
            payload = cloudpickle.dumps(
                (transferable_problem, data, np.asarray(dec)),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        except Exception as exc:
            parent.close()
            child.close()
            raise ObjectiveEvaluationError(
                "Could not serialize the problem and data for an isolated "
                "objective timeout worker."
            ) from exc
        worker_target = _serialized_timeout_worker
        worker_arguments = (child, payload)
    process = context.Process(target=worker_target, args=worker_arguments, daemon=True)
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
    # Population-individual parallelism is intentionally unsupported. Runtime
    # workers are allocated only to candidate algorithms or independent runs.
    problem._autoopt_eval_workers = 1
    # Each runtime worker invokes objectives serially. Keeping a thread lock on
    # the problem would add no protection and would make an already-evaluated
    # problem impossible to serialize when switching to process execution.
    problem._autoopt_eval_lock = None
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
    checkpoint_signature: str,
    complete: bool,
    stream_events: int = 0,
    termination_reason: str = "running",
) -> dict[str, Any]:
    return {
        "schema": "autooptlib.checkpoint",
        "schema_version": 2,
        "population_size": population_size,
        "evaluation_budget": evaluation_budget,
        "checkpoint_signature": checkpoint_signature,
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
        "stream_events": int(stream_events),
        "termination_reason": str(termination_reason),
    }


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    *,
    population_size: int,
    evaluation_budget: int,
    checkpoint_signature: str,
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
    if payload.get("schema_version") != 2:
        raise ValueError("Unsupported AutoOptLib checkpoint schema version.")
    if payload.get("population_size") != population_size:
        raise ValueError("Checkpoint population size does not match ProbN.")
    if payload.get("evaluation_budget") != evaluation_budget:
        raise ValueError("Checkpoint evaluation budget does not match ProbFE.")
    if payload.get("checkpoint_signature") != checkpoint_signature:
        raise ValueError(
            "Checkpoint semantics do not match the current algorithm, problem, "
            "data, or evaluation settings (checkpoint_signature mismatch)."
        )
    return payload


def _sequence_checkpoint_payload(
    *,
    solutions: Sequence[Solution],
    cumulative: float,
    population_size: int,
    evaluation_budget: int,
    checkpoint_signature: str,
    rng: np.random.Generator,
    complete: bool,
) -> dict[str, Any]:
    return {
        "schema": "autooptlib.sequence-checkpoint",
        "schema_version": 2,
        "population_size": population_size,
        "evaluation_budget": evaluation_budget,
        "checkpoint_signature": checkpoint_signature,
        "solutions": list(solutions),
        "cumulative": cumulative,
        "rng_state": rng.bit_generator.state,
        "complete": complete,
    }


def _load_sequence_checkpoint(
    path: Path,
    *,
    population_size: int,
    evaluation_budget: int,
    checkpoint_signature: str,
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
    if payload.get("schema_version") != 2:
        raise ValueError("Unsupported AutoOptLib sequential checkpoint schema version.")
    if payload.get("population_size") != population_size:
        raise ValueError("Sequential checkpoint population size does not match ProbN.")
    if payload.get("evaluation_budget") != evaluation_budget:
        raise ValueError(
            "Sequential checkpoint evaluation budget does not match ProbFE."
        )
    if payload.get("checkpoint_signature") != checkpoint_signature:
        raise ValueError(
            "Sequential checkpoint semantics do not match the current algorithm, "
            "problem, data, or evaluation settings (checkpoint_signature mismatch)."
        )
    return payload


def _checkpoint_value(value: Any, seen: set[int] | None = None) -> Any:
    """Return a deterministic, compact representation of execution semantics."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isnan(number):
            return {"float": "nan"}
        if np.isposinf(number):
            return {"float": "inf"}
        if np.isneginf(number):
            return {"float": "-inf"}
        return number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            return {
                "array": _checkpoint_value(array.tolist(), seen),
                "shape": list(array.shape),
                "dtype": array.dtype.str,
            }
        return {
            "array_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "shape": list(array.shape),
            "dtype": array.dtype.str,
        }
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return {"recursive": f"{type(value).__module__}.{type(value).__qualname__}"}
    if isinstance(value, dict):
        seen.add(identity)
        try:
            return {
                str(key): _checkpoint_value(item, seen)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        finally:
            seen.remove(identity)
    if isinstance(value, (list, tuple)):
        seen.add(identity)
        try:
            return [_checkpoint_value(item, seen) for item in value]
        finally:
            seen.remove(identity)
    if isinstance(value, (set, frozenset)):
        encoded = [_checkpoint_value(item, seen) for item in value]
        return sorted(
            encoded,
            key=lambda item: json.dumps(item, sort_keys=True, default=repr),
        )
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        qualname = getattr(value, "__qualname__", type(value).__qualname__)
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
        result: dict[str, Any] = {
            "callable": f"{module}.{qualname}",
            "implementation_sha256": hashlib.sha256(implementation).hexdigest(),
        }
        defaults = getattr(value, "__defaults__", None)
        if defaults:
            result["defaults"] = _checkpoint_value(defaults, seen)
        keyword_defaults = getattr(value, "__kwdefaults__", None)
        if keyword_defaults:
            result["keyword_defaults"] = _checkpoint_value(keyword_defaults, seen)
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
                    stable_closure.append(_checkpoint_value(item, seen))
                else:
                    stable_closure.append(
                        {
                            "mutable_type": f"{type(item).__module__}.{type(item).__qualname__}"
                        }
                    )
            result["stable_closure"] = stable_closure
        explicit_semantics = getattr(value, "__autoopt_checkpoint_semantics__", None)
        if explicit_semantics is not None:
            result["semantics"] = _checkpoint_value(explicit_semantics, seen)
        # Callable objects often carry immutable configuration. Functions and
        # bound methods are intentionally not fingerprinted through closures or
        # instance counters, which may change during a legitimate resume.
        if not inspect.isfunction(value) and not inspect.ismethod(value):
            state = getattr(value, "__dict__", None)
            if isinstance(state, dict):
                public = {
                    key: item
                    for key, item in state.items()
                    if not key.startswith("_") and key != "rng"
                }
                if public:
                    result["state"] = _checkpoint_value(public, seen)
        return result
    state = getattr(value, "__dict__", None)
    if isinstance(state, dict):
        seen.add(identity)
        try:
            public = {
                key: item
                for key, item in state.items()
                if not key.startswith("_autoopt_") and key != "rng"
            }
            return {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "state": _checkpoint_value(public, seen),
            }
        finally:
            seen.remove(identity)
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _solve_checkpoint_signature(
    pathways: Sequence[Any],
    params: Sequence[Any],
    problem: Any,
    data: Any,
    setting: Any,
    archive_names: Sequence[str],
) -> str:
    """Fingerprint everything that can change a resumed solve trajectory."""

    setting_names = (
        "ProbFE",
        "ProbN",
        "Metric",
        "Tmax",
        "Thres",
        "Seed",
        "InitialPopulations",
        "_InitialPopulation",
        "_AlgorithmPopulationSize",
        "_AlgorithmOffspringSize",
        "_AlgorithmCrossoverRate",
        "_AlgorithmMutationRate",
        "_AlgorithmBoundaryHandling",
        "EvalRetries",
        "EvalTimeoutSec",
        "EvalFailure",
        "EvalPenalty",
        "StreamEventBudgetMultiplier",
        "EvalCache",
        "CheckpointSignature",
    )
    component_names: set[str] = set(archive_names)
    for pathway in pathways:
        if isinstance(pathway, StreamPathway):
            component_names.update(
                str(stage.choose)
                for stage in pathway.stages
                if stage.choose is not None
            )
        else:
            component_names.add(str(pathway.choose))
        component_names.add(str(pathway.update))
        for step in pathway.search:
            component_names.add(str(step.primary))
            if step.secondary:
                component_names.add(str(step.secondary))
    components = {
        name: _checkpoint_value(get_component(name)) for name in sorted(component_names)
    }
    payload = {
        "implementation": "solve-checkpoint-audit2",
        "stream_implementation": (
            STREAM_IMPLEMENTATION_REVISION
            if any(isinstance(pathway, StreamPathway) for pathway in pathways)
            else None
        ),
        "pathways": _checkpoint_value(pathways),
        "parameters": _checkpoint_value(params),
        "problem": _checkpoint_value(problem),
        "data": _checkpoint_value(data),
        "archives": list(archive_names),
        "components": components,
        "settings": {
            name: _checkpoint_value(get_flex(setting, name, None))
            for name in setting_names
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _signature_accepts(callable_obj: Any, *arguments: Any) -> bool | None:
    """Return whether a callable accepts arguments without executing it."""

    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return None
    try:
        signature.bind(*arguments)
    except TypeError:
        return False
    return True


def _apply_problem_repair(
    repair_fn: Any, data: Any, decision: np.ndarray
) -> np.ndarray:
    """Call a one- or two-argument repair hook without masking its errors."""

    accepts_data = _signature_accepts(repair_fn, data, decision)
    if accepts_data is True:
        return np.asarray(repair_fn(data, decision))
    accepts_decision = _signature_accepts(repair_fn, decision)
    if accepts_decision is True:
        return np.asarray(repair_fn(decision))
    if accepts_data is None and accepts_decision is None:
        # The documented contract is repair(data, decision). Native callables
        # without inspectable signatures are invoked once; a TypeError raised
        # inside the hook must remain visible to the caller.
        return np.asarray(repair_fn(data, decision))
    raise TypeError("A problem repair hook must accept (data, decision) or (decision).")


def _repair_decision(decision: np.ndarray, problem: Any, data: Any) -> np.ndarray:
    """Apply the public decision-repair contract without evaluating it."""

    d = np.array(decision, dtype=float)
    bound = getattr(problem, "bound", None)
    expected_dimension = (
        np.asarray(bound).shape[-1] if bound is not None else d.shape[-1]
    )
    if d.shape != (expected_dimension,):
        raise ValueError(
            "Candidate decision dimension does not match the problem bounds."
        )
    d = repair_sol(d, problem)
    repair_fn = getattr(problem, "repair", None)
    if callable(repair_fn):
        d = _apply_problem_repair(repair_fn, data, d)
    else:
        name = getattr(problem, "name", None)
        # Legacy MATLAB-style problem callables expose repair/evaluate through
        # ``name``. Modern Python definitions expose ``evaluate`` separately.
        if callable(name) and not callable(getattr(problem, "evaluate", None)):
            if _signature_accepts(name, data, d, "repair") is True:
                repaired_output = name(data, d, "repair")
                if isinstance(repaired_output, tuple):
                    repaired_output = repaired_output[0]
                d = np.asarray(repaired_output)
    if d.shape != (expected_dimension,):
        raise ValueError(
            "A problem repair hook must return one decision vector with "
            "the original dimension."
        )
    return d


def make_solutions(decs: np.ndarray, problem: Any, data: Any) -> SolutionSet:
    decs = np.asarray(decs)
    if decs.ndim != 2:
        raise ValueError("Candidate decisions must be a two-dimensional matrix.")
    repaired = [_repair_decision(decs[i], problem, data) for i in range(decs.shape[0])]

    cache = getattr(problem, "_autoopt_eval_cache", None)
    unique_repaired = repaired
    original_keys: List[tuple[str, tuple[int, ...], bytes]] | None = None
    if isinstance(cache, dict):
        original_keys = [_evaluation_cache_key(decision) for decision in repaired]
        unique_by_key: dict[tuple[str, tuple[int, ...], bytes], np.ndarray] = {}
        for key, decision in zip(original_keys, repaired):
            unique_by_key.setdefault(key, decision)
        unique_repaired = list(unique_by_key.values())

    # CPU parallelism is deliberately scheduled at the algorithm/run level.
    # Keeping a run's population evaluation local avoids nested process pools,
    # preserves algorithm barriers, and keeps the black-box environment resident.
    unique_results = [_call_evaluator(problem, data, d) for d in unique_repaired]

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
    pop_n = int(
        get_flex(
            setting,
            "_AlgorithmPopulationSize",
            get_flex(problem, "N", get_flex(setting, "ProbN", 10)),
        )
    )
    supplied = get_flex(setting, "_InitialPopulation", None)
    if supplied is not None:
        decs = np.asarray(supplied)
        if decs.ndim != 2:
            raise ValueError("Each supplied initial population must be a 2-D array.")
        dimension = np.asarray(get_flex(problem, "bound")).shape[-1]
        if decs.shape[1] != dimension:
            raise ValueError(
                "Supplied initial-population dimension does not match the problem."
            )
        if decs.shape[0] < pop_n:
            raise ValueError(
                "A supplied initial population must contain at least ProbN rows."
            )
        decs = np.array(decs[:pop_n], copy=True)
    elif ptype == "continuous":
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
    return [
        np.asarray(split, dtype=int).reshape(-1).tolist()
        for split in np.array_split(np.asarray(normalized_indices, dtype=int), alg_p)
    ]


def _resize_parent_indices(
    indices: Sequence[int],
    *,
    population_size: int,
    offspring_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return exactly ``offspring_size`` valid parent indices.

    Selection components still determine preference and ordering.  When
    lambda exceeds the number selected, additional parents are sampled from
    that selected set with replacement; when lambda is smaller, its leading
    selected parents are retained.
    """

    selected = np.asarray(indices, dtype=int).reshape(-1)
    selected = selected[(selected >= 0) & (selected < population_size)]
    if selected.size == 0:
        selected = np.arange(population_size, dtype=int)
    if selected.size >= offspring_size:
        return selected[:offspring_size]
    extra = rng.choice(
        selected, size=offspring_size - selected.size, replace=True
    ).astype(int)
    return np.concatenate((selected, extra))


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
    offspring_size = int(get_flex(setting, "_AlgorithmOffspringSize", len(solutions)))

    for step_index, step in enumerate(path.search):
        improve = None
        inner_generation = 1
        limit = int(step.termination[1]) if step.termination.size > 1 else 1
        threshold = float(step.termination[0]) if step.termination.size else -np.inf
        while (
            (improve is None or improve[0] >= threshold)
            and inner_generation <= limit
            and evaluations < remaining_evaluations
        ):
            component_generation = generation + iterations
            # Stateful population operators (PSO/CMA) must continue to see the
            # configured lambda even in the final partial FE batch. Surplus
            # decisions are truncated immediately before objective calls.
            target_offspring = offspring_size
            parent_count = target_offspring * component_parent_arity(step.primary)
            choose_fn = get_component(path.choose)
            choose_param = getattr(params, "choose", None)
            aux_state["_selection_count"] = parent_count
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
            aux_state.pop("_selection_count", None)
            if isinstance(choose_aux, dict):
                aux_state = choose_aux
                aux_state.setdefault("rng", ensure_rng(setting))
            selected_indices = _resize_parent_indices(
                selected,
                population_size=len(current),
                offspring_size=parent_count,
                rng=ensure_rng(setting),
            )
            parent = SolutionSet([current[int(index)] for index in selected_indices])

            step_params = params.search[step_index] if params.search else None
            aux_state["_population_context"] = current
            aux_state["_target_population_indices"] = selected_indices
            new_dec, aux_state = _execute_search_operators(
                parent,
                step,
                step_params,
                aux_state,
                problem,
                data,
                setting,
                component_generation,
                inner_generation,
                output_count=target_offspring,
            )
            aux_state.pop("_population_context", None)
            aux_state.pop("_target_population_indices", None)

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

            aux_state["_active_search_operator"] = str(step.primary)
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
            aux_state.pop("_active_search_operator", None)
            if len(updated) != len(current):
                updated = sorted(list(updated), key=lambda solution: solution.fit)[
                    : len(current)
                ]
            current = SolutionSet(updated)
            evaluations += len(new)
            iterations += 1
            improve = improve_rate(current, improve, inner_generation, "solution")
            archives = _update_archives(current, archive_names, archives, problem)
            if len(current):
                iteration_bests.append(min(list(current), key=lambda sol: sol.fit))
            inner_generation += 1

    return current, aux_state, evaluations, iterations, archives, iteration_bests


def _execute_search_operators(
    parent: SolutionSet,
    step: Any,
    step_params: Any,
    aux_state: Any,
    problem: Any,
    data: Any,
    setting: Any,
    generation: int,
    inner_generation: int,
    *,
    output_count: int | None = None,
) -> tuple[np.ndarray, Any]:
    """Execute one primary/secondary pair with algorithm-level call rates.

    For a crossover-primary step, ``crossover_rate`` chooses crossed versus
    cloned offspring independently, while ``mutation_rate`` independently
    controls application of the paired secondary mutation. The same mutation
    rate gates a mutation used directly as the primary search. Configurations
    loaded from older artifacts default both rates to one and retain legacy
    behavior.
    """

    if not isinstance(aux_state, dict):
        aux_state = {}
    rng = ensure_rng(setting)
    aux_state.setdefault("rng", rng)
    primary_param = getattr(step_params, "primary", None)
    secondary_param = getattr(step_params, "secondary", None)
    primary_fn = get_component(step.primary)
    crossed_or_searched, aux_state = primary_fn(
        parent,
        problem,
        primary_param,
        aux_state,
        generation,
        inner_generation,
        data,
        "execute",
    )
    decisions = np.asarray(crossed_or_searched)
    if decisions.ndim == 1:
        decisions = decisions.reshape(1, -1)
    if decisions.ndim != 2:
        raise ValueError(
            f"Search component {step.primary!r} must return a 2-D decision matrix."
        )

    primary_is_crossover = str(step.primary).startswith("cross_")
    primary_is_mutation = is_mutation_component(step.primary)
    if output_count is not None:
        decisions = decisions[: int(output_count)]
    crossover_mask = np.ones(decisions.shape[0], dtype=bool)
    parent_decisions = np.asarray([item.dec for item in parent])
    if output_count is not None:
        parent_decisions = parent_decisions[: int(output_count)]
    if parent_decisions.shape != decisions.shape:
        raise ValueError(
            f"Search component {step.primary!r} returned shape {decisions.shape}; "
            f"expected {parent_decisions.shape}."
        )
    if primary_is_crossover:
        crossover_rate = float(get_flex(setting, "_AlgorithmCrossoverRate", 1.0))
        crossover_mask = rng.random(decisions.shape[0]) < crossover_rate
        decisions = np.where(crossover_mask[:, None], decisions, parent_decisions)
    elif primary_is_mutation:
        mutation_rate = float(get_flex(setting, "_AlgorithmMutationRate", 1.0))
        mutation_mask = rng.random(decisions.shape[0]) < mutation_rate
        decisions = np.where(mutation_mask[:, None], decisions, parent_decisions)

    if step.secondary:
        decisions = repair_sol(decisions, problem)
        mutation_rate = float(get_flex(setting, "_AlgorithmMutationRate", 1.0))
        secondary_mask = rng.random(decisions.shape[0]) < mutation_rate
        if np.any(secondary_mask):
            secondary_fn = get_component(step.secondary)
            changed, aux_state = secondary_fn(
                decisions[secondary_mask],
                problem,
                secondary_param,
                aux_state,
                generation,
                inner_generation,
                data,
                "execute",
            )
            changed = np.asarray(changed)
            expected_shape = decisions[secondary_mask].shape
            if changed.shape != expected_shape:
                raise ValueError(
                    f"Secondary component {step.secondary!r} returned shape "
                    f"{changed.shape}; expected {expected_shape}."
                )
            decisions = decisions.copy()
            decisions[secondary_mask] = changed
    return decisions, aux_state


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
    output_count: int | None = None,
) -> Tuple[List[Any], Any, int]:
    """Execute every search row in one branch, in graph order.

    Multi-path genotypes intentionally share their outer choose and update
    nodes.  The outer executor selects and partitions parents once, each
    branch then applies its complete search sequence, and the branch outputs
    are joined for one shared update.  Intermediate rows are evaluated because
    later operators may use their fitness or auxiliary state.
    """

    if not isinstance(aux_state, dict):
        aux_state = {}
    aux_state.setdefault("rng", ensure_rng(setting))
    if not path.search or remaining_evaluations <= 0:
        return [], aux_state, 0

    current = subset
    target_count = len(subset) if output_count is None else max(0, int(output_count))
    evaluations = 0
    for step_index, step in enumerate(path.search):
        remaining = remaining_evaluations - evaluations
        if remaining <= 0 or len(current) == 0:
            break
        step_params = (
            params.search[step_index]
            if getattr(params, "search", None) and step_index < len(params.search)
            else None
        )
        required = target_count * component_parent_arity(step.primary)
        parent_indices = _resize_parent_indices(
            np.arange(len(current), dtype=int),
            population_size=len(current),
            offspring_size=required,
            rng=ensure_rng(setting),
        )
        parents = SolutionSet([current[int(index)] for index in parent_indices])
        aux_state["_population_context"] = current
        aux_state["_target_population_indices"] = parent_indices
        decisions, aux_state = _execute_search_operators(
            parents,
            step,
            step_params,
            aux_state,
            problem,
            data,
            setting,
            generation + step_index,
            1,
            output_count=target_count,
        )
        aux_state.pop("_population_context", None)
        aux_state.pop("_target_population_indices", None)
        decisions = np.asarray(decisions)
        if decisions.ndim == 1:
            decisions = decisions.reshape(1, -1)
        decisions = decisions[:remaining]
        current = make_solutions(decisions, problem, data)
        evaluations += len(current)
        if step.primary == "search_cma":
            aux_state = get_component("para_cma")(
                current, problem, aux_state, "solution"
            )
        elif step.primary == "search_pso":
            aux_state = get_component("para_pso")(current, problem, aux_state)
    return list(current), aux_state, evaluations


def _stream_choose(
    pool: SolutionSet,
    population: SolutionSet,
    stage: Any,
    stage_params: Any,
    aux_state: dict[str, Any],
    problem: Any,
    data: Any,
    setting: Any,
    generation: int,
    count: int,
) -> tuple[SolutionSet, dict[str, Any]]:
    """Run one stage-local selector against its incoming solution stream."""

    if stage.choose is None:
        if pool is population:
            aux_state["_target_population_indices"] = np.arange(len(pool), dtype=int)
        else:
            aux_state.pop("_target_population_indices", None)
        return pool, aux_state
    choose_fn = get_component(stage.choose)
    choose_param = getattr(stage_params, "choose", None)
    aux_state["_selection_count"] = int(count)
    selected, choose_aux = choose_fn(
        pool,
        problem,
        choose_param,
        aux_state,
        generation,
        1,
        data,
        "execute",
    )
    aux_state.pop("_selection_count", None)
    if isinstance(choose_aux, dict):
        aux_state = choose_aux
        aux_state.setdefault("rng", ensure_rng(setting))
    aux_state.pop("_selection_count", None)
    indices = _resize_parent_indices(
        selected,
        population_size=len(pool),
        offspring_size=int(count),
        rng=ensure_rng(setting),
    )
    if pool is population:
        aux_state["_target_population_indices"] = np.asarray(indices, dtype=int)
    else:
        aux_state.pop("_target_population_indices", None)
    return SolutionSet([pool[int(index)] for index in indices]), aux_state


def _stream_evaluate_changed(
    decisions: np.ndarray,
    sources: SolutionSet,
    problem: Any,
    data: Any,
    remaining_evaluations: int,
    invalidated: Any = None,
) -> tuple[SolutionSet | None, int, list[Solution]]:
    """Evaluate invalidated decisions while preserving valid FastGA clones."""

    array = np.asarray(decisions)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    repaired = [_repair_decision(row, problem, data) for row in array]
    force = np.zeros(len(repaired), dtype=bool)
    if invalidated is not None:
        raw_force = np.asarray(invalidated, dtype=bool).reshape(-1)
        if raw_force.size == 1:
            force[:] = bool(raw_force[0])
        elif raw_force.size == len(force):
            force[:] = raw_force
        else:
            raise ValueError(
                "Stream invalidation mask must contain one value per offspring."
            )
    resolved: list[Solution | None] = []
    changed: list[np.ndarray] = []
    for index, decision in enumerate(repaired):
        clone = None
        if not force[index]:
            clone = next(
                (
                    source
                    for source in sources
                    if np.array_equal(np.asarray(source.dec), decision)
                ),
                None,
            )
        resolved.append(clone)
        if clone is None:
            changed.append(decision)
    # eoEvalCounterThrowException evaluates the threshold-th invalid
    # individual and throws immediately afterwards.  Consequently an event
    # that consumes the final available FE is interrupted just like an event
    # that would overrun the budget: later selectors and replacement are not
    # reached, while the completed objective call still counts and contributes
    # to best-so-far.
    if changed and len(changed) >= int(remaining_evaluations):
        allowed = max(0, int(remaining_evaluations))
        evaluated = (
            list(make_solutions(np.vstack(changed[:allowed]), problem, data))
            if allowed
            else []
        )
        # eoFastGA's evaluation counter stops in the middle of a pair or
        # offspring batch.  Those completed objective calls still contribute
        # to best-so-far, but the interrupted event never reaches replacement.
        return None, len(evaluated), evaluated
    evaluated = (
        list(make_solutions(np.vstack(changed), problem, data)) if changed else []
    )
    iterator = iter(evaluated)
    completed = [item if item is not None else next(iterator) for item in resolved]
    return SolutionSet(completed), len(changed), evaluated


def _execute_stream_event(
    path: StreamPathway,
    params: StreamPathwayParam,
    population: SolutionSet,
    aux_state: Any,
    problem: Any,
    data: Any,
    setting: Any,
    generation: int,
    remaining_evaluations: int,
    mutation_rate: float,
    maximum_outputs: int,
) -> tuple[
    list[Solution | _PendingStreamOffspring] | None,
    dict[str, Any],
    int,
    list[Solution],
]:
    """Execute one pathway event and return its completed offspring stream."""

    path_state = dict(aux_state) if isinstance(aux_state, dict) else {}
    path_state.setdefault("rng", ensure_rng(setting))
    raw_stage_states = path_state.get("_stream_stage_aux", [])
    stage_states = [
        dict(value) if isinstance(value, dict) else {} for value in raw_stage_states
    ]
    while len(stage_states) < len(path.stages):
        stage_states.append({})
    for value in stage_states:
        value.setdefault("rng", ensure_rng(setting))
    pool = population
    evaluations = 0
    observed: list[Solution] = []
    crossed = False
    rng = ensure_rng(setting)
    for stage_index, (stage, stage_params) in enumerate(
        zip(path.stages, params.stages)
    ):
        state = stage_states[stage_index]
        state["_stream_generation_target"] = int(
            get_flex(setting, "_AlgorithmOffspringSize", len(population))
        )
        parent_count = component_parent_arity(stage.search.primary)
        selected, state = _stream_choose(
            pool,
            population,
            stage,
            stage_params,
            state,
            problem,
            data,
            setting,
            generation + stage_index,
            parent_count,
        )
        primary = str(stage.search.primary)
        execute = True
        if crossed and is_mutation_component(primary):
            # eoFastGA tosses this coin only after after-cross selection.  Do
            # not precompute stage flags: doing so changes every intervening
            # selector/crossover RNG event even though the marginal rate is
            # unchanged.
            execute = bool(rng.random() < mutation_rate)
        if not execute:
            # In eoFastGA the after-cross selector still selects one of the
            # evaluated pair when the optional mutation coin is false.
            pool = selected
            stage_states[stage_index] = state
            continue
        state["_population_context"] = population
        search_fn = get_component(stage.search.primary)
        decisions, search_aux = search_fn(
            selected,
            problem,
            getattr(stage_params.search, "primary", None),
            state,
            generation + stage_index,
            1,
            data,
            "execute",
        )
        if isinstance(search_aux, dict):
            state = search_aux
            state.setdefault("rng", ensure_rng(setting))
        invalidated = state.pop("_stream_invalidated_mask", None)
        if invalidated is None and not str(stage.search.primary).startswith(
            "search_bit_"
        ):
            # Legacy Search evaluates every executed variation result.  Only
            # FastGA bit mutations expose a meaningful valid-clone outcome via
            # an unchanged bit string; compatibility crossovers provide an
            # explicit mask above.  Treating arbitrary equal coordinates as a
            # clone silently gives those components free objective calls.
            invalidated = True
        state.pop("_population_context", None)
        state.pop("_target_population_indices", None)
        defer_final_fastga = stage_index == len(path.stages) - 1 and primary.startswith(
            ("search_fastga_bit_", "search_fastga_de_")
        )
        if defer_final_fastga:
            decision_array = np.asarray(decisions)
            if decision_array.ndim == 1:
                decision_array = decision_array.reshape(1, -1)
            decision_array = decision_array[: max(0, int(maximum_outputs))]
            invalidation_array = np.asarray(invalidated, dtype=bool).reshape(-1)
            if invalidation_array.size == 1:
                invalidation_array = np.repeat(invalidation_array, len(decision_array))
            else:
                invalidation_array = invalidation_array[: len(decision_array)]
            if invalidation_array.size != len(decision_array):
                raise ValueError(
                    "A final FastGA mutation must return one invalidation flag "
                    "per offspring."
                )
            results: list[Solution | _PendingStreamOffspring] = []
            for decision, force in zip(decision_array, invalidation_array):
                if not bool(force):
                    clone = next(
                        (
                            source
                            for source in selected
                            if np.array_equal(np.asarray(source.dec), decision)
                        ),
                        None,
                    )
                    if clone is None:
                        raise RuntimeError(
                            "A valid FastGA mutation clone changed its decision values."
                        )
                    results.append(clone)
                else:
                    results.append(
                        _PendingStreamOffspring(
                            np.asarray(decision).reshape(1, -1),
                            selected,
                            np.asarray([True]),
                        )
                    )
            stage_states[stage_index] = state
            path_state["_stream_stage_aux"] = stage_states
            return results, path_state, evaluations, observed
        decision_array = np.asarray(decisions)
        if decision_array.ndim == 1:
            decision_array = decision_array.reshape(1, -1)
        if stage_index == len(path.stages) - 1:
            decision_array = decision_array[: max(0, int(maximum_outputs))]
            if invalidated is not None:
                raw_invalidated = np.asarray(invalidated, dtype=bool).reshape(-1)
                if raw_invalidated.size > 1:
                    invalidated = raw_invalidated[: len(decision_array)]
        produced, used, newly_evaluated = _stream_evaluate_changed(
            decision_array,
            selected,
            problem,
            data,
            remaining_evaluations - evaluations,
            invalidated,
        )
        if produced is None:
            evaluations += used
            observed.extend(newly_evaluated)
            stage_states[stage_index] = state
            path_state["_stream_stage_aux"] = stage_states
            return None, path_state, evaluations, observed
        pool = produced
        if primary == "search_cma":
            disturbances = np.asarray(state.get("cma_Disturb"), dtype=float)
            if disturbances.ndim != 2 or len(disturbances) < len(produced):
                raise RuntimeError(
                    "Stream CMA did not retain one disturbance per evaluated sample."
                )
            pending_disturbances = state.setdefault("_stream_cma_disturbances", [])
            pending_solutions = state.setdefault("_stream_cma_solutions", [])
            pending_disturbances.extend(disturbances[: len(produced)].copy())
            pending_solutions.extend(list(produced))
        evaluations += used
        observed.extend(newly_evaluated)
        stage_states[stage_index] = state
        crossed |= primary.startswith("cross_")
    path_state["_stream_stage_aux"] = stage_states
    return list(pool), path_state, evaluations, observed


def _execute_stream_generation(
    pathways: Sequence[StreamPathway],
    params: Sequence[StreamPathwayParam],
    solutions: SolutionSet,
    aux_cache: list[Any],
    problem: Any,
    data: Any,
    setting: Any,
    generation: int,
    remaining_evaluations: int,
    archive_names: Sequence[str],
    archives: list[Any],
    remaining_stream_events: int | None = None,
) -> tuple[SolutionSet, list[Any], int, list[Any], list[Solution], int]:
    """Execute decoded pathways and apply their terminal update components."""

    if not pathways or len(pathways) != len(params):
        raise ValueError(
            "Search v6.0 requires matching non-empty pathways and parameters."
        )
    rng = ensure_rng(setting)
    target = int(get_flex(setting, "_AlgorithmOffspringSize", len(solutions)))
    crossover_rate = float(get_flex(setting, "_AlgorithmCrossoverRate", 1.0))
    mutation_rate = float(get_flex(setting, "_AlgorithmMutationRate", 1.0))
    crossover_paths = [
        index
        for index, path in enumerate(pathways)
        if any(stage.search.primary.startswith("cross_") for stage in path.stages)
    ]
    noncrossover_paths = [
        index for index in range(len(pathways)) if index not in crossover_paths
    ]
    offspring: list[tuple[int, Solution | _PendingStreamOffspring]] = []
    observed: list[Solution] = []
    evaluations = 0
    attempts = 0
    interrupted = False
    maximum_attempts = max(10, target * 10)
    if remaining_stream_events is not None:
        maximum_attempts = min(maximum_attempts, max(0, int(remaining_stream_events)))
    while (
        len(offspring) < target
        and evaluations < remaining_evaluations
        and attempts < maximum_attempts
    ):
        attempts += 1
        if len(crossover_paths) == 1 and len(noncrossover_paths) == 1:
            preferred = (
                crossover_paths[0]
                if rng.random() < crossover_rate
                else noncrossover_paths[0]
            )
            order = [preferred]
        else:
            preferred = int(rng.integers(0, len(pathways)))
            order = [preferred, *[i for i in range(len(pathways)) if i != preferred]]
        completed = False
        for path_index in order:
            remaining = remaining_evaluations - evaluations
            children, state, used, event_observed = _execute_stream_event(
                pathways[path_index],
                params[path_index],
                solutions,
                aux_cache[path_index],
                problem,
                data,
                setting,
                generation,
                remaining,
                mutation_rate,
                target - len(offspring),
            )
            aux_cache[path_index] = state
            evaluations += used
            observed.extend(event_observed)
            if children:
                offspring.extend(
                    (path_index, child) for child in children[: target - len(offspring)]
                )
                completed = True
            else:
                interrupted = True
            break
        if interrupted or not completed:
            break
    if not offspring or interrupted:
        iteration_best = (
            [min([*observed, *list(solutions)], key=lambda solution: solution.fit)]
            if observed
            else []
        )
        return solutions, aux_cache, evaluations, archives, iteration_best, attempts
    # eoFastGA evaluates final offspring as one generation-end batch.  Crossed
    # pairs were already evaluated immediately because after-cross selection
    # needs their fitness.  Keeping these phases separate also preserves the
    # RNG order of resampling boundary repair for BBOB trials.
    completed_offspring: list[tuple[int, Solution]] = []
    for path_index, child in offspring:
        if isinstance(child, Solution):
            completed_offspring.append((path_index, child))
            continue
        produced, used, newly_evaluated = _stream_evaluate_changed(
            child.decisions,
            child.sources,
            problem,
            data,
            remaining_evaluations - evaluations,
            child.invalidated,
        )
        evaluations += used
        observed.extend(newly_evaluated)
        if produced is None:
            interrupted = True
            break
        completed_offspring.extend((path_index, item) for item in produced)
    if interrupted:
        iteration_best = (
            [min([*observed, *list(solutions)], key=lambda solution: solution.fit)]
            if observed
            else []
        )
        return solutions, aux_cache, evaluations, archives, iteration_best, attempts
    # CMA samples are produced event-by-event so pathway routing remains
    # uniform, but its distribution update is intrinsically generational.
    # Accumulate each CMA stage's own evaluated samples and update that stage
    # once.  A later Search may change stream cardinality or decisions, so the
    # pathway's final offspring are not a valid fitness proxy for CMA samples.
    for path_index, path in enumerate(pathways):
        cma_stage_indices = [
            index
            for index, stage in enumerate(path.stages)
            if stage.search.primary == "search_cma"
        ]
        if not cma_stage_indices:
            continue
        path_state = aux_cache[path_index]
        stage_states = path_state.get("_stream_stage_aux", [])
        for stage_index in cma_stage_indices:
            if stage_index >= len(stage_states):
                continue
            stage_state = stage_states[stage_index]
            disturbances = stage_state.pop("_stream_cma_disturbances", [])
            sampled_solutions = stage_state.pop("_stream_cma_solutions", [])
            if not disturbances and not sampled_solutions:
                continue
            if len(disturbances) != len(sampled_solutions):
                raise RuntimeError(
                    "Stream CMA sample bookkeeping does not match its disturbances."
                )
            stage_state["cma_Disturb"] = np.asarray(disturbances, dtype=float)
            stage_states[stage_index] = get_component("para_cma")(
                SolutionSet(sampled_solutions), problem, stage_state, "solution"
            )
        path_state["_stream_stage_aux"] = stage_states
    # PSO also owns population-sized state.  Event routing advances only the
    # selected particle velocity; personal/global bests are updated once all
    # routed particles have been evaluated for this generation.
    for path_index, path in enumerate(pathways):
        pso_stage_indices = [
            index
            for index, stage in enumerate(path.stages)
            if stage.search.primary == "search_pso"
        ]
        if not pso_stage_indices:
            continue
        path_children = [
            child for owner, child in completed_offspring if owner == path_index
        ]
        if not path_children:
            continue
        path_state = aux_cache[path_index]
        stage_states = path_state.get("_stream_stage_aux", [])
        for stage_index in pso_stage_indices:
            if stage_index >= len(stage_states):
                continue
            stage_state = stage_states[stage_index]
            targets = stage_state.pop("_stream_pso_target_indices", [])
            pbest = list(stage_state.get("Pbest", []))
            if len(targets) != len(path_children) or not pbest:
                raise RuntimeError(
                    "Stream PSO target bookkeeping does not match its offspring."
                )
            for target_index, child in zip(targets, path_children):
                index = int(target_index)
                if child.fit < pbest[index].fit:
                    pbest[index] = child
            stage_state["Pbest"] = SolutionSet(pbest)
            stage_state["Gbest"] = min(pbest, key=lambda item: item.fit)
            stage_states[stage_index] = stage_state
        path_state["_stream_stage_aux"] = stage_states
    # Pathways that terminate at the same update component and parameter share
    # one sink: merge all of their offspring and call replacement once.  Truly
    # different terminal updates remain path-local and are applied in stable
    # pathway order to their own offspring groups.
    update_groups: list[dict[str, Any]] = []
    for path_index, child in completed_offspring:
        update_name = pathways[path_index].update
        update_param = getattr(params[path_index], "update", None)
        active_search = str(pathways[path_index].stages[-1].search.primary)
        group = next(
            (
                item
                for item in update_groups
                if item["name"] == update_name
                and _same_optional_array(item["parameter"], update_param)
                and (
                    update_name != "update_iterated_local_search"
                    or item["active_search"] == active_search
                )
            ),
            None,
        )
        if group is None:
            group = {
                "path_index": path_index,
                "name": update_name,
                "parameter": update_param,
                "active_search": active_search,
                "offspring": [],
            }
            update_groups.append(group)
        group["offspring"].append(child)

    current = solutions
    for group in update_groups:
        path_index = int(group["path_index"])
        update_fn = get_component(str(group["name"]))
        update_state = (
            aux_cache[path_index]
            if isinstance(aux_cache[path_index], dict)
            else {"rng": rng}
        )
        update_state["_active_search_operator"] = str(group["active_search"])
        updated, update_aux = update_fn(
            list(current) + list(group["offspring"]),
            problem,
            group["parameter"],
            update_state,
            generation,
            1,
            data,
            "execute",
        )
        update_state.pop("_active_search_operator", None)
        if isinstance(update_aux, dict):
            update_aux.pop("_active_search_operator", None)
            aux_cache[path_index] = update_aux
            aux_cache[path_index].setdefault("rng", rng)
        if len(updated) != len(current):
            updated = sorted(list(updated), key=lambda solution: solution.fit)[
                : len(current)
            ]
        current = SolutionSet(updated)
    archives = _update_archives(current, archive_names, archives, problem)
    best = min([*observed, *list(current)], key=lambda solution: solution.fit)
    return current, aux_cache, evaluations, archives, [best], attempts


def _same_optional_array(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _validate_shared_path_endpoints(
    pathways: Sequence[Any], params: Sequence[Any]
) -> None:
    """Reject multi-path phenotypes which the shared-endpoint genotype cannot express."""

    if len(pathways) <= 1:
        return
    if len(pathways) != len(params):
        raise ValueError("Pathway operator and parameter counts do not match.")
    first_path = pathways[0]
    first_params = params[0]
    for path, path_params in zip(pathways[1:], params[1:]):
        if path.choose != first_path.choose or path.update != first_path.update:
            raise ValueError(
                "AutoOptLib multi-path algorithms must share choose and update nodes."
            )
        if not _same_optional_array(path_params.choose, first_params.choose) or not (
            _same_optional_array(path_params.update, first_params.update)
        ):
            raise ValueError(
                "AutoOptLib multi-path algorithms must share choose/update parameters."
            )


def run_design(
    pathways: List[Any],
    params: List[Any],
    problem: Any,
    data: Any,
    setting: Any,
    *,
    algorithm_configuration: Any = None,
) -> dict:
    stream_graph = bool(pathways) and isinstance(pathways[0], StreamPathway)
    if stream_graph:
        if not all(isinstance(path, StreamPathway) for path in pathways):
            raise TypeError("Legacy and Search v6.0 pathways cannot be mixed.")
        if not all(isinstance(item, StreamPathwayParam) for item in params):
            raise TypeError("Search v6.0 pathways require stream pathway parameters.")
        validate_stream_phenotype(pathways, params)
        for pathway in pathways:
            if component_category(pathway.update) != "update":
                raise ValueError(
                    f"Search v6.0 update component {pathway.update!r} is invalid."
                )
            if any(component_category(name) != "archive" for name in pathway.archive):
                raise ValueError("Search v6.0 contains an invalid archive component.")
            for stage in pathway.stages:
                if (
                    stage.choose is not None
                    and component_category(stage.choose) != "choose"
                ):
                    raise ValueError(
                        f"Search v6.0 choose component {stage.choose!r} is invalid."
                    )
                if component_category(stage.search.primary) != "search":
                    raise ValueError(
                        "Search v6.0 contains an invalid search component."
                    )
    else:
        _validate_shared_path_endpoints(pathways, params)
    if algorithm_configuration is not None:
        configuration = normalize_configuration(algorithm_configuration, setting)
        setting = copy(setting)
        setting._AlgorithmPopulationSize = configuration["population_size"]
        setting._AlgorithmOffspringSize = configuration["offspring_size"]
        setting._AlgorithmCrossoverRate = configuration["crossover_rate"]
        setting._AlgorithmMutationRate = configuration["mutation_rate"]
        setting._AlgorithmBoundaryHandling = configuration["boundary_handling"]
    population_override = get_flex(setting, "_AlgorithmPopulationSize", None)
    if population_override is not None:
        problem = copy(problem)
        try:
            problem.N = int(population_override)
            offspring_override = int(
                get_flex(setting, "_AlgorithmOffspringSize", population_override)
            )
            problem.Gmax = max(
                int(get_flex(problem, "Gmax", 1)),
                int(
                    np.ceil(
                        get_flex(setting, "ProbFE", population_override)
                        / max(1, offspring_override)
                    )
                )
                + 1,
            )
        except (AttributeError, TypeError):
            pass
    rng = ensure_rng(setting)
    _configure_evaluation_runtime(problem, setting)
    try:
        problem._AlgorithmBoundaryHandling = str(
            get_flex(setting, "_AlgorithmBoundaryHandling", "clip")
        )
    except (AttributeError, TypeError):
        pass
    try:
        problem.rng = rng
    except (AttributeError, TypeError):
        pass
    population_size = int(
        get_flex(
            setting,
            "_AlgorithmPopulationSize",
            get_flex(problem, "N", get_flex(setting, "ProbN", 10)),
        )
    )
    offspring_size = int(get_flex(setting, "_AlgorithmOffspringSize", population_size))
    if (
        any(
            str(getattr(step, "primary", "")) == "search_pso"
            for pathway in pathways
            for step in getattr(pathway, "search", ())
        )
        and offspring_size != population_size
    ):
        raise ValueError(
            "search_pso requires offspring_size == population_size so velocity "
            "and personal-best state remain aligned with particles."
        )
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
    stream_event_budget_multiplier = get_flex(
        setting, "StreamEventBudgetMultiplier", None
    )
    stream_event_budget: int | None = None
    if stream_event_budget_multiplier is not None:
        multiplier = float(stream_event_budget_multiplier)
        if not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError(
                "StreamEventBudgetMultiplier must be a positive finite number or None."
            )
        if stream_graph:
            stream_event_budget = int(np.ceil(evaluation_budget * multiplier))

    checkpoint_value = get_flex(setting, "_checkpoint_path", None)
    checkpoint_path = Path(checkpoint_value) if checkpoint_value else None
    resume = bool(get_flex(setting, "Resume", False))
    checkpoint_every = int(get_flex(setting, "CheckpointEvery", 1))

    embedded_archives = getattr(pathways[0], "archive", []) if pathways else []
    archive_names = list(embedded_archives or get_flex(setting, "archive", []))
    checkpoint_signature = _solve_checkpoint_signature(
        pathways,
        params,
        problem,
        data,
        setting,
        archive_names,
    )
    resume_state = None
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        resume_state = _load_checkpoint(
            checkpoint_path,
            population_size=population_size,
            evaluation_budget=evaluation_budget,
            checkpoint_signature=checkpoint_signature,
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
    stream_events = 0
    termination_reason = "completed"

    if resume_state is not None:
        history = resume_state["history"]
        fit_history = resume_state["fit_history"]
        evaluations = int(resume_state["evaluations"])
        elapsed_seconds = float(resume_state["elapsed_seconds"])
        generation = int(resume_state["generation"])
        stream_events = int(resume_state.get("stream_events", 0))
        termination_reason = (
            str(resume_state.get("termination_reason", "completed"))
            if resume_state.get("complete", False)
            else "completed"
        )
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
                "stream_events": stream_events,
                "stream_event_budget": stream_event_budget,
                "termination_reason": termination_reason,
            }

    if not history:
        if len(solutions):
            initial_best = min(list(solutions), key=lambda s: s.fit)
            history.append(initial_best)
            fit_history.append(float(initial_best.fit))
        else:
            history.append(None)
            fit_history.append(np.inf)

    stream_zero_evaluation_generations = 0
    if stream_graph and aux_cache and isinstance(aux_cache[0], dict):
        stream_zero_evaluation_generations = int(
            aux_cache[0].get("_stream_zero_evaluation_generations", 0)
        )

    while True:
        # eoFastGA is FE-terminated rather than generation-terminated.  Its
        # mutation laws may legally return valid clones for an entire
        # generation, so applying the legacy Gmax estimate here can stop after
        # the first zero-FE generation even though later generations progress.
        if not stream_graph and generation > gmax:
            break
        if evaluations >= evaluation_budget:
            break
        if (
            stream_graph
            and stream_event_budget is not None
            and stream_events >= stream_event_budget
        ):
            termination_reason = "stream_event_budget_exhausted"
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
        if stream_graph:
            remaining = evaluation_budget - evaluations
            if metric == "runtimeFE":
                remaining = min(remaining, int(Tmax) - evaluations)
            (
                solutions,
                aux_cache,
                evals,
                archives,
                iteration_bests,
                events_used,
            ) = _execute_stream_generation(
                pathways,
                params,
                solutions,
                aux_cache,
                problem,
                data,
                setting,
                generation,
                remaining,
                archive_names,
                archives,
                (
                    None
                    if stream_event_budget is None
                    else stream_event_budget - stream_events
                ),
            )
            evaluations += evals
            stream_events += events_used
            if (
                stream_event_budget is not None
                and stream_events >= stream_event_budget
                and evaluations < evaluation_budget
            ):
                termination_reason = "stream_event_budget_exhausted"
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
                if not iteration_bests:
                    break
                stream_zero_evaluation_generations += 1
                if aux_cache and isinstance(aux_cache[0], dict):
                    aux_cache[0]["_stream_zero_evaluation_generations"] = (
                        stream_zero_evaluation_generations
                    )
                if (
                    stream_zero_evaluation_generations
                    >= STREAM_MAX_ZERO_EVALUATION_GENERATIONS
                ):
                    break
            else:
                stream_zero_evaluation_generations = 0
                if aux_cache and isinstance(aux_cache[0], dict):
                    aux_cache[0]["_stream_zero_evaluation_generations"] = 0
            if (
                stream_event_budget is not None
                and stream_events >= stream_event_budget
                and evaluations < evaluation_budget
            ):
                termination_reason = "stream_event_budget_exhausted"
                break
        elif len(pathways) == 1:
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
            target_total = offspring_size
            target_groups = [
                int(len(group))
                for group in np.array_split(
                    np.arange(target_total, dtype=int), len(pathways)
                )
            ]
            required_groups = [
                count * component_parent_arity(path.search[0].primary)
                if count and path.search
                else 0
                for count, path in zip(target_groups, pathways)
            ]
            required_total = int(sum(required_groups))
            choose_fn = get_component(pathways[0].choose)
            choose_param = getattr(params[0], "choose", None)
            choose_state = aux_cache[0] if isinstance(aux_cache[0], dict) else {}
            choose_state["_selection_count"] = required_total
            selected, choose_aux = choose_fn(
                solutions,
                problem,
                choose_param,
                choose_state,
                generation,
                1,
                data,
                "execute",
            )
            choose_state.pop("_selection_count", None)
            if isinstance(choose_aux, dict):
                aux_cache[0] = choose_aux
                aux_cache[0].setdefault("rng", rng)
            selected = np.asarray(selected, dtype=int).reshape(-1)
            selected = _resize_parent_indices(
                selected,
                population_size=len(solutions),
                offspring_size=required_total,
                rng=rng,
            )
            index_groups: list[list[int]] = []
            offset = 0
            for count in required_groups:
                index_groups.append(selected[offset : offset + count].tolist())
                offset += count

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
                if not subset_idx:
                    continue
                subset = SolutionSet([solutions[i] for i in subset_idx])
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
                    output_count=target_groups[p_idx],
                )
                aux_cache[p_idx] = aux_state
                new_items.extend(produced)
                evaluations += evals

            if not new_items:
                break

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
            if len(updated) != len(solutions):
                updated = sorted(list(updated), key=lambda solution: solution.fit)[
                    : len(solutions)
                ]
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
                    checkpoint_signature=checkpoint_signature,
                    complete=False,
                    stream_events=stream_events,
                    termination_reason="running",
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
                checkpoint_signature=checkpoint_signature,
                complete=True,
                stream_events=stream_events,
                termination_reason=termination_reason,
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
        "stream_events": stream_events,
        "stream_event_budget": stream_event_budget,
        "termination_reason": termination_reason,
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
        phenotype_groups = getattr(alg, "operator_pheno", None) or []
        setting.AlgP = len(phenotype_groups[0]) if phenotype_groups else 0
        return alg, setting

    alg_name = getattr(setting, "AlgName", getattr(setting, "alg_name", "")).strip()
    if not alg_name:
        raise ValueError("Please specify AlgFile or AlgName in solve mode.")
    from comparisons.manual.presets import _build_default_algorithm

    design = _build_default_algorithm(alg_name.lower(), setting)
    setting.AlgP = len(design.operator_pheno[0]) if design.operator_pheno else 0
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


def _auc_score(
    fit_history: Sequence[float],
    tmax: Any,
    thresholds: Any,
    offspring_size: int,
    initial_population_size: int | None = None,
) -> float:
    history = np.asarray(fit_history, dtype=float).reshape(-1)
    times = np.asarray(tmax if tmax is not None else [], dtype=float).reshape(-1)
    targets = np.asarray(
        thresholds if thresholds is not None else [], dtype=float
    ).reshape(-1)
    if history.size == 0 or times.size == 0 or targets.size != times.size:
        return float("inf")
    offspring = max(1, int(offspring_size))
    initial = (
        offspring
        if initial_population_size is None
        else max(1, int(initial_population_size))
    )
    indices = np.ceil(np.maximum(0.0, times - initial) / offspring).astype(int)
    indices = np.clip(indices, 0, history.size - 1)
    success_fraction = float(np.mean(history[indices] <= targets))
    return float(1.0 / (success_fraction + np.finfo(float).eps))


def _solve_result_score(
    result: dict[str, Any],
    metric: str,
    setting: Any,
    offspring_size: int,
    population_size: int,
    fallback: float,
) -> float:
    normalized = str(metric).lower()
    if normalized == "runtimefe":
        return float(result["evaluations"])
    if normalized == "runtimesec":
        return float(result["elapsed"])
    if normalized == "auc":
        return _auc_score(
            result["fit_history"],
            get_flex(setting, "Tmax", None),
            get_flex(setting, "Thres", None),
            offspring_size,
            population_size,
        )
    history = result.get("fit_history", ())
    return float(history[-1] if history else fallback)


def _advance_sequential_problem(
    current_problem: Any, current_data: Any, best: Solution
) -> tuple[Any, Any]:
    def normalize(result: Any, source: str) -> tuple[Any, Any]:
        if not isinstance(result, (tuple, list)) or len(result) not in {2, 3}:
            raise TypeError(
                f"{source} must return (next_problem, next_data) or "
                "(next_problem, next_data, auxiliary)."
            )
        return result[0], result[1]

    next_fn = getattr(current_problem, "advance_sequence", None)
    if callable(next_fn):
        return normalize(
            next_fn(best, current_data),
            "problem.advance_sequence",
        )
    name = getattr(current_problem, "name", None)
    if callable(name):
        return normalize(
            name(current_problem, current_data, best, "sequence"),
            "problem.name sequence callback",
        )
    if isinstance(name, str):
        separator = ":" if ":" in name else "."
        if separator in name:
            module_name, function_name = name.rsplit(separator, 1)
            try:
                module = __import__(module_name, fromlist=[function_name])
                callback = getattr(module, function_name)
            except (ImportError, AttributeError) as exc:
                raise RuntimeError(
                    f"Cannot resolve sequential update callable {name!r}."
                ) from exc
            if not callable(callback):
                raise TypeError(f"Sequential update target {name!r} is not callable.")
            return normalize(
                callback(current_problem, current_data, best, "sequence"),
                f"sequential update {name!r}",
            )
    raise RuntimeError(
        "Sequential problems must provide an advance_sequence callable or a "
        "callable problem.name sequence callback."
    )


@dataclass
class _SolveSession:
    pathways: Any
    params: Any
    problems: Sequence[Any]
    data: Sequence[Any]
    setting: Any
    custom_components: Any
    worker_initializer: Any = None
    worker_finalizer: Any = None
    worker_config: Any = None
    worker_environment: Any = None


@dataclass
class _SolveTask:
    instance: int
    run: int
    random_seed: int


def _run_solve_task(
    session: _SolveSession, task: _SolveTask
) -> tuple[int, int, str, list[Solution], Solution, float]:
    _restore_custom_components(session.custom_components)
    idx, run = task.instance, task.run
    problem = copy(session.problems[idx])
    data_obj = session.data[idx] if idx < len(session.data) else None
    mode = _extract_mode(problem)
    setting = copy(session.setting)
    setting.rng = np.random.default_rng(task.random_seed)
    setting.EvalWorkers = 1
    setting.EvalBackend = "serial"
    population_size = int(
        get_flex(setting, "_AlgorithmPopulationSize", get_flex(setting, "ProbN", 1))
    )
    offspring_size = int(get_flex(setting, "_AlgorithmOffspringSize", population_size))
    try:
        problem.N = population_size
        problem.Gmax = max(
            int(get_flex(problem, "Gmax", 1)),
            int(np.ceil(get_flex(setting, "ProbFE", population_size) / offspring_size))
            + 1,
        )
    except (AttributeError, TypeError):
        pass
    checkpoint_dir = get_flex(setting, "CheckpointDir", None)
    if checkpoint_dir is not None:
        if mode == "static":
            setting._checkpoint_path = str(
                Path(checkpoint_dir) / f"instance_{idx + 1}_run_{run + 1}.pkl"
            )
        else:
            setting._sequence_checkpoint_path = str(
                Path(checkpoint_dir) / f"instance_{idx + 1}_run_{run + 1}_sequence.pkl"
            )

    if mode == "static":
        result = run_design(
            session.pathways, session.params, problem, data_obj, setting
        )
        history = [sol for sol in result["history"] if isinstance(sol, Solution)]
        best = result["best_solution"]
        if best is None and history:
            best = history[-1]
        resolved = _ensure_solution_instance(best)
        score = _solve_result_score(
            result,
            get_flex(setting, "metric", "quality"),
            setting,
            offspring_size,
            population_size,
            float(resolved.fit),
        )
        return idx, run, mode, history, resolved, score

    if mode != "sequential":
        raise NotImplementedError(f"Unsupported problem mode: {mode}")
    current_problem = problem
    current_data = data_obj
    sequence_solutions: List[Solution] = []
    cumulative = 0.0
    total_budget = int(get_flex(setting, "ProbFE", 0))
    metric = str(get_flex(setting, "metric", "quality")).lower()
    population_size = int(get_flex(current_problem, "N", get_flex(setting, "ProbN", 1)))
    sequence_signature = _solve_checkpoint_signature(
        session.pathways,
        session.params,
        current_problem,
        current_data,
        setting,
        [],
    )
    sequence_path_value = get_flex(setting, "_sequence_checkpoint_path", None)
    sequence_path = Path(sequence_path_value) if sequence_path_value else None
    sequence_state = None
    sequence_rng = ensure_rng(setting)
    if (
        bool(get_flex(setting, "Resume", False))
        and sequence_path is not None
        and sequence_path.exists()
    ):
        sequence_state = _load_sequence_checkpoint(
            sequence_path,
            population_size=population_size,
            evaluation_budget=total_budget,
            checkpoint_signature=sequence_signature,
        )
        sequence_solutions = list(sequence_state["solutions"])
        cumulative = float(sequence_state["cumulative"])
        sequence_rng.bit_generator.state = sequence_state["rng_state"]
    if sequence_state is not None and sequence_state.get("complete", False):
        best = sequence_solutions[-1] if sequence_solutions else _placeholder_solution()
        return idx, run, mode, sequence_solutions, best, cumulative
    for completed_solution in sequence_solutions:
        advanced = _advance_sequential_problem(
            current_problem, current_data, completed_solution
        )
        current_problem, current_data = advanced
    while getattr(current_data, "continue", False):
        if checkpoint_dir is not None:
            stage_number = len(sequence_solutions) + 1
            setting._checkpoint_path = str(
                Path(checkpoint_dir)
                / f"instance_{idx + 1}_run_{run + 1}_stage_{stage_number}.pkl"
            )
        stage_budget = get_flex(
            getattr(current_data, "task", None),
            "budget",
            total_budget,
        )
        stage_setting = _setting_with_budget(setting, int(stage_budget))
        result = run_design(
            session.pathways,
            session.params,
            current_problem,
            current_data,
            stage_setting,
        )
        best = result["best_solution"]
        if best is None and result["history"]:
            history = [sol for sol in result["history"] if isinstance(sol, Solution)]
            best = history[-1] if history else None
        if best is None:
            break
        sequence_solutions.append(best)
        cumulative += _solve_result_score(
            result,
            metric,
            setting,
            offspring_size,
            population_size,
            float(best.fit),
        )
        if sequence_path is not None:
            _write_checkpoint(
                sequence_path,
                _sequence_checkpoint_payload(
                    solutions=sequence_solutions,
                    cumulative=cumulative,
                    population_size=population_size,
                    evaluation_budget=total_budget,
                    checkpoint_signature=sequence_signature,
                    rng=sequence_rng,
                    complete=False,
                ),
            )
        advanced = _advance_sequential_problem(current_problem, current_data, best)
        current_problem, current_data = advanced
    if sequence_path is not None:
        _write_checkpoint(
            sequence_path,
            _sequence_checkpoint_payload(
                solutions=sequence_solutions,
                cumulative=cumulative,
                population_size=population_size,
                evaluation_budget=total_budget,
                checkpoint_signature=sequence_signature,
                rng=sequence_rng,
                complete=True,
            ),
        )
    best = sequence_solutions[-1] if sequence_solutions else _placeholder_solution()
    return idx, run, mode, sequence_solutions, best, cumulative


def run_algorithm(
    alg: "Design", problems: Sequence[Any], data: Sequence[Any], app: Any, setting: Any
):
    if not alg.operator_pheno or not alg.parameter_pheno:
        raise ValueError("The algorithm must be decoded before solve execution.")
    caller_setting = setting
    setting = copy(setting)
    configuration = copy_configuration(alg, setting)
    setting._AlgorithmPopulationSize = configuration["population_size"]
    setting._AlgorithmOffspringSize = configuration["offspring_size"]
    setting._AlgorithmCrossoverRate = configuration["crossover_rate"]
    setting._AlgorithmMutationRate = configuration["mutation_rate"]
    for problem in problems:
        _configure_evaluation_runtime(problem, setting)
    pathways = alg.operator_pheno[0]
    params = alg.parameter_pheno[0]
    alg_runs = int(get_flex(setting, "AlgRuns", 1))
    rng = ensure_rng(setting)
    problem_signatures = [
        DesignEvaluationRuntime._build_problem_signature([problem])
        for problem in problems
    ]
    tasks = [
        ScheduledTask(
            _SolveTask(
                instance=idx,
                run=run,
                random_seed=int(
                    rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
                ),
            ),
            TaskResources(
                resident_key=f"problem:{idx}",
                profile_key=f"solve:{problem_signatures[idx]}:problem:{idx}",
            ),
        )
        for idx in range(len(problems))
        for run in range(alg_runs)
    ]
    worker_initializer = get_flex(setting, "EvalWorkerInitializer", None)
    worker_finalizer = get_flex(setting, "EvalWorkerFinalizer", None)
    worker_config = get_flex(setting, "EvalWorkerConfig", None)
    session = _SolveSession(
        pathways=pathways,
        params=params,
        problems=problems if worker_initializer is None else (),
        data=data if worker_initializer is None else (),
        setting=setting,
        custom_components=_custom_component_snapshot(),
        worker_initializer=worker_initializer,
        worker_finalizer=worker_finalizer,
        worker_config=worker_config,
    )
    config = EvaluationRuntimeConfig.from_setting(setting)
    backend = config.resolved_backend(len(tasks))
    workers = config.resolved_workers(len(tasks))
    runtime_stats = None
    wall_started = time.monotonic()
    if backend == "process" and workers > 1:
        with PersistentProcessExecutor(
            _run_solve_task,
            session,
            config=config,
            jobs_hint=len(tasks),
            initializer=(
                initialize_evaluation_session
                if worker_initializer is not None
                else None
            ),
            finalizer=(
                finalize_evaluation_session if worker_initializer is not None else None
            ),
        ) as executor:
            results = executor.map(
                tasks,
                timeout=config.task_timeout,
                # Per-evaluation retries are handled around the black-box call.
                retries=0,
            )
            runtime_stats = executor.statistics.as_dict()
            setting.EvalRuntimeProfiles = executor.resource_profiles
            setting.EvalConcurrencyProfiles = executor.concurrency_profiles
    else:
        results = [_run_solve_task(session, task.payload) for task in tasks]
    if runtime_stats is None:
        elapsed = time.monotonic() - wall_started
        topology = discover_cpu_topology()
        runtime_stats = RuntimeStatistics(
            requested_workers=config.workers,
            workers=workers,
            spawned_workers=0,
            eligible_workers=1,
            active_workers=1,
            selected_workers=1,
            submitted=len(tasks),
            completed=len(results),
            task_seconds=elapsed if workers == 1 else 0.0,
            wall_seconds=elapsed,
            capacity_seconds=elapsed * max(1, workers),
            waves=1,
            observed_cpu_seconds=elapsed if workers == 1 else 0.0,
            allocated_core_seconds=elapsed if workers == 1 else 0.0,
            peak_concurrency=1 if tasks else 0,
            logical_cpus=len(topology),
            physical_cores=len({item.physical_core for item in topology}),
            numa_nodes=len({item.node_id for item in topology}),
        ).as_dict()
    setting.EvalRuntimeStats = runtime_stats
    if not hasattr(setting, "EvalRuntimeProfiles"):
        setting.EvalRuntimeProfiles = {}
    if not hasattr(setting, "EvalConcurrencyProfiles"):
        setting.EvalConcurrencyProfiles = {}
    caller_setting.EvalRuntimeStats = setting.EvalRuntimeStats
    caller_setting.EvalRuntimeProfiles = setting.EvalRuntimeProfiles
    caller_setting.EvalConcurrencyProfiles = setting.EvalConcurrencyProfiles

    by_instance: dict[int, list[tuple[int, str, list[Solution], Solution, float]]] = {}
    for idx, run, mode, history, best, score in results:
        by_instance.setdefault(idx, []).append((run, mode, history, best, score))
    best_solutions: List[List[Solution]] = []
    all_solutions: List[List[Solution]] = []
    for idx in range(len(problems)):
        values = sorted(by_instance[idx], key=lambda value: value[0])
        run_best = [value[3] for value in values]
        run_histories = [value[2] for value in values]
        scores = [value[4] for value in values]
        best_idx = int(np.argmin(scores)) if scores else 0
        best_solutions.append(run_best)
        all_solutions.append(run_histories[best_idx])
    if app is not None and hasattr(app, "TextArea"):
        app.TextArea.Value = "Solving... 100.0%"
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
