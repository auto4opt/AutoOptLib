"""Python translation of Utilities/General/Process.m."""

from __future__ import annotations

import math
import pickle
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, List, Sequence, Tuple

import numpy as np

from ...components import get_component
from ...problems.base import validate_constructed_problems
from ..design import Approximate, Design
from ..general.improve_rate import improve_rate
from ..select import select as _select_alg
from ..solve import input_algorithm, run_algorithm
from ..space import space


def _progress(app: Any, message: str, *, done: bool = False) -> None:
    if app is not None:
        if hasattr(app, "TextArea"):
            try:
                app.TextArea.Value = message
            except Exception:  # pragma: no cover
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
            if not name.startswith("__") and not callable(getattr(setting, name))
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
            module_name, func_name = descriptor.split(":", 1)
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


def _ensure_algorithm_list(obj: Iterable[Design]) -> List[Design]:
    return list(obj)


def _select(
    algs: Sequence[Design], problem: Any, data: Any, setting: Any, seeds: Sequence[int]
) -> List[Design]:
    return _select_alg(algs, problem, data, setting, seeds)


def _mean_performance(algs: Sequence[Design]) -> np.ndarray:
    values = []
    for alg in algs:
        arr = alg.ave_perform_all()
        values.append(float(np.mean(arr)) if arr.size else float("inf"))
    return np.asarray(values, dtype=float)


def _update_cma_parameters(
    algs: Sequence[Design], problems: Sequence[Any], aux_list: List[Any]
) -> List[Any]:
    for idx, aux in enumerate(aux_list):
        if isinstance(aux, dict) and "cma_Disturb" in aux:
            get_component("para_cma")(algs[idx], problems, aux, "algorithm")
    return aux_list


def _write_design_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist design-mode search state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _load_design_checkpoint(
    path: Path,
    *,
    alg_n: int,
    alg_fe: int,
    eval_mode: str,
    instance_train: Sequence[Any],
    instance_test: Sequence[Any],
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
        or payload.get("schema") != "autooptlib.design-checkpoint"
    ):
        raise ValueError(f"Invalid AutoOptLib design checkpoint: {path}")
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported AutoOptLib design checkpoint schema version.")
    expected = {
        "alg_n": alg_n,
        "alg_fe": alg_fe,
        "eval_mode": eval_mode,
        "instance_train": list(instance_train),
        "instance_test": list(instance_test),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(
                f"Design checkpoint {name} does not match the current run."
            )
    return payload


def _design_checkpoint_payload(
    *,
    algs: Sequence[Design],
    alg_trace: Sequence[Design],
    surrogate: Any,
    evaluated_count: int,
    generation: int,
    seed_train: Sequence[int],
    seed_test: Sequence[int],
    rng: np.random.Generator,
    inner_state: dict[str, Any] | None,
    alg_n: int,
    alg_fe: int,
    eval_mode: str,
    instance_train: Sequence[Any],
    instance_test: Sequence[Any],
    complete: bool,
) -> dict[str, Any]:
    return {
        "schema": "autooptlib.design-checkpoint",
        "schema_version": 1,
        "alg_n": alg_n,
        "alg_fe": alg_fe,
        "eval_mode": eval_mode,
        "instance_train": list(instance_train),
        "instance_test": list(instance_test),
        "algs": list(algs),
        "alg_trace": list(alg_trace),
        "surrogate": surrogate,
        "evaluated_count": evaluated_count,
        "generation": generation,
        "seed_train": list(seed_train),
        "seed_test": list(seed_test),
        "rng_state": rng.bit_generator.state,
        "inner_state": inner_state,
        "complete": complete,
    }


def _process_design(
    problem_descriptor: Any,
    instance_train: Sequence[Any],
    instance_test: Sequence[Any],
    setting: Any,
    app: Any,
) -> Tuple[List[Design], List[Design]]:
    setting_ns = _normalize_setting(setting)
    rng = _configure_rng(setting_ns)

    instance_train = list(instance_train)
    instance_test = list(instance_test)
    instances = instance_train + instance_test

    problems = _build_problem_struct(problem_descriptor, instances, setting_ns)
    construct_fn = _resolve_problem_callable(problem_descriptor)
    problems, data, _ = construct_fn(problems, instances, "construct")
    validate_constructed_problems(problems, data)

    setting_ns = space(problems, setting_ns)
    alg_n = int(getattr(setting_ns, "AlgN", getattr(setting_ns, "alg_n", 1)))
    alg_fe = int(getattr(setting_ns, "AlgFE", getattr(setting_ns, "alg_fe", alg_n)))

    seed_train = rng.permutation(len(instance_train)).tolist()
    seed_test = (rng.permutation(len(instance_test)) + len(instance_train)).tolist()

    eval_mode = str(
        getattr(setting_ns, "Evaluate", getattr(setting_ns, "evaluate", "exact"))
    ).lower()

    checkpoint_dir = getattr(setting_ns, "CheckpointDir", None)
    checkpoint_path = (
        Path(checkpoint_dir) / "design.pkl" if checkpoint_dir is not None else None
    )
    checkpoint_every = int(getattr(setting_ns, "CheckpointEvery", 1))
    resume = bool(getattr(setting_ns, "Resume", False))
    resume_state = None
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        resume_state = _load_design_checkpoint(
            checkpoint_path,
            alg_n=alg_n,
            alg_fe=alg_fe,
            eval_mode=eval_mode,
            instance_train=instance_train,
            instance_test=instance_test,
        )

    _progress(app, "Initializing...")

    algs: List[Design] = []
    surrogate = None
    # AlgFE follows Process.m: it counts algorithms proposed after the
    # initial incumbent population, whose evaluation is initialization cost.
    evaluated_count = 0
    alg_trace: List[Design] = []
    G = 1
    pending_inner_state: dict[str, Any] | None = None

    if resume_state is not None:
        algs = list(resume_state["algs"])
        alg_trace = list(resume_state["alg_trace"])
        surrogate = resume_state["surrogate"]
        evaluated_count = int(resume_state["evaluated_count"])
        G = int(resume_state["generation"])
        seed_train = list(resume_state["seed_train"])
        seed_test = list(resume_state["seed_test"])
        rng.bit_generator.state = resume_state["rng_state"]
        pending_inner_state = resume_state.get("inner_state")
        if resume_state.get("complete", False):
            _progress(app, "Complete", done=True)
            return algs[:alg_n], alg_trace
    elif eval_mode in {"exact", "intensification"}:
        for _ in range(alg_n):
            alg = Design(problems, setting_ns)
            alg.evaluate(problems, data, setting_ns, seed_train)
            algs.append(alg)
    elif eval_mode == "racing":
        racing_k = int(
            getattr(setting_ns, "RacingK", getattr(setting_ns, "racingk", 1))
        )
        subset = seed_train[: racing_k or 1]
        for _ in range(alg_n):
            alg = Design(problems, setting_ns)
            alg.evaluate(problems, data, setting_ns, subset)
            algs.append(alg)
    elif eval_mode == "approximate":
        surrogate = Approximate(problems, data, setting_ns, seed_train)
        pool = surrogate.data
        indices = rng.choice(len(pool), size=min(alg_n, len(pool)), replace=False)
        algs = [pool[i] for i in indices]
        evaluated_count = 0
    else:
        raise NotImplementedError(
            f"Evaluation mode '{eval_mode}' is not implemented in Python translation yet."
        )

    if checkpoint_path is not None and resume_state is None:
        _write_design_checkpoint(
            checkpoint_path,
            _design_checkpoint_payload(
                algs=algs,
                alg_trace=alg_trace,
                surrogate=surrogate,
                evaluated_count=evaluated_count,
                generation=G,
                seed_train=seed_train,
                seed_test=seed_test,
                rng=rng,
                inner_state=None,
                alg_n=alg_n,
                alg_fe=alg_fe,
                eval_mode=eval_mode,
                instance_train=instance_train,
                instance_test=instance_test,
                complete=False,
            ),
        )

    while evaluated_count < alg_fe:
        _progress(app, f"Designing... {100 * evaluated_count / alg_fe:.1f}%")
        if pending_inner_state is not None:
            improve = pending_inner_state["improve"]
            inner_g = int(pending_inner_state["inner_g"])
            aux_list = list(pending_inner_state["aux_list"])
            inner_gmax = int(pending_inner_state["inner_gmax"])
            pending_inner_state = None
        else:
            improve = None
            inner_g = 1
            aux_list = [None] * len(algs)
            inner_gmax = int(math.ceil(alg_fe / 10)) if alg_n == 1 else 1

        while (
            (
                improve is None
                or improve[0]
                >= getattr(setting_ns, "IncRate", getattr(setting_ns, "inc_rate", 0.05))
            )
            and inner_g <= inner_gmax
            and evaluated_count < alg_fe
        ):
            proposal_count = min(len(algs), alg_fe - evaluated_count)
            new_algs: List[Design] = []
            for idx, alg in enumerate(algs[:proposal_count]):
                new_alg, aux_list[idx] = alg.get_new(
                    problems, setting_ns, inner_g, aux_list[idx]
                )
                new_algs.append(new_alg)
            proposed_count = len(new_algs)
            if proposed_count == 0:
                break

            if eval_mode == "exact":
                for new_alg in new_algs:
                    new_alg.evaluate(problems, data, setting_ns, seed_train)
                algs = _select(algs + new_algs, problems, data, setting_ns, seed_train)
            elif eval_mode == "approximate":
                for new_alg in new_algs:
                    new_alg.estimate(problems, setting_ns, seed_train, surrogate)
                algs = _select(algs + new_algs, problems, data, setting_ns, seed_train)
                if surrogate is not None and G in set(
                    int(x) for x in np.asarray(surrogate.exact_g).reshape(-1)
                ):
                    for new_alg in new_algs:
                        new_alg.evaluate(problems, data, setting_ns, seed_train)
                    surrogate.UpdateModel(new_algs, setting_ns)
            elif eval_mode == "racing":
                racing_k = int(
                    getattr(setting_ns, "RacingK", getattr(setting_ns, "racingk", 1))
                )
                subset = seed_train[: racing_k or 1]
                for new_alg in new_algs:
                    new_alg.evaluate(problems, data, setting_ns, subset)
                algs = _select(algs + new_algs, problems, data, setting_ns, subset)
                remaining = seed_train[racing_k:]
                while len(algs) > alg_n and remaining:
                    algs = _select(algs, problems, data, setting_ns, [remaining[0]])
                    remaining = remaining[1:]
                seed_train = rng.permutation(len(instance_train)).tolist()
                if len(algs) > alg_n:
                    indices = rng.choice(
                        len(algs), size=len(algs) - alg_n, replace=False
                    )
                    algs = [alg for idx, alg in enumerate(algs) if idx not in indices]
            elif eval_mode == "intensification":
                while new_algs and seed_train:
                    subset = [seed_train[0]]
                    for alg in new_algs:
                        alg.evaluate(problems, data, setting_ns, subset)
                    new_algs = _select(
                        algs + new_algs, problems, data, setting_ns, subset
                    )
                    seed_train = seed_train[1:]
                seed_train = rng.permutation(len(instance_train)).tolist()
                for alg in new_algs:
                    for seed in seed_train:
                        if not np.any(alg.performance[seed, :]):
                            alg.evaluate(problems, data, setting_ns, [seed])
                if new_algs:
                    replace_idx = rng.choice(
                        len(algs), size=min(len(new_algs), len(algs)), replace=False
                    )
                    for dest, src in zip(replace_idx, new_algs):
                        algs[dest] = src
            else:
                raise NotImplementedError(
                    f"Evaluation mode '{eval_mode}' is not implemented."
                )

            evaluated_count += proposed_count
            aux_list = _update_cma_parameters(algs, problems, aux_list)

            perf_values = _mean_performance(algs)
            wrapper = SimpleNamespace(avePerformAll=lambda: perf_values)
            improve = improve_rate(wrapper, improve, inner_g, "algorithm")
            inner_g += 1
            G += 1

            best_idx = int(np.argmin(perf_values)) if len(perf_values) else 0
            alg_trace.append(algs[best_idx])

            if checkpoint_path is not None and (
                evaluated_count % checkpoint_every == 0 or evaluated_count >= alg_fe
            ):
                _write_design_checkpoint(
                    checkpoint_path,
                    _design_checkpoint_payload(
                        algs=algs,
                        alg_trace=alg_trace,
                        surrogate=surrogate,
                        evaluated_count=evaluated_count,
                        generation=G,
                        seed_train=seed_train,
                        seed_test=seed_test,
                        rng=rng,
                        inner_state={
                            "improve": improve,
                            "inner_g": inner_g,
                            "aux_list": aux_list,
                            "inner_gmax": inner_gmax,
                        },
                        alg_n=alg_n,
                        alg_fe=alg_fe,
                        eval_mode=eval_mode,
                        instance_train=instance_train,
                        instance_test=instance_test,
                        complete=False,
                    ),
                )

    _progress(app, "Testing...")
    setting_ns.Evaluate = "exact"
    setting_ns.evaluate = "exact"
    for alg in algs:
        alg.evaluate(problems, data, setting_ns, seed_test)
    algs = _select(algs, problems, data, setting_ns, seed_test)
    if checkpoint_path is not None:
        _write_design_checkpoint(
            checkpoint_path,
            _design_checkpoint_payload(
                algs=algs,
                alg_trace=alg_trace,
                surrogate=surrogate,
                evaluated_count=evaluated_count,
                generation=G,
                seed_train=seed_train,
                seed_test=seed_test,
                rng=rng,
                inner_state=None,
                alg_n=alg_n,
                alg_fe=alg_fe,
                eval_mode=eval_mode,
                instance_train=instance_train,
                instance_test=instance_test,
                complete=True,
            ),
        )
    _progress(app, "Complete", done=True)
    return algs[:alg_n], alg_trace


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
        instance_train, instance_test = args[:2]
        return _process_design(
            problem_descriptor, instance_train, instance_test, setting_ns, app
        )

    if mode == "solve":
        if len(args) < 1:
            raise ValueError("Solve mode requires instance list argument")
        instance_solve = args[0]
        problems = _build_problem_struct(problem_descriptor, instance_solve, setting_ns)
        construct_fn = _resolve_problem_callable(problem_descriptor)
        problems, data, _ = construct_fn(problems, instance_solve, "construct")
        validate_constructed_problems(problems, data)
        alg, setting_ns = input_algorithm(setting_ns)
        _progress(app, "Solving...")
        best_solutions, all_solutions = run_algorithm(
            alg, problems, data, app, setting_ns
        )
        _progress(app, "Complete", done=True)
        return best_solutions, all_solutions

    raise ValueError("Mode must be 'design' or 'solve'.")
