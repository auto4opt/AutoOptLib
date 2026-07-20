from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from ..components import compatible_custom_components, get_component
from .design._helpers import get_problem_type


def _to_namespace(setting: Any) -> SimpleNamespace:
    if isinstance(setting, SimpleNamespace):
        return setting
    if isinstance(setting, dict):
        return SimpleNamespace(**setting)
    # Fallback: copy attributes
    data = {
        k: getattr(setting, k)
        for k in dir(setting)
        if not k.startswith("__") and not callable(getattr(setting, k))
    }
    return SimpleNamespace(**data)


def _to_behavior_matrix(behavior: Any) -> list[list[Any]] | None:
    """Convert component behavior metadata to a list-of-lists structure."""
    if behavior is None:
        return None
    if isinstance(behavior, np.ndarray):
        behavior = behavior.tolist()
    elif isinstance(behavior, tuple):
        behavior = list(behavior)
    elif not isinstance(behavior, list):
        return None

    matrix: list[list[Any]] = []
    for row in behavior:
        if isinstance(row, np.ndarray):
            matrix.append(row.tolist())
        elif isinstance(row, (list, tuple)):
            matrix.append(list(row))
        elif row is None:
            matrix.append([])
        else:
            matrix.append([row])
    return matrix


def _compute_local_bounds(
    behavior: list[list[Any]] | None, bounds: np.ndarray | None, ls_range: float
) -> np.ndarray | None:
    """Compute the parameter subspace used for local search, mirroring MATLAB Space.m."""
    if behavior is None or bounds is None or bounds.size == 0:
        return None

    if not behavior or not behavior[0]:
        return None
    first_col = behavior[0][0]
    if first_col in (None, "", 0):
        return None

    has_global = (
        len(behavior) > 1 and behavior[1] and behavior[1][0] not in (None, "", 0)
    )
    if not has_global:
        return bounds.copy()

    ls_range = float(ls_range)
    ls_range = max(0.0, min(1.0, ls_range))
    local_bounds = np.array(bounds, copy=True, dtype=float)
    span = bounds[:, 1] - bounds[:, 0]
    trends = behavior[0][1:] if len(behavior[0]) > 1 else []

    for idx in range(bounds.shape[0]):
        lower = bounds[idx, 0]
        upper = bounds[idx, 1]
        trend = trends[idx] if idx < len(trends) else None
        if trend == "small":
            local_bounds[idx, 0] = lower
            local_bounds[idx, 1] = lower + span[idx] * ls_range
        elif trend == "large":
            # Keep the reference search region used by the published MATLAB
            # experiments (Space.m), including its upper-bound scaling rule.
            local_bounds[idx, 0] = upper - upper * ls_range
            local_bounds[idx, 1] = upper
        else:
            local_bounds[idx, 0] = lower
            local_bounds[idx, 1] = upper

    return local_bounds


def space(problem: Any, setting: Any) -> SimpleNamespace:
    """Define operator/parameter space for the design problem (continuous only)."""
    set_obj = _to_namespace(setting)
    ls_range = float(getattr(set_obj, "LSRange", 0.25))
    set_obj.LSRange = ls_range
    ptype = get_problem_type(problem)

    if ptype == "continuous":
        choose = [
            "choose_traverse",
            "choose_tournament",
            "choose_roulette_wheel",
            "choose_nich",
        ]
        search = [
            "search_de_current",
            "search_de_current_best",
            "search_de_random",
            "cross_arithmetic",
            "cross_sim_binary",
            "cross_point_one",
            "cross_point_two",
            "cross_point_n",
            "cross_point_uniform",
            "search_mu_gaussian",
            "search_mu_cauchy",
            "search_mu_polynomial",
            "search_mu_uniform",
            "search_eda",
            "search_cma",
            "reinit_continuous",
        ]
        update = [
            "update_greedy",
            "update_round_robin",
            "update_pairwise",
            "update_always",
            "update_simulated_annealing",
        ]
    elif ptype == "discrete":
        choose = [
            "choose_traverse",
            "choose_tournament",
            "choose_roulette_wheel",
            "choose_nich",
        ]
        search = [
            "cross_point_one",
            "cross_point_two",
            "cross_point_uniform",
            "cross_point_n",
            "search_reset_one",
            "search_reset_n",
            "search_reset_rand",
            "reinit_discrete",
        ]
        update = [
            "update_greedy",
            "update_round_robin",
            "update_pairwise",
            "update_always",
            "update_simulated_annealing",
        ]
    elif ptype == "permutation":
        choose = [
            "choose_traverse",
            "choose_tournament",
            "choose_roulette_wheel",
            "choose_nich",
        ]
        search = [
            "cross_order_two",
            "cross_order_n",
            "search_swap",
            "search_swap_multi",
            "search_scramble",
            "search_insert",
            "reinit_permutation",
        ]
        update = [
            "update_greedy",
            "update_round_robin",
            "update_pairwise",
            "update_always",
            "update_simulated_annealing",
        ]
    else:
        raise NotImplementedError(
            "space() currently supports continuous, discrete, and permutation problems"
        )

    choose.extend(
        name
        for name in compatible_custom_components("choose", ptype)
        if name not in choose
    )
    search.extend(
        name
        for name in compatible_custom_components("search", ptype)
        if name not in search
    )
    update.extend(
        name
        for name in compatible_custom_components("update", ptype)
        if name not in update
    )

    all_op = choose + search + update
    op_space = np.array(
        [
            [1, len(choose)],
            [len(choose) + 1, len(choose) + len(search)],
            [len(choose) + len(search) + 1, len(all_op)],
        ],
        dtype=int,
    )

    para_space = []
    behav_space = []
    para_local_space = []

    for name in all_op:
        arr = None
        try:
            comp = get_component(name)
        except KeyError:
            para_space.append(None)
            behav_space.append(None)
            para_local_space.append(None)
            continue

        try:
            params, _ = comp(problem, "parameter")
        except TypeError:
            params = None
        try:
            behavior, _ = comp("behavior")
        except TypeError:
            behavior = None

        if params is None:
            arr = None
        else:
            arr = np.asarray(params, dtype=float)
            if arr.ndim == 1:
                if arr.size % 2 != 0:
                    raise ValueError(f"Parameter bounds for {name} must contain pairs")
                arr = arr.reshape(-1, 2)
            elif arr.ndim > 1 and arr.shape[1] != 2:
                total = arr.size
                if total % 2 != 0:
                    raise ValueError(f"Parameter bounds for {name} must contain pairs")
                arr = arr.reshape(-1, 2)

        behav_matrix = _to_behavior_matrix(behavior)
        para_space.append(arr)
        behav_space.append(behav_matrix if behav_matrix is not None else behavior)
        para_local_space.append(_compute_local_bounds(behav_matrix, arr, ls_range))

    set_obj.AllOp = all_op
    set_obj.OpSpace = op_space
    set_obj.ParaSpace = para_space
    set_obj.BehavSpace = behav_space
    set_obj.ParaLocalSpace = para_local_space

    # Provide defaults for downstream code
    set_obj.TunePara = getattr(
        set_obj, "TunePara", getattr(set_obj, "tune_para", False)
    )
    set_obj.alg_p = getattr(set_obj, "alg_p", getattr(set_obj, "AlgP", 1))
    set_obj.alg_q = getattr(set_obj, "alg_q", getattr(set_obj, "AlgQ", 3))
    set_obj.alg_n = getattr(set_obj, "alg_n", getattr(set_obj, "AlgN", 1))
    return set_obj
