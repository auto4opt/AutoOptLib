from __future__ import annotations

from copy import deepcopy
from typing import Any, List, Sequence, Tuple

import numpy as np

from ._helpers import ensure_rng, get_flex


def _as_int(value: Any) -> int:
    arr = np.asarray(value)
    if arr.size == 0:
        raise ValueError("Cannot convert empty array to int")
    return int(arr.reshape(-1)[0])


def _ensure_alg_list(algs: Any) -> List[Any]:
    if isinstance(algs, Sequence) and not isinstance(algs, (str, bytes)):
        return list(algs)
    return [algs]


def _copy_parameter_structure(parameter: Sequence[Any]) -> List[List[Any]]:
    copied: List[List[Any]] = []
    for entry in parameter:
        if entry is None:
            copied.append([None, None])
            continue
        values = entry[0] if len(entry) > 0 else None
        behavior = entry[1] if len(entry) > 1 else None
        values_copy = None if values is None else np.array(values, copy=True)
        copied.append([values_copy, behavior])
    return copied


def _gather_operator_positions(
    paths: List[np.ndarray],
) -> Tuple[List[Tuple[str, int, int]], List[int]]:
    positions: List[Tuple[str, int, int]] = []
    values: List[int] = []
    if not paths or paths[0].size == 0:
        return positions, values
    # Choose and update are shared by all pathways in the reference encoding;
    # only search operators are pathway-specific.
    positions.append(("choose", 0, 0))
    values.append(_as_int(paths[0][0, 0]))
    for path_idx, path in enumerate(paths):
        if path.size == 0:
            continue
        for row in range(1, path.shape[0]):
            positions.append(("search", path_idx, row))
            values.append(_as_int(path[row, 0]))
    positions.append(("update", 0, paths[0].shape[0] - 1))
    values.append(_as_int(paths[0][-1, -1]))
    return positions, values


def _mutable_operator_positions(
    paths: List[np.ndarray], op_space: np.ndarray
) -> List[Tuple[str, int, int]]:
    positions, _ = _gather_operator_positions(paths)
    mutable = []
    for position in positions:
        category = position[0]
        row = {"choose": 0, "search": 1, "update": 2}[category]
        if int(op_space[row, 1]) > int(op_space[row, 0]):
            mutable.append(position)
    return mutable


def _disturb_search_single(
    path: np.ndarray,
    row_idx: int,
    rng: np.random.Generator,
    op_space: np.ndarray,
    alg_q: int,
) -> tuple[np.ndarray, set[int]]:
    pool = np.arange(op_space[1, 0], op_space[1, 1] + 1)
    current = _as_int(path[row_idx, 0])
    pool = pool[pool != current]
    num_search = path.shape[0] - 1
    if pool.size == 0:
        ind_new = current
    else:
        ind_new = _as_int(rng.choice(pool))
    if num_search == 1 and num_search < alg_q:
        sample_pool = list(pool) + [np.inf]
    elif num_search > 1 and num_search < alg_q:
        sample_pool = list(pool) + [np.inf, -np.inf]
    elif num_search > 1 and num_search == alg_q:
        sample_pool = list(pool) + [-np.inf]
    else:
        sample_pool = list(pool)
    sample_choice = rng.choice(sample_pool) if sample_pool else ind_new
    activated: set[int] = set()
    if sample_choice == np.inf:
        insert_row = np.zeros((1, 2), dtype=int)
        insert_row[0, 0] = ind_new
        insert_row[0, 1] = path[row_idx, 1]
        path = np.insert(path, row_idx + 1, insert_row, axis=0)
        path[row_idx, 1] = ind_new
        activated.add(ind_new)
    elif sample_choice == -np.inf and path.shape[0] > 2:
        path[row_idx - 1, 1] = path[row_idx, 1]
        path = np.delete(path, row_idx, axis=0)
    else:
        replacement = int(sample_choice)
        path[row_idx, 0] = replacement
        path[row_idx - 1, 1] = replacement
        activated.add(replacement)
    return path, activated


def _disturb_search_multi(
    path: np.ndarray,
    row_idx: int,
    rng: np.random.Generator,
    op_space: np.ndarray,
) -> tuple[np.ndarray, set[int]]:
    """Apply one primitive component replacement on a multi-path graph."""
    pool = np.arange(op_space[1, 0], op_space[1, 1] + 1)
    current = _as_int(path[row_idx, 0])
    pool = pool[pool != current]
    if pool.size:
        replacement = _as_int(rng.choice(pool))
        path[row_idx, 0] = replacement
        path[row_idx - 1, 1] = replacement
        return path, {replacement}
    return path, set()


def _clear_parameter_search_state(aux: dict[str, Any]) -> dict[str, Any]:
    """Drop optimizer state after a graph or conditional space change."""
    return {
        key: value
        for key, value in aux.items()
        if not str(key).startswith(("cma_", "ils_"))
    }


def active_parameter_loci(
    alg: Any,
    setting: Any,
) -> list[tuple[str, int, int, float, float]]:
    """Return active mixed-type parameter loci and their legal bounds."""
    paths = [np.asarray(path, dtype=int) for path in getattr(alg, "operator", [])]
    parameter = getattr(alg, "parameter", [])
    para_space = list(get_flex(setting, "para_space", required=True))
    para_type_space = list(
        get_flex(
            setting,
            "para_type_space",
            default=[
                () if bounds is None else ("continuous",) * len(bounds)
                for bounds in para_space
            ],
        )
    )
    from ._stream_graph import active_operator_indices, is_stream_graph

    if is_stream_graph(setting):
        active_operators = active_operator_indices(paths)
    else:
        _, active_operators = _gather_operator_positions(paths)
    active = set(active_operators)
    loci: list[tuple[str, int, int, float, float]] = []
    for para_idx, (bounds_value, kinds) in enumerate(zip(para_space, para_type_space)):
        if para_idx + 1 not in active or bounds_value is None:
            continue
        values_value = parameter[para_idx][0]
        if values_value is None:
            continue
        bounds = np.asarray(bounds_value, dtype=float).reshape(-1, 2)
        values = np.asarray(values_value, dtype=float).reshape(-1)
        if bounds.shape[0] != values.size or len(kinds) != values.size:
            raise ValueError(
                f"Parameter {para_idx + 1} metadata does not match its values"
            )
        for value_idx, kind in enumerate(kinds):
            lower = float(bounds[value_idx, 0])
            upper = float(bounds[value_idx, 1])
            if kind == "integer":
                lower = float(int(np.ceil(lower)))
                upper = float(int(np.floor(upper)))
            if upper > lower:
                loci.append((str(kind), para_idx, value_idx, lower, upper))
    return loci


def active_parameter_counts(
    alg: Any,
    setting: Any,
) -> tuple[int, int]:
    """Return active discrete and continuous parameter dimensions."""
    loci = active_parameter_loci(alg, setting)
    discrete = sum(locus[0] == "integer" for locus in loci)
    continuous = sum(locus[0] == "continuous" for locus in loci)
    return discrete, continuous


def mutate_structure(
    alg: Any,
    setting: Any,
    aux: Any = None,
) -> tuple[list[np.ndarray], list[list[Any]], dict[str, Any]]:
    """Mutate only graph nodes; edge changes are induced by the path grammar."""
    from ._stream_graph import is_stream_graph
    from ._stream_graph import mutate_structure as mutate_stream_graph

    if is_stream_graph(setting):
        return mutate_stream_graph(alg, setting, aux)

    rng = ensure_rng(setting)
    op_space = np.asarray(get_flex(setting, "op_space", required=True), dtype=int)
    alg_p = int(get_flex(setting, "alg_p", required=True))
    alg_q = int(get_flex(setting, "alg_q", required=True))
    rate = float(get_flex(setting, "structure_mutation_rate", 0.30))
    paths = [np.asarray(path, dtype=int).copy() for path in alg.operator]
    parameters = _copy_parameter_structure(alg.parameter)
    state = dict(aux) if isinstance(aux, dict) else {}
    _, old_operator_values = _gather_operator_positions(paths)
    old_active = set(old_operator_values)

    positions, values = _gather_operator_positions(paths)
    mutable = _mutable_operator_positions(paths, op_space)
    requested = max(1, int(np.floor(rate * len(values) + 0.5)))
    edit_count = min(requested, len(mutable))
    if edit_count == 0:
        state["last_move_kind"] = "structure"
        state["last_structure_edits"] = 0
        return paths, parameters, _clear_parameter_search_state(state)

    chosen_indices = np.asarray(
        rng.choice(len(mutable), size=edit_count, replace=False), dtype=int
    ).reshape(-1)
    chosen = [mutable[int(index)] for index in chosen_indices]
    # Search-node edits are applied from the end of each path so insertion or
    # deletion cannot invalidate yet-to-be-edited row indices.
    chosen.sort(
        key=lambda item: (
            item[0] != "choose",
            item[0] == "update",
            item[1],
            -item[2] if item[0] == "search" else item[2],
        )
    )

    edits = 0
    changed_operator_indices: set[int] = set()
    for category, path_idx, row_idx in chosen:
        if category == "choose":
            pool = np.arange(op_space[0, 0], op_space[0, 1] + 1)
            current = _as_int(paths[0][0, 0])
            pool = pool[pool != current]
            if pool.size:
                replacement = _as_int(rng.choice(pool))
                for path in paths:
                    path[0, 0] = replacement
                edits += 1
                changed_operator_indices.add(replacement)
        elif category == "update":
            pool = np.arange(op_space[2, 0], op_space[2, 1] + 1)
            current = _as_int(paths[0][-1, -1])
            pool = pool[pool != current]
            if pool.size:
                replacement = _as_int(rng.choice(pool))
                for path in paths:
                    path[-1, -1] = replacement
                edits += 1
                changed_operator_indices.add(replacement)
        else:
            before = paths[path_idx].copy()
            if alg_p == 1:
                paths[path_idx], activated = _disturb_search_single(
                    paths[path_idx], row_idx, rng, op_space, alg_q
                )
            else:
                paths[path_idx], activated = _disturb_search_multi(
                    paths[path_idx], row_idx, rng, op_space
                )
            if before.shape != paths[path_idx].shape or not np.array_equal(
                before, paths[path_idx]
            ):
                edits += 1
                changed_operator_indices.update(activated)

    state = _clear_parameter_search_state(state)
    _, new_operator_values = _gather_operator_positions(paths)
    newly_active = set(new_operator_values) - old_active
    conditional_operators = changed_operator_indices & set(new_operator_values)
    parameter_bank = get_flex(setting, "_component_parameter_bank", None)
    if parameter_bank is not None:
        for operator_index in newly_active:
            if 0 < operator_index <= len(parameters) and operator_index <= len(
                parameter_bank
            ):
                parameters[operator_index - 1] = deepcopy(
                    parameter_bank[operator_index - 1]
                )
    state["last_move_kind"] = "structure"
    state["last_structure_edits"] = edits
    state["structure_mutation_rate"] = rate
    state["conditional_operator_indices"] = sorted(conditional_operators)
    return paths, parameters, state


def disturb(
    algs: Any, setting: Any, inner_g: int, aux: Any
) -> Tuple[List[List[np.ndarray]], List[List[List[Any]]], Any]:
    """Compatibility wrapper for graph-ILS proposals.

    Hyperparameters require evaluated CMA-ES batches and are therefore handled
    only by the Search controller.  This one-proposal compatibility surface
    mutates graph representations exclusively.
    """
    alg_list = _ensure_alg_list(algs)
    aux_list: list[Any]
    if aux is None:
        aux_list = [{} for _ in alg_list]
    elif isinstance(aux, Sequence) and not isinstance(aux, (str, bytes, dict)):
        aux_list = list(aux)
        while len(aux_list) < len(alg_list):
            aux_list.append({})
    else:
        aux_list = [aux] + [{} for _ in range(len(alg_list) - 1)]
    new_ops: List[List[np.ndarray]] = []
    new_paras: List[List[List[Any]]] = []
    for idx, alg in enumerate(alg_list):
        state = aux_list[idx] if isinstance(aux_list[idx], dict) else {}
        paths, parameters, state = mutate_structure(alg, setting, state)
        state.pop("forced_action", None)
        new_ops.append(paths)
        new_paras.append(parameters)
        aux_list[idx] = state
    return new_ops, new_paras, aux_list
