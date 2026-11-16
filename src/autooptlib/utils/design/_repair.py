"""Python port of the MATLAB Repair routine."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._helpers import (
    ensure_rng,
    get_flex,
    get_problem_type,
    has_global_behavior,
    has_local_behavior,
    set_behavior,
)


def _prefers_global_behavior(behavior: Any, values: Any, bounds: Any) -> bool:
    """Classify dual-behavior operators without restricting their search range.

    Component metadata describes which end of each legal parameter range
    corresponds to local or global behavior.  Classification uses normalized
    distance to those endpoints; unlike the removed ``LSRange`` mechanism it
    never narrows the legal parameter domain seen by the parameter optimizer.
    """
    if not has_global_behavior(behavior):
        return False
    if not has_local_behavior(behavior):
        return True
    if values is None or bounds is None:
        return False
    values = np.asarray(values, dtype=float).reshape(-1)
    bounds = np.asarray(bounds, dtype=float).reshape(-1, 2)
    if values.size != bounds.shape[0]:
        return False
    span = bounds[:, 1] - bounds[:, 0]
    normalized = np.divide(
        values - bounds[:, 0],
        span,
        out=np.full(values.shape, 0.5, dtype=float),
        where=span > 0,
    )

    def distance(row: Any) -> float:
        trends = list(row[1:]) if isinstance(row, (list, tuple)) else []
        terms = []
        for idx, value in enumerate(normalized):
            trend = trends[idx] if idx < len(trends) else None
            if trend == "small":
                terms.append(float(value) ** 2)
            elif trend == "large":
                terms.append(float(1.0 - value) ** 2)
        return float(np.mean(terms)) if terms else 0.0

    try:
        local_distance = distance(behavior[0])
        global_distance = distance(behavior[1])
    except (IndexError, TypeError):
        return False
    return global_distance < local_distance


def _repair_once(
    operators: list[list[np.ndarray]],
    paras: list[list[list[Any]]],
    problem: Any,
    setting: Any,
):
    """Ensure the designed algorithm(s) remain reasonable."""
    rng = ensure_rng(setting)
    all_op = list(get_flex(setting, "all_op", required=True))
    para_space = list(get_flex(setting, "para_space", default=[None] * len(all_op)))
    behav_space = list(get_flex(setting, "behav_space", default=[None] * len(all_op)))

    while len(para_space) < len(all_op):
        para_space.append(None)
    while len(behav_space) < len(all_op):
        behav_space.append(None)

    problem_type = get_problem_type(problem) or ""
    if problem_type == "continuous":
        ind_mu = [idx + 1 for idx, name in enumerate(all_op) if "search_mu" in name]
    elif problem_type in {"discrete", "permutation"}:
        ind_mu = [idx + 1 for idx, name in enumerate(all_op) if "search" in name]
    else:
        ind_mu = []
    ind_cross = [idx + 1 for idx, name in enumerate(all_op) if "cross" in name]

    repaired_ops: list[list[np.ndarray]] = []
    repaired_paras: list[list[list[Any]]] = []

    for algo_idx, algo_ops in enumerate(operators):
        curr_ops = [np.array(path, dtype=int, copy=True) for path in algo_ops]
        curr_paras = [
            list(entry) if entry is not None else [None, None]
            for entry in paras[algo_idx]
        ]

        for path_idx, matrix in enumerate(curr_ops):
            if matrix.size == 0:
                continue

            for row in range(1, matrix.shape[0] - 1):
                if (
                    ind_cross
                    and matrix[row, 0] in ind_cross
                    and matrix[row, 1] not in ind_mu
                    and ind_mu
                ):
                    ind_new = int(rng.choice(ind_mu))
                    matrix[row, 1] = ind_new
                    matrix[row + 1, 0] = ind_new

            if matrix[-1, 0] in ind_cross and ind_mu:
                ind_new = int(rng.choice(ind_mu))
                matrix[-1, 0] = ind_new
                matrix[-2, 1] = ind_new

            mask = matrix[:, 0] != matrix[:, 1]
            matrix = matrix[mask]

            ind_pso = next(
                (idx + 1 for idx, name in enumerate(all_op) if name == "search_pso"),
                None,
            )
            if ind_pso is not None and np.any(matrix == ind_pso):
                ind_choose = next(
                    (
                        idx + 1
                        for idx, name in enumerate(all_op)
                        if name == "choose_traverse"
                    ),
                    None,
                )
                ind_update = next(
                    (
                        idx + 1
                        for idx, name in enumerate(all_op)
                        if name == "update_always"
                    ),
                    None,
                )
                if ind_choose and ind_update:
                    # Choose and update are shared graph endpoints.  Updating
                    # only the first and PSO-containing paths leaves a
                    # multi-path phenotype internally contradictory and makes
                    # serialization disagree with execution.
                    for shared_idx, shared_path in enumerate(curr_ops):
                        shared_matrix = np.array(shared_path, dtype=int, copy=True)
                        if shared_matrix.size == 0:
                            shared_matrix = np.zeros((2, 2), dtype=int)
                        shared_matrix[0, 0] = ind_choose
                        shared_matrix[-1, 1] = ind_update
                        curr_ops[shared_idx] = shared_matrix
                    matrix = np.array(
                        [[ind_choose, ind_pso], [ind_pso, ind_update]], dtype=int
                    )

            ind_search = matrix[1:, 0].tolist()
            previously_global = {
                idx for idx in ind_search if curr_paras[idx - 1][1] == "GS"
            }
            for idx in ind_search:
                set_behavior(curr_paras[idx - 1], "LS")

            ind_search = [
                idx
                for idx in ind_search
                if behav_space[idx - 1] is not None
                and has_global_behavior(behav_space[idx - 1])
            ]

            ind_gs: list[int] = []
            for idx in ind_search:
                behav_entry = behav_space[idx - 1]
                entry = curr_paras[idx - 1]
                if (
                    _prefers_global_behavior(
                        behav_entry,
                        entry[0],
                        para_space[idx - 1],
                    )
                    and idx not in ind_gs
                ):
                    # One component may occur more than once in a path. Keep
                    # the global-candidate set unique: otherwise retaining one
                    # occurrence leaves a duplicate ID behind, and the later
                    # pruning loop removes every occurrence of the component,
                    # including a mutation paired with a retained crossover.
                    ind_gs.append(idx)

            if len(ind_gs) == 1:
                set_behavior(curr_paras[ind_gs[0] - 1], "GS")
            elif len(ind_gs) > 1:
                # Preserve a canonical choice already written by a prior
                # repair pass. Pruning repeated global nodes can change path
                # order, so simply taking the first current node is not
                # idempotent. An unmarked raw graph still uses graph order,
                # which is randomized by initialization/mutation.
                preferred = [idx for idx in ind_gs if idx in previously_global]
                retain = int(preferred[0] if preferred else ind_gs[0])
                # A crossover and its mandatory mutation form one decoded
                # search step. If both were marked global, retain the primary
                # crossover even if shared parameter metadata listed the
                # secondary first elsewhere in the path.
                for row in range(1, matrix.shape[0] - 1):
                    primary = int(matrix[row, 0])
                    secondary = int(matrix[row, 1])
                    if (
                        primary in ind_cross
                        and primary in preferred
                        and secondary in preferred
                        and secondary in ind_mu
                    ):
                        retain = primary
                        break
                rows_retain = np.where(matrix[:, 0] == retain)[0]
                if rows_retain.size > 1:
                    rows_to_delete = rng.choice(
                        rows_retain, size=rows_retain.size - 1, replace=False
                    )
                    rows_to_delete = np.sort(rows_to_delete)
                    for row in rows_to_delete[::-1]:
                        if row > 0:
                            matrix[row - 1, 1] = matrix[row, 1]
                        matrix = np.delete(matrix, row, axis=0)
                row_retain = np.where(matrix[:, 0] == retain)[0]
                retained = {retain}
                if (
                    retain in ind_cross
                    and row_retain.size > 0
                    and matrix[row_retain[0], 1] in ind_mu
                    and matrix[row_retain[0], 1] in ind_gs
                ):
                    retained.add(int(matrix[row_retain[0], 1]))
                for idx in retained:
                    set_behavior(curr_paras[idx - 1], "GS")
                    if idx in ind_gs:
                        ind_gs.remove(idx)

                for idx in list(ind_gs):
                    if has_local_behavior(behav_space[idx - 1]):
                        set_behavior(curr_paras[idx - 1], "LS")
                    else:
                        rows_to_remove = np.where(matrix[:, 0] == idx)[0]
                        for row in rows_to_remove[::-1]:
                            if row > 0:
                                matrix[row - 1, 1] = matrix[row, 1]
                            matrix = np.delete(matrix, row, axis=0)

            # Global-behaviour pruning reconnects predecessor/successor edges
            # after the earlier self-loop pass. That reconnection can itself
            # create ``[operator, operator]`` rows, so canonicalize once more
            # before exposing the repaired genotype or computing structure
            # keys. Category ranges are disjoint, hence the choose/search and
            # search/update boundary rows cannot both disappear.
            matrix = matrix[matrix[:, 0] != matrix[:, 1]]
            local_mu = [
                idx
                for idx in ind_mu
                if behav_space[idx - 1] is not None
                and has_local_behavior(behav_space[idx - 1])
            ]
            # Pruning can also remove the mutation node paired with a
            # crossover. Re-establish the crossover grammar after pruning,
            # selecting an LS-capable mutation for an LS crossover so the
            # repair does not accidentally introduce another global step.
            for row in range(1, matrix.shape[0] - 1):
                primary = int(matrix[row, 0])
                if primary not in ind_cross or int(matrix[row, 1]) in ind_mu:
                    continue
                primary_behavior = curr_paras[primary - 1][1]
                pool = ind_mu if primary_behavior == "GS" else (local_mu or ind_mu)
                secondary = int(pool[0])
                matrix[row, 1] = secondary
                matrix[row + 1, 0] = secondary
                set_behavior(
                    curr_paras[secondary - 1],
                    "GS" if primary_behavior == "GS" else "LS",
                )
            if matrix.shape[0] > 1 and int(matrix[-1, 0]) in ind_cross:
                primary = int(matrix[-1, 0])
                primary_behavior = curr_paras[primary - 1][1]
                pool = ind_mu if primary_behavior == "GS" else (local_mu or ind_mu)
                replacement = int(pool[0])
                matrix[-1, 0] = replacement
                matrix[-2, 1] = replacement
                set_behavior(
                    curr_paras[replacement - 1],
                    "GS" if primary_behavior == "GS" else "LS",
                )
            matrix = matrix[matrix[:, 0] != matrix[:, 1]]
            curr_ops[path_idx] = matrix

        repaired_ops.append(curr_ops)
        repaired_paras.append(curr_paras)

    return repaired_ops, repaired_paras


def _repair_states_equal(
    left_ops: list[list[np.ndarray]],
    left_paras: list[list[list[Any]]],
    right_ops: list[list[np.ndarray]],
    right_paras: list[list[list[Any]]],
) -> bool:
    if len(left_ops) != len(right_ops) or len(left_paras) != len(right_paras):
        return False
    for left_paths, right_paths in zip(left_ops, right_ops):
        if len(left_paths) != len(right_paths) or any(
            not np.array_equal(left, right)
            for left, right in zip(left_paths, right_paths)
        ):
            return False
    for left_entries, right_entries in zip(left_paras, right_paras):
        if len(left_entries) != len(right_entries):
            return False
        for left, right in zip(left_entries, right_entries):
            left_values = left[0] if left else None
            right_values = right[0] if right else None
            if left_values is None or right_values is None:
                if left_values is not None or right_values is not None:
                    return False
            elif not np.array_equal(left_values, right_values):
                return False
            left_behavior = left[1] if left and len(left) > 1 else None
            right_behavior = right[1] if right and len(right) > 1 else None
            if left_behavior != right_behavior:
                return False
    return True


def repair(
    operators: list[list[np.ndarray]],
    paras: list[list[list[Any]]],
    problem: Any,
    setting: Any,
):
    """Return the canonical fixed point of graph and behavior repair.

    Behaviour metadata is shared across pathways. A later path can therefore
    affect how an earlier path must be normalized; one left-to-right pass is
    insufficient. Iterate the deterministic repair closure and reject a graph
    if it cannot stabilize instead of exposing representation keys whose value
    depends on the number of repair calls.
    """

    from ._stream_graph import is_stream_graph
    from ._stream_graph import repair as repair_stream_graph

    if is_stream_graph(setting):
        return repair_stream_graph(operators, paras, problem, setting)

    current_ops = operators
    current_paras = paras
    for _ in range(16):
        repaired_ops, repaired_paras = _repair_once(
            current_ops, current_paras, problem, setting
        )
        if _repair_states_equal(
            current_ops,
            current_paras,
            repaired_ops,
            repaired_paras,
        ):
            return repaired_ops, repaired_paras
        current_ops, current_paras = repaired_ops, repaired_paras
    raise RuntimeError("Graph repair did not converge to a canonical fixed point.")
