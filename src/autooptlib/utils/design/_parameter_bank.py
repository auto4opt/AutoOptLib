"""Occurrence-aware parameter storage for the existing graph genotype.

The list payload remains the legacy component bank. When occurrence scope is
enabled, nodes is the authoritative active parameter mapping consumed by decode
and CMA. It survives deepcopy, checkpoints and worker transport.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from ._helpers import get_flex


class ParameterBank(list):
    def __init__(self, entries=(), nodes=None):
        super().__init__(entries)
        self.nodes = {} if nodes is None else nodes


def copy_bank(bank, entries=None):
    values = deepcopy(list(bank)) if entries is None else entries
    return (
        ParameterBank(values, deepcopy(bank.nodes))
        if isinstance(bank, ParameterBank)
        else values
    )


def iter_nodes(paths, setting):
    from ._stream_graph import _operator_category, path_nodes

    names = get_flex(setting, "all_op", required=True)
    categories = get_flex(setting, "op_space", required=True)
    for pi, path in enumerate(paths):
        nodes = path_nodes(path)
        position, si = 0, 0
        while position < len(nodes) - 1:
            choose = None
            if _operator_category(nodes[position], categories) == "choose":
                choose = names[nodes[position] - 1]
                position += 1
            yield (pi, si, "choose", choose)
            yield (pi, si, "search", names[nodes[position] - 1])
            position += 1
            si += 1
        yield (pi, -1, "update", names[nodes[-1] - 1])


def node_domain(key, setting):
    names = list(get_flex(setting, "all_op", required=True))
    index = names.index(key[3])
    raw = get_flex(setting, "para_space", required=True)[index]
    kinds = get_flex(setting, "para_type_space", required=True)[index]
    domain = get_flex(setting, "DesignSpace", None)
    if domain is not None:
        for node in domain.nodes:
            if node.location == key[:3]:
                override = dict(node.parameters).get(key[3])
                if override is not None:
                    proposed = np.asarray(override, dtype=float)
                    if raw is None or proposed.shape != np.asarray(raw).shape:
                        raise ValueError(
                            f"{key[:3]}: node parameter count does not match {key[3]}"
                        )
                    if key[3] in dict(domain.parameters) and (
                        np.any(proposed[:, 0] < raw[:, 0])
                        or np.any(proposed[:, 1] > raw[:, 1])
                    ):
                        raise ValueError(
                            f"{key[:3]}: node bounds exceed an explicit component domain"
                        )
                    from ...components.contracts import get_component_contract

                    contract = get_component_contract(key[3])
                    contract.validate_search_bounds(proposed)
                    for row, kind in zip(proposed, kinds):
                        if kind == "integer" and np.ceil(row[0]) > np.floor(row[1]):
                            raise ValueError(f"{key[:3]}: empty integer node domain")
                    raw = proposed
    return raw, kinds


def bind_bank(bank, paths, setting):
    scope = get_flex(setting, "ParameterScope", "component")
    if scope != "node" and not isinstance(bank, ParameterBank):
        return bank
    result = copy_bank(bank)
    if not isinstance(result, ParameterBank):
        result = ParameterBank(result)
    active = {}
    names = list(get_flex(setting, "all_op", required=True))
    rng = get_flex(setting, "rng", None)
    for key in iter_nodes(paths, setting):
        if key[3] is None:
            continue
        bounds, kinds = node_domain(key, setting)
        if key in result.nodes:
            value = result.nodes[key]
        else:
            entry = result[names.index(key[3])]
            value = (
                None if not entry or entry[0] is None else np.asarray(entry[0]).copy()
            )
            if (
                bounds is not None
                and value is not None
                and (np.any(value < bounds[:, 0]) or np.any(value > bounds[:, 1]))
            ):
                if rng is None:
                    value = bounds.mean(axis=1)
                else:
                    value = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * rng.random(
                        len(bounds)
                    )
                for i, kind in enumerate(kinds):
                    if kind == "integer":
                        value[i] = np.clip(
                            np.rint(value[i]),
                            np.ceil(bounds[i, 0]),
                            np.floor(bounds[i, 1]),
                        )
        if value is not None and bounds is not None:
            a = np.asarray(value)
            if (
                a.shape != (len(bounds),)
                or np.any(a < bounds[:, 0])
                or np.any(a > bounds[:, 1])
            ):
                raise ValueError(f"{key[:3]}: parameter value outside node domain")
        active[key] = None if value is None else np.asarray(value).copy()
    result.nodes = active
    return result


def node_loci(algorithm, setting):
    bank = algorithm.parameter
    if not isinstance(bank, ParameterBank):
        return []
    result = []
    for key, values in bank.nodes.items():
        bounds, kinds = node_domain(key, setting)
        if bounds is None or values is None:
            continue
        for i, (row, kind) in enumerate(zip(bounds, kinds)):
            lo, hi = row
            if kind == "integer":
                lo, hi = np.ceil(lo), np.floor(hi)
            if hi > lo:
                result.append((kind, key, i, float(lo), float(hi)))
    return result


def node_value(bank, pi, si, role, name, fallback):
    if not isinstance(bank, ParameterBank):
        return fallback
    value = bank.nodes.get((pi, si, role, name), fallback)
    return None if value is None else np.asarray(value).copy()


def matches_nodes(paths, setting):
    domain = get_flex(setting, "DesignSpace", None)
    if domain is None or not domain.nodes:
        return True
    actual = {key[:3]: key[3] for key in iter_nodes(paths, setting)}
    return all(
        node.location in actual and actual[node.location] in node.choices
        for node in domain.nodes
    )


def sample_constrained_paths(setting, rng, maximum_paths, maximum_stages):
    from ._population import is_mutation_component
    from ._stream_graph import _random_path_nodes, nodes_to_path

    domain = get_flex(setting, "DesignSpace", None)
    if domain is None or not domain.nodes:
        return None
    names = list(get_flex(setting, "all_op", required=True))
    ranges = get_flex(setting, "op_space", required=True)
    rules = {n.location: n.choices for n in domain.nodes}
    minimum_paths = max(n.pathway for n in domain.nodes) + 1
    if minimum_paths > maximum_paths:
        raise ValueError("Node constraint exceeds max_pathways")
    count = int(rng.integers(minimum_paths, maximum_paths + 1))
    paths = []

    def draw(pi, si, role, allowed):
        values = [v for v in rules.get((pi, si, role), allowed) if v in allowed]
        if not values:
            raise ValueError(f"{(pi, si, role)}: no admissible component")
        return values[int(rng.integers(len(values)))]

    for pi in range(count):
        local = [n for n in domain.nodes if n.pathway == pi]
        if not local:
            paths.append(
                nodes_to_path(_random_path_nodes(rng, setting, maximum_stages))
            )
            continue
        stages = max([n.stage + 1 for n in local if n.role != "update"] or [1])
        if stages > maximum_stages:
            raise ValueError("Node constraint exceeds max_stages")
        indices = []
        for si in range(stages):
            selectors = [None] + names[ranges[0, 0] - 1 : ranges[0, 1]]
            choose = draw(pi, si, "choose", selectors)
            if choose is not None:
                indices.append(names.index(choose) + 1)
            searches = names[ranges[1, 0] - 1 : ranges[1, 1]]
            if stages == 2:
                searches = (
                    [n for n in searches if n.startswith("cross_")]
                    if si == 0
                    else [n for n in searches if is_mutation_component(n)]
                )
            selected = draw(pi, si, "search", searches)
            indices.append(names.index(selected) + 1)
        update = draw(pi, -1, "update", names[ranges[2, 0] - 1 : ranges[2, 1]])
        indices.append(names.index(update) + 1)
        paths.append(nodes_to_path(indices))
    return paths
