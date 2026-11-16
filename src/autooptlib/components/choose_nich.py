"""Python translation of choose_nich."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ._execution import Computation, DependencyGuard, Primitive
from ._utils import (
    _input_bindings,
    ensure_rng,
    extract_fits,
    flex_get,
    pairwise_distances,
    rng_method,
)


@dataclass
class Species:
    seed: int
    idx: np.ndarray


def _extract_decisions(solution) -> np.ndarray:
    bindings = _input_bindings.get()
    if bindings is not None:
        return bindings.read(
            solution, float, "decision", lambda: _reference_decisions(solution)
        )
    return _reference_decisions(solution)


def _reference_decisions(solution):
    decs = flex_get(solution, "decs")
    if decs is None:
        raise ValueError("Solution object must provide decision vectors")
    return np.asarray(decs, dtype=float)


def _nearest_better_clustering(
    distance: np.ndarray, fai: float, min_size: int, g: int, gmax: int, _native=None
) -> List[Species]:
    if min_size == -1:
        # MATLAB rounds positive .5 values away from zero; Python uses bankers'
        # rounding, which changes species splitting at exact half steps.
        min_size = int(np.floor(3 + g / max(gmax, 1) * 7 + 0.5))

    npop = distance.shape[0]
    prefix = _cluster_prefix
    prepared = _native.prefix(prefix, distance) if _native is not None else None
    nearest, dist_val, follow = prefix(distance) if prepared is None else prepared

    meandis = fai * dist_val[1:].mean() if npop > 1 else 0.0
    order = np.argsort(dist_val)[::-1]
    finish = _cluster_finish
    values = (nearest, dist_val, follow, meandis, order, min_size)
    prepared = _native.finish(finish, values) if _native is not None else None
    seeds, roots = (
        finish(*values) if prepared is None else ([0, *prepared[0][1:]], prepared[1])
    )
    species = []
    for seed in seeds:
        members = np.where(roots == seed)[0]
        species.append(Species(seed=seed, idx=members))
    return species


def _cluster_prefix(distance):
    bindings = _input_bindings.get()
    if bindings is not None:
        result = bindings.frame.cluster_block("nich_nearest", (distance,))
        if result is not None:
            return result
    npop = distance.shape[0]
    nearest = np.full(npop, -1, dtype=int)
    dist_val = np.zeros(npop, dtype=float)

    for i in range(1, npop):
        segment = distance[i, :i]
        idx = int(np.argmin(segment))
        nearest[i] = idx
        dist_val[i] = segment[idx]

    follow = np.ones(npop, dtype=int)
    for i in range(npop - 1, 0, -1):
        parent = nearest[i]
        if parent >= 0:
            follow[parent] += follow[i]

    return nearest, dist_val, follow


def _cluster_finish(nearest, dist_val, follow, meandis, order, min_size):
    bindings = _input_bindings.get()
    if bindings is not None:
        result = bindings.frame.cluster_block(
            "nich_finish", (nearest, dist_val, follow, meandis, order, min_size)
        )
        if result is not None:
            # The initial seed is a Python int; later seeds are NumPy indices.
            return [0, *result[0][1:]], result[1]
    npop = len(nearest)
    seeds: List[int] = [0]
    for inf_index in order:
        if dist_val[inf_index] <= meandis:
            continue
        sup_index = nearest[inf_index]
        if sup_index < 0:
            continue
        chain = [sup_index]
        top_index = sup_index
        while nearest[top_index] != -1:
            top_index = nearest[top_index]
            chain.append(top_index)
        if (
            follow[inf_index] >= min_size
            and follow[top_index] - follow[inf_index] >= min_size
        ):
            nearest[inf_index] = -1
            dist_val[inf_index] = 0.0
            seeds.append(inf_index)
            for node in chain:
                follow[node] -= follow[inf_index]

    roots = np.zeros(npop, dtype=int)
    for i in range(npop):
        node = i
        parent = nearest[node]
        while parent != -1:
            node = parent
            parent = nearest[node]
        roots[i] = node if node != -1 else i

    return seeds, roots


def _read(args):
    problem = args[1] if len(args) > 1 else None
    aux = args[3] if len(args) > 3 else None
    g = int(args[4]) if len(args) > 4 else 0
    gmax = int(flex_get(problem, "Gmax", g + 1))
    rng = ensure_rng(aux)
    decs = _extract_decisions(args[0])
    fitness = extract_fits(args[0])
    if fitness.size != decs.shape[0]:
        raise ValueError("Fitness count must match decision rows for niching.")
    return decs, fitness, g, gmax, rng


def _distances(decs, fitness):
    # Stable fitness order, followed by the original global distance operation.
    fitness_order = np.argsort(fitness, kind="stable")
    return fitness_order, pairwise_distances(decs[fitness_order])


def _species(distance, g, gmax, _native=None):
    clustering = _nearest_better_clustering
    if _native is not None and _native.clustering(clustering):
        return clustering(distance, 1.0, -1, g, gmax, _native)
    return clustering(distance, 1.0, -1, g, gmax)


def _permutations(species, fitness_order, rng):
    index_list = []
    for specie in species:
        if specie.idx.size == 0:
            continue
        permuted = fitness_order[rng_method(rng, "permutation")(specie.idx)]
        index_list.extend(permuted.tolist())
    return index_list


def _order(index_list):
    if not index_list:
        return np.array([], dtype=int), None
    if len(index_list) % 2 == 1:
        odd = True
        index_list.append(index_list[-1])
    else:
        odd = False
    pairs = np.asarray(index_list, dtype=int).reshape(-1, 2)
    ordered = np.concatenate((pairs[:, 0], pairs[:, 1]))
    if odd:
        ordered = ordered[:-1]
    return ordered.astype(int), None


def choose_nich(*args):
    mode = args[-1]
    if mode == "execute":
        decs, fitness, g, gmax, rng = _read(args)
        fitness_order, distance = _distances(decs, fitness)
        species = _species(distance, g, gmax)
        index_list = _permutations(species, fitness_order, rng)
        return _order(index_list)

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["", ""], None

    raise ValueError(f"Unsupported mode: {mode}")


choose_nich.__execution_computation__ = Computation(
    choose_nich,
    (
        Primitive(
            "read-ordered-population",
            _read,
            ("args",),
            ("decs", "fitness", "g", "gmax", "rng"),
            ("external-read",),
            reads=(
                "population.decisions",
                "population.fitness",
                "generation",
                "Gmax",
                "auxiliary.rng",
            ),
        ),
        Primitive(
            "stable-order-and-global-distances",
            _distances,
            ("decs", "fitness"),
            ("fitness_order", "distance"),
            preparation="nich-distance",
        ),
        Primitive(
            "reference-nearest-better-clustering",
            _species,
            ("distance", "g", "gmax"),
            ("species",),
            preparation="nich-species",
        ),
        Primitive(
            "ordered-species-permutations",
            _permutations,
            ("species", "fitness_order", "rng"),
            ("indices",),
            ("rng",),
            writes=("rng",),
        ),
        Primitive("assemble-ordered-pairs", _order, ("indices",), ("result",)),
    ),
    native_dependencies=(
        DependencyGuard((_distances, np.geterr)),
        DependencyGuard((_cluster_prefix,)),
        DependencyGuard((_cluster_finish,)),
        (_nearest_better_clustering, _nearest_better_clustering.__code__),
    ),
)
# The skipped distance stage also depends on the current error policy reader.
choose_nich.__execution_computation__.native_dependencies[0].bindings.append(
    (vars(np), "geterr", np.geterr)
)
