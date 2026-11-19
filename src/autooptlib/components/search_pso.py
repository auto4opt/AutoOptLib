"""Python translation of search_pso."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils.solve import Solution, SolutionSet
from ._utils import solution_matrix, ensure_rng, flex_get, input_boundary, write_aux, assign_rows, append_aux, rng_method, _input_bindings
from ._execution import Computation, Primitive, Requirement, NativePattern, DependencyGuard

_STATE_ZEROS = np.zeros
_STATE_NUMPY = np


def _ensure_solution_set(solution: Any) -> SolutionSet:
    if isinstance(solution, SolutionSet):
        return solution
    return SolutionSet(solution)


def _extract_dec_matrix(solution_set: SolutionSet) -> np.ndarray:
    return solution_matrix(solution_set)


_SHAPE_READERS = DependencyGuard((_extract_dec_matrix, solution_matrix))
_PSO_DEC_READER = _extract_dec_matrix


def _read(args):
    para = args[2] if len(args) > 2 else None
    aux = args[3] if len(args) > 3 else None
    rng = ensure_rng(aux)
    inertia = float(np.asarray(para).reshape(-1)[0]) if para is not None else 0.5
    sol_set = _ensure_solution_set(args[0])
    decs = _extract_dec_matrix(sol_set)
    context = aux.get("_population_context") if isinstance(aux, dict) else None
    population = sol_set if context is None else _ensure_solution_set(context)
    bindings = _input_bindings.get()
    shape = (bindings.shape_only(population)
             if bindings is not None and bindings.frame.native_blocks
             and _extract_dec_matrix is _PSO_DEC_READER
             and _SHAPE_READERS.trusted() else None)
    if shape is None:
        population_decs = _extract_dec_matrix(population)
        n, d = population_decs.shape
    else:
        population_decs = None  # private unused input-prefix register
        n, d = shape
    input_boundary("input-prefix-end")
    return aux, rng, inertia, sol_set, decs, context, population, population_decs, n, d


def _state(aux, population, rng, n, d):
    if aux is None or not isinstance(aux, dict):
        aux = {}
    if "Pbest" not in aux:
        write_aux(aux, "Pbest", SolutionSet(list(population)))
        write_aux(aux, "Gbest", aux["Pbest"][int(rng_method(rng, "integers")(n))])
        write_aux(aux, "V", np.zeros((n, d)))
    if (type(aux) is dict and "V" in aux and np is _STATE_NUMPY
            and vars(np).get("zeros") is _STATE_ZEROS):
        velocity = np.asarray(aux["V"], dtype=float)
    else:
        velocity = np.asarray(aux.get("V", np.zeros((n, d))), dtype=float)
    bindings = _input_bindings.get()
    pbest = None
    if (bindings is not None and bindings.frame.native_blocks and bindings.reuse_enabled
            and type(aux) is dict and type(rng) is np.random.Generator
            and aux.get("_population_context") is not None
            and _ROW_STATE_GUARD.trusted()
            and all(globals().get(name) is value for name,value in _ROW_STATE_HELPERS.items())
            and type(aux.get("Gbest")) is Solution):
        guide = aux["Gbest"]
        # The guide reader must not invoke a user callback after this snapshot.
        if (type(guide).__getattribute__ is object.__getattribute__
                and "dec" not in vars(type(guide))
                and type(vars(guide).get("dec")) is np.ndarray
                and guide.dec.dtype == np.dtype(float) and guide.dec.shape == (d,)):
            pbest = bindings.selected_snapshot(aux["Pbest"],
                                               aux.get("_target_population_indices"),(n,d))
    if pbest is None:
        pbest = _extract_dec_matrix(aux["Pbest"])
    gbest = np.asarray(flex_get(aux["Gbest"], "dec"), dtype=float).reshape(1, -1)
    return aux, velocity, pbest, gbest


def _targets(aux, context, sol_set, n):
    if context is None:
        return np.arange(len(sol_set), dtype=int), None
    raw_indices = aux.get("_target_population_indices")
    indices = np.asarray(raw_indices, dtype=int).reshape(-1)
    if indices.size != len(sol_set) or np.any((indices < 0) | (indices >= n)):
        raise ValueError("Stream PSO requires one current-population index per target.")
    return indices, raw_indices


def _coefficient(rng, target_indices):
    return rng_method(rng, "random")((len(target_indices), 1))


def _velocity(inertia, velocity, target_indices, r1, pbest, decs, r2, gbest):
    return (inertia * velocity[target_indices]
            + 2 * r1 * (pbest[target_indices] - decs)
            + 2 * r2 * (gbest - decs))


def _offspring(decs, target_velocity):
    return decs + target_velocity


def _commit(velocity, target_indices, target_velocity, aux, context, offspring):
    assign_rows(velocity, target_indices, target_velocity, field="V")
    write_aux(aux, "V", velocity)
    if context is not None:
        append_aux(aux, "_stream_pso_target_indices", (int(index) for index in target_indices))
    return offspring, aux


_ROW_STATE_HELPERS = {name: globals()[name] for name in
                      ("_state","_targets","_coefficient","_velocity","_offspring",
                       "_commit","_extract_dec_matrix","solution_matrix")}
_ROW_STATE_GUARD = DependencyGuard(tuple(_ROW_STATE_HELPERS.values()))


def search_pso(*args):
    """Particle swarm optimization particle fly operator."""

    mode = args[-1]
    if mode == "execute":
        aux, rng, inertia, sol_set, decs, context, population, population_decs, n, d = _read(args)
        aux, velocity, pbest, gbest = _state(aux, population, rng, n, d)
        target_indices, _target_owner = _targets(aux, context, sol_set, n)
        r1 = _coefficient(rng, target_indices)
        r2 = _coefficient(rng, target_indices)
        target_velocity = _velocity(inertia, velocity, target_indices, r1, pbest, decs, r2, gbest)
        offspring = _offspring(decs, target_velocity)
        return _commit(velocity, target_indices, target_velocity, aux, context, offspring)

    if mode == "parameter":
        return [0, 0.5], None

    if mode == "behavior":
        return ["", "GS"], None

    raise ValueError(f"Unsupported mode: {mode}")


search_pso.__execution_computation__ = Computation(search_pso, (
    Primitive("bind-current-particle-inputs", _read, ("args",),
              ("aux", "rng", "inertia", "sol_set", "decs", "context", "population", "population_decs", "n", "d"),
              ("parameter-read", "external-read"), reads=("parameters", "parents.decisions", "population.decisions", "auxiliary.rng")),
    Primitive("bind-or-initialize-pso-state", _state, ("aux", "population", "rng", "n", "d"),
              ("current_aux", "velocity", "pbest", "gbest"), ("state-read", "conditional-rng", "state-write"),
              reads=("Pbest", "Gbest", "V"), writes=("Pbest", "Gbest", "V", "rng"),
              requirements=(Requirement("auxiliary.Pbest", rows="selected", reader="solution"),)),
    Primitive("bind-ordered-targets", _targets, ("current_aux", "context", "sol_set", "n"), ("indices", "target_owner"),
              ("state-read",), reads=("target_indices",)),
    Primitive("draw-particle-r1", _coefficient, ("rng", "indices"), ("r1",), ("rng",), writes=("rng",)),
    Primitive("draw-particle-r2", _coefficient, ("rng", "indices"), ("r2",), ("rng",), writes=("rng",)),
    Primitive("particle-velocity", _velocity, ("inertia", "velocity", "indices", "r1", "pbest", "decs", "r2", "gbest"), ("next_velocity",)),
    Primitive("particle-position", _offspring, ("decs", "next_velocity"), ("offspring",)),
    Primitive("commit-current-pso-state", _commit, ("velocity", "indices", "next_velocity", "current_aux", "context", "offspring"),
              ("result",), ("state-write",), writes=("V", "pso_target_indices")),
), native_patterns=(NativePattern("pso", ("particle-velocity", "particle-position"),
    ("inertia", "velocity", "indices", "r1", "pbest", "decs", "r2", "gbest"),
    ("next_velocity", "offspring")),))
