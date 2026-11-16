"""Reference differential-evolution primitives shared by component and plan."""
from functools import partial

import numpy as np

from ._execution import Computation, Primitive, NativePattern
from ._utils import decision_matrix, ensure_rng, input_boundary, population_matrix, rng_method


def prepare(args, best=False):
    parent = args[0]
    aux = args[3] if len(args) > 3 else None
    rng = ensure_rng(aux, args[1] if len(args) > 1 else None)
    gbest = None
    owners = ()
    if best:
        context = aux.get("_population_context") if isinstance(aux, dict) else None
        population_obj = parent if context is None else context
        fits = getattr(population_obj, "fits", None)
        if callable(fits):
            fit_vals = np.asarray(fits()).reshape(-1)
        else:
            fit_vals = np.asarray(fits if fits is not None else getattr(population_obj, "fit", None)).reshape(-1)
        best_idx = int(np.argmin(fit_vals))
        best_dec = getattr(population_obj[best_idx], "dec", None)
        if callable(best_dec):
            gbest = np.asarray(best_dec())
        else:
            gbest = np.asarray(best_dec if best_dec is not None else getattr(population_obj, "decs")[best_idx])
        targets = decision_matrix(parent, dtype=float)
        population = population_matrix(population_obj)
        owners = (population_obj, fits, fit_vals, best_dec)
    else:
        targets = decision_matrix(parent, dtype=float)
        context = aux.get("_population_context") if isinstance(aux, dict) else None
        population = targets if context is None else population_matrix(context)
    n, d = targets.shape
    input_boundary("input-prefix-end")
    return rng, targets, population, context, n, d, gbest, owners


def parameters(args):
    para = args[2] if len(args) > 2 else None
    if para is None:
        return 0.5, 0.5, None
    arr = np.asarray(para).reshape(-1)
    return float(arr[0]), float(arr[1]) if arr.size > 1 else 0.5, arr


def donors_current(targets, population, context, rng, n, gbest):
    if context is None:
        p2 = population[rng_method(rng, "permutation")(n)]
        p3 = population[rng_method(rng, "permutation")(n)]
    else:
        p2 = population[rng_method(rng, "integers")(0, len(population), size=n)]
        p3 = population[rng_method(rng, "integers")(0, len(population), size=n)]
    return targets, p2, p3


def donors_random(targets, population, context, rng, n, gbest):
    if context is None:
        p1 = population[rng_method(rng, "permutation")(n)]
        p2 = population[rng_method(rng, "permutation")(n)]
        p3 = population[rng_method(rng, "permutation")(n)]
    else:
        p1 = population[rng_method(rng, "integers")(0, len(population), size=n)]
        p2 = population[rng_method(rng, "integers")(0, len(population), size=n)]
        p3 = population[rng_method(rng, "integers")(0, len(population), size=n)]
    return p1, p2, p3


def donors_best(targets, population, context, rng, n, gbest):
    p2 = np.repeat(gbest.reshape(1, -1), n, axis=0)
    if context is None:
        p3 = population[rng_method(rng, "permutation")(n)]
    else:
        p3 = population[rng_method(rng, "integers")(0, len(population), size=n)]
    return targets, p2, p3


def crossover_mask(rng, n, d, cr):
    return rng_method(rng, "random")((n, d)) < cr


def vary(targets, base, p2, p3, mask, f):
    offspring = targets.copy()
    donor = base + f * (p2 - p3)
    offspring[mask] = donor[mask]
    return offspring


def result(offspring, args):
    return offspring, args[3] if len(args) > 3 else None


def specification(reference, donors, best=False):
    return Computation(reference, (
        Primitive("read-current-matrices-and-guide", partial(prepare, best=True) if best else prepare, ("args",),
                  ("rng", "targets", "population", "context", "n", "d", "gbest", "read_owners"),
                  ("external-read",), reads=("parents.decisions", "population.decisions", "fitness", "auxiliary.rng")),
        Primitive("bind-current-parameters", parameters, ("args",), ("f", "cr", "parameter_owner"),
                  ("parameter-read",), reads=("parameters",)),
        Primitive("draw-ordered-donors", donors, ("targets", "population", "context", "rng", "n", "gbest"),
                  ("base", "p2", "p3"), ("rng",), writes=("rng",)),
        Primitive("draw-coordinate-mask", crossover_mask, ("rng", "n", "d", "cr"), ("mask",), ("rng",), writes=("rng",)),
        Primitive("differential-arithmetic", vary, ("targets", "base", "p2", "p3", "mask", "f"), ("offspring",)),
        Primitive("return-current-auxiliary", result, ("offspring", "args"), ("result",), binding_point=4),
    ), native_patterns=(NativePattern("de_vary", ("differential-arithmetic",),
        ("targets", "base", "p2", "p3", "mask", "f"), ("offspring",)),))
