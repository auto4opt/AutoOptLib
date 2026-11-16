"""Python translation of search_de_current."""
from ._differential import prepare, parameters, donors_current, crossover_mask, vary, result, specification


def search_de_current(*args):
    mode = args[-1]
    if mode == "execute":
        rng, targets, population, context, n, d, gbest, _owners = prepare(args)
        f, cr, _parameter_owner = parameters(args)
        base, p2, p3 = donors_current(targets, population, context, rng, n, gbest)
        mask = crossover_mask(rng, n, d, cr)
        return result(vary(targets, base, p2, p3, mask, f), args)
    if mode == "parameter":
        return [[0, 1], [0, 1]], None
    if mode == "behavior":
        return [["LS", "small", "small"], ["GS", "large", "large"]], None
    raise ValueError(f"Unsupported mode: {mode}")


search_de_current.__execution_computation__ = specification(search_de_current, donors_current)
