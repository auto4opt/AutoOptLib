"""Extensive unit tests for individual components."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from autooptlib.components.archive_best import archive_best
from autooptlib.components.archive_diversity import archive_diversity
from autooptlib.components.archive_statistic import archive_statistic
from autooptlib.components.choose_brainstorm import choose_brainstorm
from autooptlib.components.choose_ica import choose_ica
from autooptlib.components.choose_nich import choose_nich
from autooptlib.components.choose_roulette_wheel import choose_roulette_wheel
from autooptlib.components.choose_tournament import choose_tournament
from autooptlib.components.choose_traverse import choose_traverse
from autooptlib.components.cross_arithmetic import cross_arithmetic
from autooptlib.components.cross_order_n import cross_order_n
from autooptlib.components.cross_order_two import cross_order_two
from autooptlib.components.cross_point_n import cross_point_n
from autooptlib.components.cross_point_one import cross_point_one
from autooptlib.components.cross_point_two import cross_point_two
from autooptlib.components.cross_point_uniform import cross_point_uniform
from autooptlib.components.cross_sim_binary import cross_sim_binary
from autooptlib.components.para_cma import para_cma
from autooptlib.components.para_cmaes import para_cmaes
from autooptlib.components.para_pso import para_pso
from autooptlib.components.reinit_continuous import reinit_continuous
from autooptlib.components.reinit_discrete import reinit_discrete
from autooptlib.components.reinit_permutation import reinit_permutation
from autooptlib.components.search_cma import search_cma
from autooptlib.components.search_de_current import search_de_current
from autooptlib.components.search_de_random import search_de_random
from autooptlib.components.search_eda import search_eda
from autooptlib.components.search_ica import search_ica
from autooptlib.components.search_insert import search_insert
from autooptlib.components.search_mu_polynomial import search_mu_polynomial
from autooptlib.components.search_mu_uniform import search_mu_uniform
from autooptlib.components.search_pso import search_pso
from autooptlib.components.search_reset_creep import search_reset_creep
from autooptlib.components.search_reset_rand import search_reset_rand
from autooptlib.components.search_scramble import search_scramble
from autooptlib.components.search_swap import search_swap
from autooptlib.components.search_swap_multi import search_swap_multi
from autooptlib.components.update_greedy import update_greedy
from autooptlib.components.update_pairwise import update_pairwise
from autooptlib.components.update_round_robin import update_round_robin
from autooptlib.components.update_simulated_annealing import update_simulated_annealing


class DummyPopulation:
    """Simple helper exposing fits/decs/objs."""

    def __init__(self, decs: np.ndarray, fits: np.ndarray):
        self._decs = np.asarray(decs, dtype=float)
        self._fits = np.asarray(fits, dtype=float)

    def decs(self):
        return self._decs

    def fits(self):
        return self._fits

    def objs(self):
        return self._fits.reshape(-1, 1)


@dataclass
class DummySolution:
    dec: np.ndarray
    fit: float
    con: float = 0.0

    def __post_init__(self):
        self.dec = np.asarray(self.dec, dtype=float)

    def fits(self):
        return np.array([self.fit])


# ---------------------------------------------------------------------------
# Choose operators
# ---------------------------------------------------------------------------


def test_choose_traverse_indexes_all():
    problem = SimpleNamespace(N=5)
    idx, meta = choose_traverse(
        DummyPopulation(np.zeros((5, 1)), np.arange(5)), problem, None, None, "execute"
    )
    assert np.all(idx == np.arange(5))
    assert meta is None


def test_choose_tournament_returns_indices():
    problem = SimpleNamespace(N=3)
    rng_aux = {"rng": np.random.default_rng(0)}
    idx, _ = choose_tournament(
        DummyPopulation(np.zeros((4, 1)), [4, 1, 3, 2]),
        problem,
        None,
        rng_aux,
        "execute",
    )
    assert idx.shape == (problem.N,)


def test_choose_roulette_wheel_biases_low_fitness():
    problem = SimpleNamespace(N=2)
    rng_aux = {"rng": np.random.default_rng(1)}
    idx, _ = choose_roulette_wheel(
        DummyPopulation(np.zeros((3, 1)), [3, 1, 5]), problem, None, rng_aux, "execute"
    )
    assert idx.shape == (problem.N,)
    assert np.all(idx < 3)


def test_choose_ica_records_imperialist_count():
    pop = DummyPopulation(np.zeros((3, 1)), [3, 1, 2])
    idx, meta = choose_ica(pop, None, np.array([2]), "execute")
    assert np.array_equal(idx, np.array([1, 2, 0]))
    assert meta is None or meta.get("imperialist_count") >= 1


def test_choose_brainstorm_runs_kmeans():
    solution = DummyPopulation(
        np.array([[0.0, 0.0], [5.0, 5.0], [4.0, 4.5], [-1.0, -1.2]]), [1, 2, 3, 4]
    )
    problem = SimpleNamespace(type=["continuous", "static"])
    rng_aux = {"rng": np.random.default_rng(2)}
    idx, _ = choose_brainstorm(
        solution, problem, np.array([2, 0.5]), rng_aux, "execute"
    )
    assert idx.shape[0] == len(solution.fits())


def test_choose_nich_returns_shuffled_indices():
    decs = np.array([[0, 0], [0, 1], [5, 5], [5, 6]], dtype=float)
    solution = SimpleNamespace(decs=lambda: decs)
    problem = SimpleNamespace(Gmax=5)
    rng_aux = {"rng": np.random.default_rng(3)}
    idx, _ = choose_nich(solution, problem, None, rng_aux, 1, "execute")
    assert sorted(idx.tolist()) == list(range(decs.shape[0]))


# ---------------------------------------------------------------------------
# Crossover operators
# ---------------------------------------------------------------------------


def test_cross_point_variants_preserve_shape():
    parent = np.arange(12, dtype=float).reshape(6, 2)
    for fn in (cross_point_one, cross_point_two, cross_point_uniform, cross_point_n):
        child, _ = fn(parent, None, None, None, "execute")
        assert child.shape == parent.shape


def test_cross_arithmetic_and_sim_binary_bounds():
    parent = np.linspace(0, 1, 8).reshape(4, 2)
    arithmetic, _ = cross_arithmetic(parent, None, np.array([0.3]), None, "execute")
    sbx, _ = cross_sim_binary(parent, None, np.array([20.0]), None, "execute")
    for offspring in (arithmetic, sbx):
        assert offspring.shape == parent.shape
        assert np.all(offspring >= -0.1) and np.all(offspring <= 1.1)


def test_cross_order_variants_return_permutations():
    parent = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    for fn in (cross_order_two, cross_order_n):
        child, _ = fn(parent, None, None, None, "execute")
        assert child.shape == parent.shape
        for row in child:
            assert sorted(row.tolist()) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Search operators
# ---------------------------------------------------------------------------


def test_search_de_current_shape_matches():
    parent = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
    offspring, _ = search_de_current(
        parent, None, np.array([[0.5, 0.8], [0.5, 0.8]]), None, "execute"
    )
    assert offspring.shape == parent.shape


def test_search_de_random_returns_population():
    parent = np.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]])
    offspring, _ = search_de_random(
        parent, None, np.array([[0.6, 0.7]]), None, "execute"
    )
    assert offspring.shape == parent.shape


def test_search_mu_uniform_within_bounds():
    parent = np.array([[0.2, 0.2], [0.5, 0.5]])
    problem = SimpleNamespace(bound=np.array([[0.0, 0.0], [1.0, 1.0]]))
    offspring, _ = search_mu_uniform(parent, problem, np.array([0.5]), None, "execute")
    assert np.all(offspring >= 0.0) and np.all(offspring <= 1.0)


def test_search_reset_rand_respects_integer_bounds():
    parent = np.array([[0, 0], [1, 1]])
    problem = SimpleNamespace(bound=np.array([[0, 0], [5, 5]]))
    offspring, _ = search_reset_rand(
        parent, problem, np.array([0.5]), {"rng": np.random.default_rng(0)}, "execute"
    )
    assert offspring.shape == parent.shape
    assert np.issubdtype(offspring.dtype, np.integer)


def test_search_eda_samples_from_distribution():
    parent = np.array([[0.0, 0.0], [1.0, 1.0]])
    offspring, _ = search_eda(parent, None, None, None, "execute")
    assert offspring.shape == parent.shape


def test_search_ica_respects_auxiliary_imperialist_count():
    parent = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
    problem = SimpleNamespace(bound=np.array([[0.0, 0.0], [10.0, 10.0]]))
    aux = {"imperialist_count": 2}
    offspring, aux_out = search_ica(
        parent, problem, np.array([0.7, 0.1]), aux, "execute"
    )
    assert offspring.shape == parent.shape
    assert aux_out["imperialist_count"] == 2


def test_search_mu_polynomial_respects_bounds():
    parent = np.array([[0.1, 0.9], [0.4, 0.6]])
    problem = SimpleNamespace(bound=np.array([[0.0, 0.0], [1.0, 1.0]]))
    offspring, _ = search_mu_polynomial(
        parent, problem, np.array([0.3, 20.0]), None, "execute"
    )
    assert offspring.shape == parent.shape
    assert np.all(offspring >= 0.0) and np.all(offspring <= 1.0)


def test_search_reset_creep_returns_integers():
    parent = np.array([[1, 2, 3], [3, 2, 1]])
    problem = SimpleNamespace(bound=np.array([[0, 0, 0], [5, 5, 5]]))
    offspring, _ = search_reset_creep(
        parent, problem, np.array([0.5]), {"rng": np.random.default_rng(9)}, "execute"
    )
    assert offspring.shape == parent.shape
    assert np.issubdtype(offspring.dtype, np.integer)


def test_search_swap_variants_return_permutations():
    parent = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    for fn in (search_swap, search_swap_multi, search_scramble, search_insert):
        offspring, _ = fn(
            parent, None, None, {"rng": np.random.default_rng(10)}, "execute"
        )
        assert offspring.shape == parent.shape
        for row in offspring:
            assert sorted(row.tolist()) == [1, 2, 3, 4]


def test_search_pso_creates_velocity_and_gbest():
    from autooptlib.utils.solve import SolutionSet

    class Particle:
        def __init__(self, dec, fit):
            self.dec = np.asarray(dec, dtype=float)
            self.fit = fit

    sol = SolutionSet([Particle([0.0, 0.0], 1.0), Particle([1.0, 1.0], 0.5)])
    problem = SimpleNamespace(bound=np.array([[0.0, 0.0], [2.0, 2.0]]))
    aux = {}
    offspring, aux_out = search_pso(sol, problem, np.array([0.5]), aux, "execute")
    assert offspring.shape == (len(sol), len(sol[0].dec))
    assert "V" in aux_out and aux_out["V"].shape == offspring.shape
    assert "Pbest" in aux_out and "Gbest" in aux_out


def test_search_cma_initializes_auxiliary_state():
    from autooptlib.utils.solve import SolutionSet

    class Agent:
        def __init__(self, dec):
            self.dec = np.asarray(dec, dtype=float)

    sol = SolutionSet([Agent([0.0, 0.0]), Agent([1.0, -1.0])])
    problem = SimpleNamespace(bound=np.array([[0.0, 0.0], [1.0, 1.0]]))
    offspring, aux = search_cma(sol, problem, None, None, "execute")
    assert offspring.shape == (len(sol), len(sol[0].dec))
    assert "cma_mean" in aux and aux["cma_mean"].shape == (len(sol[0].dec),)


# ---------------------------------------------------------------------------
# Reinitializers
# ---------------------------------------------------------------------------


def test_reinit_continuous_returns_within_bounds():
    solution = [DummySolution([0.0, 0.0], 0.0)]
    problem = SimpleNamespace(bound=np.array([[0.0, 0.0], [1.0, 1.0]]))
    offspring, _ = reinit_continuous(
        solution, problem, None, {"rng": np.random.default_rng(4)}, "execute"
    )
    assert offspring.shape == (len(solution), 2)
    assert np.all(offspring >= 0.0) and np.all(offspring <= 1.0)


def test_reinit_discrete_generates_integers():
    solution = np.array([[0, 0], [1, 1]], dtype=int)
    problem = SimpleNamespace(bound=np.array([[0, 0], [3, 3]]))
    offspring, _ = reinit_discrete(
        solution, problem, None, {"rng": np.random.default_rng(5)}, "execute"
    )
    assert offspring.shape == solution.shape
    assert np.issubdtype(offspring.dtype, np.integer)


def test_reinit_permutation_generates_permutations():
    solution = np.array([[1, 2, 3], [3, 2, 1]], dtype=int)
    problem = SimpleNamespace(bound=np.array([[1, 2, 3], [1, 2, 3]]))
    offspring, _ = reinit_permutation(
        solution, problem, None, {"rng": np.random.default_rng(6)}, "execute"
    )
    assert offspring.shape == solution.shape
    for row in offspring:
        assert sorted(row.tolist()) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Update operators & archives
# ---------------------------------------------------------------------------


def test_update_greedy_selects_best_n():
    sols = [DummySolution([0], fit) for fit in [5, 1, 3]]
    problem = SimpleNamespace(N=2)
    updated, _ = update_greedy(sols, problem, None, None, "execute")
    fits = [sol.fit for sol in updated]
    assert len(updated) == 2 and fits == [1, 3]


def test_update_pairwise_compares_old_new_pairs():
    sols = [DummySolution([0], fit) for fit in [1, 2, 3, 4]]
    updated, _ = update_pairwise(sols, None, None, None, "execute")
    assert len(updated) == 2
    assert updated[0].fit == min(sols[0].fit, sols[2].fit)
    assert updated[1].fit == min(sols[1].fit, sols[3].fit)


def test_update_round_robin_keeps_high_scores():
    sols = [DummySolution([0], fit) for fit in [1, 2, 3, 4]]
    problem = SimpleNamespace(N=2)
    updated, _ = update_round_robin(
        sols, problem, None, {"rng": np.random.default_rng(7)}, "execute"
    )
    assert len(updated) == 2


def test_update_simulated_annealing_uses_acceptance():
    old = [DummySolution([0], fit) for fit in [5, 6, 7]]
    new = [DummySolution([0], fit) for fit in [4, 8, 6.5]]
    combined = old + new
    problem = SimpleNamespace(N=3)
    updated, _ = update_simulated_annealing(
        combined, problem, np.array([0.5]), {}, 1, "execute"
    )
    assert len(updated) == problem.N


def test_archive_best_appends_minimum():
    sols = [DummySolution([0], fit) for fit in [3, 1, 2]]
    archive, _ = archive_best(sols, [], "execute")
    assert len(archive) == 1 and archive[0].fit == 1


def test_archive_diversity_returns_spread():
    sols = DummyPopulation(np.array([[0, 0], [1, 1], [2, 2]]), [1, 2, 3])
    archive, _ = archive_diversity(
        sols, [], SimpleNamespace(type=["continuous"]), "execute"
    )
    assert len(archive) > 0


def test_archive_statistic_stores_mean_std():
    sols = DummyPopulation(np.array([[0], [1]]), [1.0, 2.0])
    stats, _ = archive_statistic(sols, [], "execute")
    assert stats.shape == (1, 2)


# ---------------------------------------------------------------------------
# Parameter updates
# ---------------------------------------------------------------------------


def _make_cma_aux(dim: int) -> dict[str, Any]:
    rng = np.random.default_rng(8)
    half_n = 2
    w = np.array([0.7, 0.3])
    disturbance = rng.normal(size=(3, dim))
    return {
        "cma_Disturb": disturbance,
        "cma_halfN": half_n,
        "cma_w": w,
        "cma_betterN": 1.0,
        "cma_mean": np.zeros(dim),
        "cma_sigma": np.ones(dim),
        "cma_csigma": 0.5,
        "cma_dsigma": 1.0,
        "cma_chiN": 1.0,
        "cma_cc": 0.5,
        "cma_ccov": 0.2,
        "cma_cmu": 0.3,
        "cma_hth": 10.0,
        "cma_ps": np.zeros(dim),
        "cma_pc": np.zeros(dim),
        "cma_C": np.eye(dim),
    }


def test_para_cma_updates_parameters_for_solution():
    aux = _make_cma_aux(2)
    solution = DummyPopulation(np.zeros((3, 2)), [3.0, 1.0, 2.0])
    problem = [SimpleNamespace(Gmax=5)]
    updated = para_cma(solution, problem, aux, "solution")
    assert "cma_mean" in updated and updated["cma_mean"].shape == (2,)


def test_para_cmaes_delegates_to_para_cma():
    aux = _make_cma_aux(2)
    solution = DummyPopulation(np.zeros((3, 2)), [2.0, 3.0, 1.0])
    problem = [SimpleNamespace(Gmax=5)]
    updated = para_cmaes(solution, problem, aux, "solution")
    assert "cma_mean" in updated


def test_para_pso_updates_pbest_and_gbest():
    from autooptlib.utils.solve import SolutionSet

    class Particle:
        def __init__(self, dec, fit):
            self.dec = np.asarray(dec, dtype=float)
            self.fit = fit
            self.obj = fit
            self.con = 0.0

    sol = SolutionSet([Particle([0.0, 0.0], 3.0), Particle([1.0, 1.0], 1.0)])
    problem = SimpleNamespace()
    aux = {"Pbest": SolutionSet([Particle([2.0, 2.0], 2.0), Particle([2.5, 2.5], 1.5)])}
    new_aux = para_pso(sol, problem, aux)
    assert "Pbest" in new_aux and len(new_aux["Pbest"]) == len(sol)
    assert "Gbest" in new_aux
