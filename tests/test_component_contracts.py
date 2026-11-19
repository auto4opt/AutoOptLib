"""Contracts for component helpers, archives, and metadata modes."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from autooptlib.components import _utils
from autooptlib.components.archive_best import archive_best
from autooptlib.components.archive_diversity import archive_diversity
from autooptlib.components.archive_statistic import archive_statistic
from autooptlib.components.archive_tabu import archive_tabu
from autooptlib.utils.space import (
    _parameter_types,
    _to_behavior_matrix,
    _to_namespace,
    space,
)


@dataclass
class Item:
    dec: np.ndarray
    obj: float = 0.0
    con: object = None
    fit: object = None


@pytest.mark.parametrize(
    "component",
    [archive_best, archive_diversity, archive_statistic, archive_tabu],
)
def test_archive_metadata_and_unknown_modes(component):
    assert component("parameter") == (None, None)
    assert component("behavior") == (["", ""], None)
    with pytest.raises(ValueError, match="Unsupported mode"):
        component("unknown")


def test_tabu_archive_returns_latest_collection():
    solutions = [Item(np.array([0.0]))]
    assert archive_tabu(solutions, "execute") == (solutions, None)


def test_best_archive_fitness_fallbacks_and_errors():
    class WithFits:
        fits = np.array([2.0, 1.0])

    np.testing.assert_array_equal(_utils.extract_fits(WithFits()), [2.0, 1.0])
    items = [
        Item(np.array([0]), obj=3e8, con=None),
        Item(np.array([1]), obj=2e8, con=[0.0]),
        Item(np.array([2]), obj=0.0, con=[2.0]),
    ]
    archive, _ = archive_best(items, [items[0]], "execute")
    assert archive[-1] is items[1]
    with pytest.raises(ValueError, match="fitness information"):
        archive_best(1, [], "execute")
    with pytest.raises(ValueError, match="must expose"):
        archive_best([SimpleNamespace()], [], "execute")


def test_diversity_archive_empty_and_discrete_distance():
    assert archive_diversity([], [], SimpleNamespace(), "execute") == ([], None)
    solutions = [
        Item(np.array([1, 2])),
        Item(np.array([1, 3])),
        Item(np.array([3, 2])),
    ]
    problem = SimpleNamespace(type=["permutation"], N=2, rng=np.random.default_rng(4))
    archive, _ = archive_diversity(solutions, [], problem, "execute")
    assert len(archive) == 2


def test_statistic_archive_accepts_property_and_appends():
    solution = SimpleNamespace(fits=np.array([1.0, 3.0]))
    first, _ = archive_statistic(solution, None, "execute")
    second, _ = archive_statistic(solution, first, "execute")
    assert first.shape == (1, 2)
    assert second.shape == (2, 2)


def test_component_utility_contracts():
    assert _utils.flex_get(None, "x", 2) == 2
    assert _utils.flex_get({"some_value": 3}, "some_value") == 3
    assert _utils.flex_get(SimpleNamespace(value=lambda: 4), "value") == 4
    # Exact spelling has deterministic precedence over aliases such as N/n.
    assert _utils.flex_get(SimpleNamespace(N=4, n=99), "N") == 4
    assert _utils.flex_get(SimpleNamespace(N=4, n=99), "n") == 99
    generator = np.random.default_rng(1)
    assert _utils.ensure_rng(generator) is generator
    assert _utils.ensure_rng({"rng": generator}) is generator
    assert _utils.ensure_rng(SimpleNamespace(rng=generator)) is generator
    assert isinstance(_utils.ensure_rng(None), np.random.Generator)
    np.testing.assert_array_equal(_utils.to_numpy([1, 2]), [1, 2])

    assert _utils.extract_fits([SimpleNamespace(fitness=1.0)])[0] == 1.0
    assert _utils.extract_fits([SimpleNamespace(fits=2.0)])[0] == 2.0
    with pytest.raises(ValueError, match="must expose"):
        _utils.extract_fits([SimpleNamespace()])
    with pytest.raises(ValueError, match="scalar-like"):
        _utils.extract_fits([SimpleNamespace(fit=[1, 2])])
    with pytest.raises(ValueError, match="fitness information"):
        _utils.extract_fits(object())
    assert _utils.solution_as_list([1]) == [1]
    assert _utils.solution_as_list((1, 2)) == [1, 2]
    with pytest.raises(TypeError, match="indexable"):
        _utils.solution_as_list(object())

    distances = _utils.pairwise_distances(np.array([[0.0], [2.0]]))
    np.testing.assert_array_equal(distances, [[0.0, 2.0], [2.0, 0.0]])
    assert sorted(_utils.randperm(3, generator).tolist()) == [0, 1, 2]
    np.testing.assert_array_equal(_utils.reshape_pairs([1, 2]), [[1, 2]])
    with pytest.raises(ValueError, match="even length"):
        _utils.reshape_pairs([1])
    assert _utils.as_int("2") == 2
    assert _utils.as_int("x", 3) == 3
    assert _utils.ensure_column([1, 2]).shape == (2, 1)
    assert _utils.ensure_column([[1, 2]]).shape == (1, 2)


def test_space_conversion_and_parameter_domain_helpers():
    assert _to_namespace({"x": 1}).x == 1

    class Setting:
        x = 2

    assert _to_namespace(Setting()).x == 2
    assert _to_behavior_matrix(None) is None
    assert _to_behavior_matrix(1) is None
    assert _to_behavior_matrix((("LS", "small"), ("GS",))) == [
        ["LS", "small"],
        ["GS"],
    ]
    converted = _to_behavior_matrix(np.array([["LS", "large"], ["GS", ""]]))
    assert converted[0] == ["LS", "large"]
    assert _to_behavior_matrix([None, "GS"]) == [[], ["GS"]]

    def continuous_component(*args):
        raise ValueError("Unsupported mode: parameter_type")

    assert _parameter_types(continuous_component, None, 2) == (
        "continuous",
        "continuous",
    )

    def legacy_component(*args):
        raise ValueError(args[-1])

    assert _parameter_types(legacy_component, None, 1) == ("continuous",)

    def broken_component(*args):
        raise TypeError("bug inside parameter_type")

    with pytest.raises(TypeError, match="bug inside"):
        _parameter_types(broken_component, None, 1)

    def mixed_component(*args):
        return ["integer", "continuous"], None

    assert _parameter_types(mixed_component, None, 2) == (
        "integer",
        "continuous",
    )
    with pytest.raises(ValueError, match="metadata"):
        _parameter_types(mixed_component, None, 1)

    def invalid_component(*args):
        return ["categorical"], None

    with pytest.raises(ValueError, match="Unsupported parameter type"):
        _parameter_types(invalid_component, None, 1)
    with pytest.raises(NotImplementedError, match="continuous, discrete"):
        space(SimpleNamespace(type=["graph"]), SimpleNamespace())


@pytest.mark.parametrize(
    "name",
    [
        "cross_arithmetic",
        "cross_point_one",
        "cross_point_two",
        "cross_point_uniform",
        "cross_sim_binary",
        "search_de_current",
        "search_de_random",
        "search_eda",
    ],
)
@pytest.mark.parametrize("callable_decisions", [False, True])
def test_components_accept_population_decisions_without_changing_random_stream(
    name, callable_decisions
):
    from autooptlib import get_component

    matrix = np.arange(24, dtype=float).reshape(4, 6) / 10

    class Population:
        reads = 0

        @property
        def decs(self):
            self.reads += 1
            return (lambda: matrix) if callable_decisions else matrix

    population = Population()
    raw_rng = np.random.default_rng(42)
    population_rng = np.random.default_rng(42)
    operator = get_component(name)
    raw, _ = operator(matrix, None, None, {"rng": raw_rng}, 0, 1, "execute")
    actual, _ = operator(
        population, None, None, {"rng": population_rng}, 0, 1, "execute"
    )
    np.testing.assert_array_equal(actual, raw)
    np.testing.assert_array_equal(matrix, np.arange(24).reshape(4, 6) / 10)
    assert population_rng.bit_generator.state == raw_rng.bit_generator.state
    assert population.reads == 1
