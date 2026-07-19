"""Behavioral tests for the public custom-problem contract."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from autooptlib import ProblemDefinition, make_problem
from autooptlib.problems.base import validate_constructed_problems


def test_problem_factory_supports_data_constraints_and_direct_modes():
    definition = make_problem(
        lambda decision, data: float(np.sum(decision) + data["offset"]),
        bounds=lambda instance: [[-instance, -instance], [instance, instance]],
        constraint=lambda decision, data: [decision[0] - data["limit"]],
        data_factory=lambda instance: {"offset": instance, "limit": 0.5},
        name="bounded_sum",
    )
    assert isinstance(definition, ProblemDefinition)
    records = [SimpleNamespace(N=2, Gmax=3)]
    problems, data, auxiliary = definition(records, [2], "construct")
    validate_constructed_problems(problems, data)
    assert auxiliary is None
    assert definition.__name__ == "bounded_sum"
    assert problems[0].dimension == 2

    repaired, _, _ = definition(None, [1.0, -1.0], "repair")
    np.testing.assert_array_equal(repaired, [1.0, -1.0])
    value, violation, _ = definition(data[0], [1.0, 2.0], "evaluate")
    assert value == 5.0
    assert violation == [0.5]
    values, violations, _ = definition(data[0], [[0.0, 0.0], [1.0, 1.0]], "evaluate")
    np.testing.assert_array_equal(values, [2.0, 4.0])
    np.testing.assert_array_equal(violations, [[-0.5], [0.5]])


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"objective": 1, "bounds": (-1, 1)}, TypeError, "objective"),
        (
            {"objective": lambda d, x: 0, "bounds": (-1, 1), "problem_type": "graph"},
            ValueError,
            "problem_type",
        ),
        (
            {"objective": lambda d, x: 0, "bounds": (-1, 1), "constraint": 1},
            TypeError,
            "constraint",
        ),
        (
            {"objective": lambda d, x: 0, "bounds": (-1, 1), "data_factory": 1},
            TypeError,
            "data_factory",
        ),
    ],
)
def test_problem_factory_rejects_invalid_arguments(kwargs, error, message):
    with pytest.raises(error, match=message):
        make_problem(**kwargs)


def test_problem_factory_rejects_missing_or_unexpandable_bounds():
    continuous = make_problem(lambda decision, data: 0.0)
    with pytest.raises(ValueError, match="require explicit bounds"):
        continuous([SimpleNamespace()], [2], "construct")

    scalar_bounds = make_problem(lambda decision, data: 0.0, bounds=(-1, 1))
    with pytest.raises(ValueError, match="integer instance dimension"):
        scalar_bounds([SimpleNamespace()], [object()], "construct")


def test_problem_factory_validates_construct_count_and_mode():
    definition = make_problem(lambda decision, data: 0.0, bounds=(-1, 1))
    with pytest.raises(ValueError, match="number of problem records"):
        definition([], [2], "construct")
    with pytest.raises(ValueError, match="Unsupported problem mode"):
        definition([], [], "unknown")


def test_permutation_problem_infers_domain_from_instance():
    definition = make_problem(
        lambda decision, data: float(np.sum(decision)), problem_type="permutation"
    )
    problems, data, _ = definition([SimpleNamespace(N=2, Gmax=1)], [4], "construct")
    np.testing.assert_array_equal(problems[0].bound[0], [1, 2, 3, 4])
    assert data == [4]


def _valid_problem(**overrides):
    values = dict(
        type=["continuous", "static", "certain"],
        bound=np.array([[-1.0], [1.0]]),
        evaluate=lambda data, decision: (0.0, 0.0, None),
        N=2,
        Gmax=3,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("problems", "data", "error", "message"),
    [
        ([], [], ValueError, "no problem"),
        ([_valid_problem()], [], ValueError, "data entries"),
        ([_valid_problem(type="continuous")], [None], TypeError, "non-empty sequence"),
        ([_valid_problem(type=["graph"])], [None], ValueError, "unsupported type"),
        ([_valid_problem(bound=[-1, 1])], [None], ValueError, "shape"),
        ([_valid_problem(bound=[[-np.inf], [1]])], [None], ValueError, "finite"),
        ([_valid_problem(evaluate=None)], [None], TypeError, "must expose evaluate"),
        ([_valid_problem(N=0)], [None], ValueError, "attribute N"),
        ([_valid_problem(Gmax=0)], [None], ValueError, "attribute Gmax"),
    ],
)
def test_constructed_problem_validation_errors(problems, data, error, message):
    with pytest.raises(error, match=message):
        validate_constructed_problems(problems, data)
