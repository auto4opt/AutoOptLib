"""Direct contracts for shared design helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from autooptlib.utils.design._helpers import (
    as_list,
    ceil_divide,
    copy_if_array,
    create_zero_matrix,
    deep_copy_operator_matrix,
    ensure_aux_list,
    ensure_para_slot,
    ensure_rng,
    get_behavior,
    get_flex,
    get_problem_type,
    has_global_behavior,
    has_local_behavior,
    inclusive_range,
    problem_list,
    reinit_parameters,
    sample_without,
    set_behavior,
    to_numpy,
)


def test_flexible_lookup_handles_dicts_acronyms_defaults_and_required():
    assert get_flex({"value": 3}, "value") == 3
    assert get_flex(SimpleNamespace(ProbFE=9), "prob_fe") == 9
    assert get_flex(None, "x", 4) == 4
    with pytest.raises(AttributeError, match="None"):
        get_flex(None, "x", required=True)
    with pytest.raises(AttributeError, match="not found"):
        get_flex(SimpleNamespace(), "x", required=True)

    class Slotted:
        __slots__ = ()

    assert get_flex(Slotted(), "x", 5) == 5


def test_rng_and_array_helpers_cover_supported_inputs():
    generator = np.random.default_rng(1)
    assert ensure_rng(SimpleNamespace(rng=generator)) is generator
    assert isinstance(ensure_rng(SimpleNamespace(rng=2)), np.random.Generator)
    assert ensure_rng(SimpleNamespace(random_state=generator)) is generator
    assert isinstance(ensure_rng(SimpleNamespace(random_state=2)), np.random.Generator)
    assert isinstance(ensure_rng(SimpleNamespace()), np.random.Generator)
    assert to_numpy(None) is None
    assert to_numpy([1, 2], dtype=float).dtype == float


def test_parameter_and_collection_helpers_cover_edge_cases():
    with pytest.raises(ValueError, match="cannot be None"):
        ensure_para_slot(None, 1)
    params = []
    assert ensure_para_slot(params, 2) == [None, None]
    params = [None]
    assert ensure_para_slot(params, 1) == [None, None]
    params = [(1, 2)]
    assert ensure_para_slot(params, 1) == [1, 2]

    assert problem_list(None) == []
    assert problem_list((1, 2)) == [1, 2]
    assert problem_list("x") == ["x"]
    assert get_problem_type(None) is None
    assert get_problem_type(SimpleNamespace(type="discrete")) == "discrete"
    assert get_problem_type(SimpleNamespace()) is None
    assert as_list(None) == []
    original = [1]
    assert as_list(original) is original
    assert as_list((1, 2)) == [1, 2]
    assert as_list(1) == [1]


def test_behavior_sampling_and_matrix_helpers():
    assert has_global_behavior([[None], ["GS"]])
    assert not has_global_behavior(None)
    assert has_local_behavior([["LS"]])
    assert not has_local_behavior([])
    rng = np.random.default_rng(2)
    assert sample_without([1, 2], 1, rng) == 2
    with pytest.raises(ValueError, match="No alternative"):
        sample_without([1], 1, rng)
    assert inclusive_range(3, 2) == []
    assert inclusive_range(2, 3) == [2, 3]

    matrix = np.array([[1, 2]])
    copied = deep_copy_operator_matrix(matrix)
    assert np.array_equal(copied, matrix) and copied is not matrix
    assert create_zero_matrix(2).shape == (2, 2)
    entry = []
    set_behavior(entry, "GS")
    assert get_behavior(entry) == "GS"
    assert get_behavior([]) is None


def test_numeric_and_copy_helpers():
    rng = np.random.default_rng(3)
    values = reinit_parameters(np.array([[0.0, 1.0], [2.0, 3.0]]), rng)
    assert np.all(values >= [0, 2]) and np.all(values <= [1, 3])
    assert ceil_divide(5, 0) == 0
    assert ceil_divide(5, 2) == 3
    array = np.array([1])
    assert copy_if_array(array) is not array
    assert copy_if_array((1, 2)) == [1, 2]
    marker = object()
    assert copy_if_array(marker) is marker
    assert ensure_aux_list(None, 2) == [None, None]
    assert ensure_aux_list([1], 2) == [1, None]
    assert ensure_aux_list([1, 2], 1) == [1, 2]
