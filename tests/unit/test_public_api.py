"""Tests for the public package API and bundled resources."""

from __future__ import annotations

from importlib import resources
from types import SimpleNamespace

import numpy as np
import pytest

import autooptlib
import autooptlib.problems as problems
from autooptlib import get_component, register_component


def test_public_version_matches_release():
    assert autooptlib.__version__ == "1.3.0"


def test_cec_archive_is_a_package_resource():
    archive = resources.files("autooptlib.problems.data").joinpath("cec2013.npz")
    assert archive.is_file()


@pytest.mark.parametrize("name", problems.__all__)
def test_all_cec2013_problems_construct_and_evaluate(name):
    problem_fn = getattr(problems, name)
    problem_list, data, _ = problem_fn([SimpleNamespace()], [10], "construct")
    values, constraints, _ = problem_fn(data[0], np.zeros((2, 10)), "evaluate")
    assert problem_list[0].bound.shape == (2, 10)
    assert np.asarray(values).shape == (2,)
    assert np.all(np.isfinite(values))
    assert np.all(np.asarray(constraints) == 0)


def test_cec2013_reference_optima_match_published_biases():
    with np.load(
        resources.files("autooptlib.problems.data").joinpath("cec2013.npz")
    ) as archive:
        composition_optimum = archive["shift_data"][0, :10]

    expected = list(range(-1400, -99, 100)) + list(range(100, 1401, 100))
    for index, bias in enumerate(expected, start=1):
        problem_fn = getattr(problems, f"cec2013_f{index}")
        _, data, _ = problem_fn([SimpleNamespace()], [10], "construct")
        optimum = data[0].o if index <= 20 else composition_optimum
        values, _, _ = problem_fn(
            data[0], np.asarray(optimum).reshape(1, -1), "evaluate"
        )
        assert float(np.asarray(values)[0]) == pytest.approx(bias, abs=2e-4)


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        (
            0.0,
            [
                17398.270025643684,
                2396412610.9019618,
                7.2542451564562992e20,
                75132346.84986454,
                40434.08125354802,
                961.2132235027589,
                62885586.662445866,
                -678.0156101069512,
                -579.7523754268578,
                2958.0111652935962,
                -68.85490363852517,
                24.409324082253647,
                158.00167500061042,
                4523.575143387677,
                3075.1654636826634,
                217.50478678005405,
                509.5833597461297,
                645.0303148911825,
                113720.48150316134,
                605.0,
                1689.8570200417998,
                5442.9812724881785,
                4297.650206927683,
                1579.9075365188896,
                1415.6995850587002,
                9036.72162529505,
                2330.500864913557,
                3009.2459654501627,
            ],
        ),
        (
            1.0,
            [
                17297.327650567775,
                2369973380.5645466,
                6.674683402465664e20,
                64674277.3745981,
                39204.023022324676,
                862.838458684628,
                67319103.6960722,
                -678.1139448621506,
                -580.5125515533873,
                2929.4272910000964,
                -53.09494281653241,
                25.88382972532719,
                158.27365776223303,
                4235.953246789083,
                3042.7039280101953,
                209.32994533227844,
                590.765721405681,
                625.822891727017,
                123842.75727931673,
                605.0,
                1692.318552845361,
                5261.297868191961,
                4287.962546015057,
                1581.1037754768658,
                1420.204178402104,
                9132.718679394688,
                2322.6991067044996,
                3030.563592994674,
            ],
        ),
    ],
)
def test_cec2013_matches_official_c_reference_at_fixed_points(point, expected):
    """Values generated with the CEC 2013 authors' 27-Jan-2013 C reference."""
    actual = []
    decisions = np.full((1, 10), point)
    for index in range(1, 29):
        problem_fn = getattr(problems, f"cec2013_f{index}")
        _, data, _ = problem_fn([SimpleNamespace()], [10], "construct")
        values, _, _ = problem_fn(data[0], decisions, "evaluate")
        actual.append(float(np.asarray(values)[0]))
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-7)


def test_custom_component_registration_without_internal_edits():
    def search_test_extension(*args):
        mode = args[-1]
        if mode == "execute":
            parent = args[0]
            decisions = (
                parent.decs() if callable(getattr(parent, "decs", None)) else parent
            )
            return decisions, args[3] if len(args) > 3 else None
        if mode in {"parameter", "behavior"}:
            return None, None
        raise ValueError(mode)

    register_component(
        "search_test_extension",
        search_test_extension,
        category="search",
        problem_types=["continuous"],
    )
    assert get_component("search_test_extension") is search_test_extension
    with pytest.raises(ValueError, match="already registered"):
        register_component("search_test_extension", search_test_extension)

    from autooptlib.utils.space import space

    problem = SimpleNamespace(type=["continuous", "static", "certain"])
    design_space = space([problem], SimpleNamespace())
    assert "search_test_extension" in design_space.AllOp
