"""Tests for the public application reference models."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autooptlib import (
    autoopt,
    generate_ris_instance,
    generate_stacking_instance,
    load_algorithm,
    load_ris_matlab,
    make_material_stacking_problem,
    make_ris_beamforming_problem,
)
from autooptlib.applications import (
    StackingWeights,
    reciprocal_sum_rate,
    stacking_constraints,
    stacking_objective,
)


def test_ris_generator_and_objective_are_deterministic():
    left = generate_ris_instance(6, seed=14, n_antennas=2, n_users=2)
    right = generate_ris_instance(6, seed=14, n_antennas=2, n_users=2)
    np.testing.assert_array_equal(left.bs_to_ris, right.bs_to_ris)
    phases = np.arange(6) % (2**left.phase_bits)
    assert reciprocal_sum_rate(phases, left) == pytest.approx(
        reciprocal_sum_rate(phases, right)
    )
    assert np.isfinite(reciprocal_sum_rate(phases, left))


def test_ris_validation_and_factory_errors():
    with pytest.raises(ValueError, match="positive"):
        generate_ris_instance(0, seed=1)
    valid = generate_ris_instance(4, seed=2, n_antennas=2, n_users=2)
    with pytest.raises(ValueError, match="Incompatible"):
        replace(valid, direct_channels=np.ones((2, 3))).validate()
    with pytest.raises(ValueError, match="phase_bits"):
        replace(valid, phase_bits=0).validate()
    with pytest.raises(ValueError, match="user_weights"):
        replace(valid, user_weights=np.ones(3)).validate()
    with pytest.raises(ValueError, match="Expected"):
        reciprocal_sum_rate(np.zeros(3), valid)
    with pytest.raises(ValueError, match="must be in"):
        reciprocal_sum_rate(np.full(4, 99), valid)
    with pytest.raises(ValueError, match="At least one"):
        make_ris_beamforming_problem({})

    problem = make_ris_beamforming_problem({"known": valid})
    with pytest.raises(KeyError, match="Unknown RIS"):
        problem([SimpleNamespace()], ["missing"], "construct")


def test_ris_problem_runs_through_solve(tmp_path):
    instance = generate_ris_instance(5, seed=3, n_antennas=2, n_users=2)
    problem = make_ris_beamforming_problem({"small": instance})
    best, histories = autoopt(
        Mode="solve",
        Problem=problem,
        InstanceSolve=["small"],
        AlgName="Discrete Random Search",
        ProbN=4,
        ProbFE=8,
        Seed=4,
        OutputDir=tmp_path,
    )
    assert np.isfinite(best[0][0].fit)
    assert histories[0]


def test_stacking_model_objective_constraints_and_validation():
    instance = generate_stacking_instance(
        6, seed=7, n_levels=3, n_positions=4, rack_width=20
    )
    decision = np.concatenate((np.arange(6) % 3, np.arange(6) % 4, np.arange(6) % 2))
    assert np.isfinite(stacking_objective(decision, instance))
    assert stacking_constraints(decision, instance).shape == (6,)
    with pytest.raises(ValueError, match="Expected"):
        stacking_objective(decision[:-1], instance)
    invalid = decision.copy()
    invalid[0] = 99
    with pytest.raises(ValueError, match="outside"):
        stacking_constraints(invalid, instance)
    with pytest.raises(ValueError, match="positive"):
        generate_stacking_instance(0, seed=1)
    with pytest.raises(ValueError, match="non-negative"):
        StackingWeights(heavy_low=-1).validate()
    with pytest.raises(ValueError, match="At least one"):
        make_material_stacking_problem({})


def test_stacking_problem_runs_through_solve(tmp_path):
    instance = generate_stacking_instance(
        5,
        seed=8,
        n_levels=3,
        n_positions=4,
        rack_width=20,
        max_level_weight=100,
    )
    problem = make_material_stacking_problem({"small": instance})
    records, data, _ = problem([SimpleNamespace(N=4, Gmax=2)], ["small"], "construct")
    assert records[0].bound.shape == (2, 15)
    assert data[0] is instance

    best, histories = autoopt(
        Mode="solve",
        Problem=problem,
        InstanceSolve=["small"],
        AlgName="Discrete Random Search",
        ProbN=4,
        ProbFE=8,
        Seed=9,
        OutputDir=tmp_path,
    )
    assert np.isfinite(best[0][0].fit)
    assert histories[0]


@pytest.mark.parametrize(
    ("filename", "choose", "primary", "secondary", "parameter"),
    [
        (
            "material_stacking_alg_star.json",
            "choose_roulette_wheel",
            "cross_point_one",
            "search_reset_rand",
            0.1342,
        ),
        (
            "ris_beamforming_alg_star.json",
            "choose_nich",
            "cross_point_uniform",
            "search_reset_one",
            0.1229,
        ),
    ],
)
def test_paper_application_algorithm_json_is_valid(
    filename, choose, primary, secondary, parameter
):
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "applications"
        / "algorithms"
        / filename
    )
    algorithm = load_algorithm(path)
    pathway = algorithm.operator_pheno[0][0]
    params = algorithm.parameter_pheno[0][0].search[0]
    assert pathway.choose == choose
    assert pathway.search[0].primary == primary
    assert pathway.search[0].secondary == secondary
    values = [params.primary, params.secondary]
    assert any(
        value is not None and value[0] == pytest.approx(parameter) for value in values
    )


def test_optional_matlab_ris_loader(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    record = {
        "G": np.ones((3, 2), dtype=complex),
        "Hd": np.ones((2, 2), dtype=complex),
        "Hr": np.ones((2, 3), dtype=complex),
        "b": 2,
        "PT": 1.5,
        "omega": np.array([1.0, 2.0]),
    }
    source = tmp_path / "Beanforming.mat"
    scipy_io.savemat(source, {"Data": np.asarray([record], dtype=object)})
    instances = load_ris_matlab(source)
    assert list(instances) == [1]
    assert instances[1].n_elements == 3
    np.testing.assert_array_equal(instances[1].user_weights, [1.0, 2.0])
    with pytest.raises(FileNotFoundError):
        load_ris_matlab(tmp_path / "missing.mat")
