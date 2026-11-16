"""Validation and compatibility tests for the portable algorithm schema."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from autooptlib.serialization import (
    _decode_value,
    _encode_value,
    algorithm_from_dict,
    algorithm_to_dict,
    load_algorithm,
)
from autooptlib.utils.design import Design
from autooptlib.utils.solve import input_algorithm


def _document():
    algorithm, _ = input_algorithm(
        SimpleNamespace(AlgName="Continuous Genetic Algorithm")
    )
    return algorithm_to_dict(
        algorithm,
        metadata={"special": [np.nan, np.inf, -np.inf], "count": np.int64(2)},
    )


def test_special_json_values_round_trip_and_reject_unknown_types():
    encoded = _encode_value([np.nan, np.inf, -np.inf, np.int64(2)])
    decoded = _decode_value(encoded)
    assert np.isnan(decoded[0])
    assert decoded[1:] == [np.inf, -np.inf, 2]
    assert _decode_value({"nested": {"$float": "inf"}}) == {"nested": np.inf}
    with pytest.raises(ValueError, match="Unknown encoded float"):
        _decode_value({"$float": "other"})
    with pytest.raises(TypeError, match="not JSON serializable"):
        _encode_value(object())


def test_export_requires_decoded_matching_pathways():
    with pytest.raises(ValueError, match="decoded"):
        algorithm_to_dict(Design())
    algorithm, _ = input_algorithm(SimpleNamespace(AlgName="Continuous Random Search"))
    algorithm.parameter_pheno[0].append(algorithm.parameter_pheno[0][0])
    with pytest.raises(ValueError, match="pathway counts"):
        algorithm_to_dict(algorithm)

    algorithm, _ = input_algorithm(SimpleNamespace(AlgName="Continuous Random Search"))
    algorithm.parameter_pheno[0][0].search.clear()
    with pytest.raises(ValueError, match="Search-step"):
        algorithm_to_dict(algorithm)


@pytest.mark.parametrize(
    ("mutate", "error", "message"),
    [
        (lambda d: d.update(schema="wrong"), ValueError, "Expected schema"),
        (lambda d: d.update(pathways=[]), ValueError, "at least one pathway"),
        (lambda d: d.update(pathways=[1]), TypeError, "must be an object"),
        (
            lambda d: d["pathways"][0].update(choose=None),
            TypeError,
            "string choose",
        ),
        (
            lambda d: d["pathways"][0].update(search=[]),
            ValueError,
            "at least one search",
        ),
        (
            lambda d: d["pathways"][0].update(search=[1]),
            TypeError,
            "must be an object",
        ),
        (
            lambda d: d["pathways"][0]["search"][0].update(primary=None),
            TypeError,
            "string primary",
        ),
        (
            lambda d: d["pathways"][0]["search"][0].update(secondary=1),
            TypeError,
            "Secondary component",
        ),
        (
            lambda d: d["pathways"][0]["search"][0].update(termination=None),
            ValueError,
            "termination vector",
        ),
        (
            lambda d: d["pathways"][0].update(archive="archive_best"),
            TypeError,
            "archive must be",
        ),
    ],
)
def test_schema_validation_rejects_malformed_documents(mutate, error, message):
    document = deepcopy(_document())
    mutate(document)
    with pytest.raises(error, match=message):
        algorithm_from_dict(document)


def test_schema_load_rejects_non_object_root(tmp_path):
    path = tmp_path / "list.json"
    path.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(TypeError, match="root must be an object"):
        load_algorithm(path)


def test_schema_reconstructs_secondary_components_and_parameters():
    restored = algorithm_from_dict(_document())
    step = restored.operator_pheno[0][0].search[0]
    params = restored.parameter_pheno[0][0].search[0]
    assert step.secondary == "search_mu_polynomial"
    assert params.primary is not None
    assert params.secondary is not None


def test_algorithm_configuration_round_trips_and_is_optional():
    document = _document()
    document["configuration"] = {
        "population_size": 30,
        "offspring_size": 80,
    }
    restored = algorithm_from_dict(document)
    assert restored.configuration == {
        "population_size": 30,
        "offspring_size": 80,
        "crossover_rate": 1.0,
        "mutation_rate": 1.0,
        "boundary_handling": "clip",
    }
    assert algorithm_to_dict(restored)["configuration"] == restored.configuration

    document.pop("configuration")
    legacy = algorithm_from_dict(document)
    assert legacy.configuration is None


def test_algorithm_configuration_canonicalizes_only_inactive_rates():
    active_document = _document()
    active_document["configuration"] = {
        "population_size": 30,
        "offspring_size": 30,
        "crossover_rate": 0.2,
        "mutation_rate": 0.3,
    }
    active = algorithm_from_dict(active_document)
    assert active.configuration["crossover_rate"] == pytest.approx(0.2)
    assert active.configuration["mutation_rate"] == pytest.approx(0.3)

    inactive_document = deepcopy(active_document)
    search = inactive_document["pathways"][0]["search"][0]
    search.update(
        primary="reinit_continuous",
        secondary=None,
        primary_parameter=None,
        secondary_parameter=None,
    )
    inactive = algorithm_from_dict(inactive_document)
    assert inactive.configuration["crossover_rate"] == pytest.approx(1.0)
    assert inactive.configuration["mutation_rate"] == pytest.approx(1.0)
    assert algorithm_to_dict(inactive)["configuration"] == inactive.configuration


def test_algorithm_configuration_is_the_only_source_for_scalar_views():
    from copy import deepcopy

    from autooptlib.utils.design import Design
    from autooptlib.utils.design._population import set_configuration

    algorithm = Design()
    set_configuration(
        algorithm,
        {
            "population_size": 4,
            "offspring_size": 2,
            "crossover_rate": 0.2,
            "mutation_rate": 0.3,
            "boundary_handling": "clip",
        },
    )
    original = deepcopy(algorithm)
    algorithm.configuration["population_size"] = 8
    algorithm.configuration["boundary_handling"] = "reflect"
    assert algorithm.population_size == 8
    assert algorithm.boundary_handling == "reflect"
    assert original.population_size == 4
    assert original.boundary_handling == "clip"
    assert "population_size" not in vars(algorithm)
