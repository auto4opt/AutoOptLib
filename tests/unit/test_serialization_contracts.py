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
