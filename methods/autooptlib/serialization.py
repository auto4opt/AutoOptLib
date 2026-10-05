"""Portable, versioned serialization for designed algorithms."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ._version import __version__
from .components import component_category, get_component
from .utils.design._helpers import Pathway, PathwayParam, SearchParam, SearchStep
from .utils.design._population import copy_configuration, set_configuration
from .utils.design._stream_graph import (
    STREAM_GRAPH_SEMANTICS,
    StreamPathway,
    StreamPathwayParam,
    StreamStage,
    StreamStageParam,
    validate_phenotype,
)

SCHEMA_NAME = "autooptlib.algorithm"
SCHEMA_VERSION = 1
STREAM_SCHEMA_VERSION = 2


def _encode_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return {"$float": "nan"}
        if math.isinf(number):
            return {"$float": "inf" if number > 0 else "-inf"}
        return number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _encode_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_encode_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _encode_value(item) for key, item in value.items()}
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


def _decode_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"$float"}:
            special = value["$float"]
            if special == "inf":
                return math.inf
            if special == "-inf":
                return -math.inf
            if special == "nan":
                return math.nan
            raise ValueError(f"Unknown encoded float {special!r}.")
        return {key: _decode_value(item) for key, item in value.items()}
    return value


def _array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(_decode_value(value), dtype=float)


def _require_component(name: str, category: str) -> None:
    get_component(name)
    actual = component_category(name)
    if actual != category:
        raise ValueError(
            f"Component {name!r} belongs to category {actual!r}, not {category!r}."
        )


def _same_optional_array(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _validate_stream_phenotype(
    pathways: list[StreamPathway], parameters: list[StreamPathwayParam]
) -> None:
    """Enforce every invariant represented by Search v6.0 schema 2."""

    validate_phenotype(pathways, parameters)
    for pathway in pathways:
        _require_component(pathway.update, "update")
        for archive in pathway.archive:
            _require_component(archive, "archive")
        for stage in pathway.stages:
            if stage.choose is not None:
                _require_component(stage.choose, "choose")
            _require_component(stage.search.primary, "search")


def algorithm_to_dict(
    design: Any, *, metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Convert a decoded ``Design`` to the stable JSON schema."""
    operator = getattr(design, "operator_pheno", None)
    parameter = getattr(design, "parameter_pheno", None)
    if not operator or not parameter or not operator[0] or not parameter[0]:
        raise ValueError("The algorithm must be decoded before it can be exported.")
    pathways = operator[0]
    pathway_parameters = parameter[0]
    if len(pathways) != len(pathway_parameters):
        raise ValueError("Operator and parameter pathway counts do not match.")

    stream_graph = isinstance(pathways[0], StreamPathway)
    if stream_graph and not all(
        isinstance(pathway, StreamPathway) for pathway in pathways
    ):
        raise TypeError(
            "Legacy and Search v6.0 pathways cannot be serialized together."
        )
    if stream_graph:
        if not all(isinstance(item, StreamPathwayParam) for item in pathway_parameters):
            raise TypeError("Search v6.0 pathways require stream parameters.")
        _validate_stream_phenotype(pathways, pathway_parameters)
    records = []
    for pathway, params in zip(pathways, pathway_parameters):
        searches = []
        if stream_graph:
            stage_pairs = [
                (stage.search, stage_params.search, stage.choose, stage_params.choose)
                for stage, stage_params in zip(pathway.stages, params.stages)
            ]
        else:
            stage_pairs = [
                (step, step_params, None, None)
                for step, step_params in zip(pathway.search, params.search)
            ]
        if len(pathway.search) != len(params.search):
            raise ValueError("Search-step and parameter counts do not match.")
        for step, step_params, stage_choose, stage_choose_param in stage_pairs:
            record = {
                "primary": step.primary,
                "secondary": step.secondary,
                "termination": _encode_value(np.asarray(step.termination, dtype=float)),
                "primary_parameter": _encode_value(step_params.primary),
                "secondary_parameter": _encode_value(step_params.secondary),
            }
            if stream_graph:
                record["choose"] = stage_choose
                record["choose_parameter"] = _encode_value(stage_choose_param)
            searches.append(record)
        records.append(
            {
                "choose": pathway.choose,
                "choose_parameter": _encode_value(params.choose),
                "search": searches,
                "update": pathway.update,
                "update_parameter": _encode_value(params.update),
                "archive": list(pathway.archive),
            }
        )

    document = {
        "schema": SCHEMA_NAME,
        "schema_version": STREAM_SCHEMA_VERSION if stream_graph else SCHEMA_VERSION,
        "autooptlib_version": __version__,
        "metadata": _encode_value(dict(metadata or {})),
        "configuration": _encode_value(copy_configuration(design))
        if (
            getattr(design, "configuration", None) is not None
            or getattr(design, "population_size", None) is not None
        )
        else None,
        "pathways": records,
    }
    if stream_graph:
        document["execution_semantics"] = STREAM_GRAPH_SEMANTICS
    return document


def algorithm_from_dict(document: Mapping[str, Any]):
    """Validate and construct a ``Design`` from the stable JSON schema."""
    if document.get("schema") != SCHEMA_NAME:
        raise ValueError(f"Expected schema {SCHEMA_NAME!r}.")
    schema_version = document.get("schema_version")
    if schema_version not in {SCHEMA_VERSION, STREAM_SCHEMA_VERSION}:
        raise ValueError(
            f"Unsupported algorithm schema version {document.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION} or {STREAM_SCHEMA_VERSION}."
        )
    stream_graph = document.get("execution_semantics") == STREAM_GRAPH_SEMANTICS
    if stream_graph != (schema_version == STREAM_SCHEMA_VERSION):
        raise ValueError(
            "Search v6.0 stream graphs require schema_version 2 and "
            "execution_semantics='stream_graph_v2'."
        )
    records = document.get("pathways")
    if not isinstance(records, list) or not records:
        raise ValueError("Algorithm document must contain at least one pathway.")

    pathways = []
    pathway_parameters = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"Pathway {index} must be an object.")
        choose = record.get("choose")
        update = record.get("update")
        if not isinstance(update, str) or (
            choose is not None and not isinstance(choose, str)
        ):
            raise TypeError(
                f"Pathway {index} must define an optional string choose and "
                "a string update name."
            )
        if not stream_graph and not isinstance(choose, str):
            raise TypeError(f"Legacy pathway {index} requires a string choose name.")
        if choose is not None:
            _require_component(choose, "choose")
        _require_component(update, "update")

        steps = []
        step_parameters = []
        stream_stages = []
        stream_stage_parameters = []
        search_records = record.get("search")
        if not isinstance(search_records, list) or not search_records:
            raise ValueError(f"Pathway {index} must contain at least one search step.")
        for step_index, step_record in enumerate(search_records):
            if not isinstance(step_record, Mapping):
                raise TypeError(
                    f"Search step {step_index} in pathway {index} must be an object."
                )
            primary = step_record.get("primary")
            secondary = step_record.get("secondary")
            if not isinstance(primary, str):
                raise TypeError("Search steps require a string primary component name.")
            if secondary is not None and not isinstance(secondary, str):
                raise TypeError("Secondary component names must be strings or null.")
            _require_component(primary, "search")
            if secondary:
                _require_component(secondary, "search")
            termination = _array_or_none(step_record.get("termination"))
            if termination is None or termination.size == 0:
                raise ValueError("Search steps require a non-empty termination vector.")
            steps.append(SearchStep(primary, termination.reshape(-1), secondary))
            search_parameter = SearchParam(
                _array_or_none(step_record.get("primary_parameter")),
                _array_or_none(step_record.get("secondary_parameter")),
            )
            step_parameters.append(search_parameter)
            if stream_graph:
                stage_choose = step_record.get("choose")
                if stage_choose is not None and not isinstance(stage_choose, str):
                    raise TypeError(
                        "Search v6.0 stage choose components must be strings or null."
                    )
                if stage_choose is not None:
                    _require_component(stage_choose, "choose")
                stream_stages.append(StreamStage(stage_choose, steps[-1]))
                stream_stage_parameters.append(
                    StreamStageParam(
                        _array_or_none(step_record.get("choose_parameter")),
                        search_parameter,
                    )
                )

        archive = record.get("archive", [])
        if not isinstance(archive, list) or not all(
            isinstance(name, str) for name in archive
        ):
            raise TypeError("archive must be a list of component names.")
        for name in archive:
            _require_component(name, "archive")
        if stream_graph:
            if choose != stream_stages[0].choose:
                raise ValueError(
                    "Search v6.0 pathway choose must match its first stage choose."
                )
            top_choose_parameter = _array_or_none(record.get("choose_parameter"))
            if not _same_optional_array(
                top_choose_parameter, stream_stage_parameters[0].choose
            ):
                raise ValueError(
                    "Search v6.0 pathway choose_parameter must match its first stage."
                )
            pathways.append(StreamPathway(stream_stages, update, archive))
            pathway_parameters.append(
                StreamPathwayParam(
                    stream_stage_parameters,
                    _array_or_none(record.get("update_parameter")),
                )
            )
        else:
            pathways.append(Pathway(choose, steps, update, archive))
            pathway_parameters.append(
                PathwayParam(
                    _array_or_none(record.get("choose_parameter")),
                    step_parameters,
                    _array_or_none(record.get("update_parameter")),
                )
            )

    if stream_graph:
        _validate_stream_phenotype(pathways, pathway_parameters)

    from .utils.design import Design

    design = Design()
    design.construct([pathways], [pathway_parameters])
    configuration = document.get("configuration")
    if configuration is not None:
        if not isinstance(configuration, Mapping):
            raise TypeError("configuration must be an object or null.")
        set_configuration(design, _decode_value(dict(configuration)))
    metadata = _decode_value(dict(document.get("metadata") or {}))
    design.metadata = metadata
    if metadata.get("designer") == "learning" and "learning_sequence" in metadata:
        design.learning_sequence = [
            int(token) for token in metadata["learning_sequence"]
        ]
    return design


def save_algorithm(
    design: Any,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write an algorithm as UTF-8 JSON and return the resolved path."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    document = algorithm_to_dict(design, metadata=metadata)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return target


def load_algorithm(path: str | Path):
    """Load and validate an algorithm JSON document."""
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, Mapping):
        raise TypeError("Algorithm JSON root must be an object.")
    return algorithm_from_dict(document)


__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "algorithm_from_dict",
    "algorithm_to_dict",
    "load_algorithm",
    "save_algorithm",
]
