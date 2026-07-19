"""Portable, versioned serialization for designed algorithms."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ._version import __version__
from .components import get_component
from .utils.design._helpers import Pathway, PathwayParam, SearchParam, SearchStep

SCHEMA_NAME = "autooptlib.algorithm"
SCHEMA_VERSION = 1


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

    records = []
    for pathway, params in zip(pathways, pathway_parameters):
        searches = []
        if len(pathway.search) != len(params.search):
            raise ValueError("Search-step and parameter counts do not match.")
        for step, step_params in zip(pathway.search, params.search):
            searches.append(
                {
                    "primary": step.primary,
                    "secondary": step.secondary,
                    "termination": _encode_value(
                        np.asarray(step.termination, dtype=float)
                    ),
                    "primary_parameter": _encode_value(step_params.primary),
                    "secondary_parameter": _encode_value(step_params.secondary),
                }
            )
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

    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "autooptlib_version": __version__,
        "metadata": _encode_value(dict(metadata or {})),
        "pathways": records,
    }


def algorithm_from_dict(document: Mapping[str, Any]):
    """Validate and construct a ``Design`` from the stable JSON schema."""
    if document.get("schema") != SCHEMA_NAME:
        raise ValueError(f"Expected schema {SCHEMA_NAME!r}.")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported algorithm schema version {document.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}."
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
        if not isinstance(choose, str) or not isinstance(update, str):
            raise TypeError(
                f"Pathway {index} must define string choose and update names."
            )
        get_component(choose)
        get_component(update)

        steps = []
        step_parameters = []
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
            get_component(primary)
            if secondary:
                get_component(secondary)
            termination = _array_or_none(step_record.get("termination"))
            if termination is None or termination.size == 0:
                raise ValueError("Search steps require a non-empty termination vector.")
            steps.append(SearchStep(primary, termination.reshape(-1), secondary))
            step_parameters.append(
                SearchParam(
                    _array_or_none(step_record.get("primary_parameter")),
                    _array_or_none(step_record.get("secondary_parameter")),
                )
            )

        archive = record.get("archive", [])
        if not isinstance(archive, list) or not all(
            isinstance(name, str) for name in archive
        ):
            raise TypeError("archive must be a list of component names.")
        for name in archive:
            get_component(name)
        pathways.append(Pathway(choose, steps, update, archive))
        pathway_parameters.append(
            PathwayParam(
                _array_or_none(record.get("choose_parameter")),
                step_parameters,
                _array_or_none(record.get("update_parameter")),
            )
        )

    from .utils.design import Design

    design = Design()
    design.construct([pathways], [pathway_parameters])
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
