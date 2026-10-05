"""Ordered portable graph identity for held-out selection (not training caches)."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from autooptlib.components import component_category

STRUCTURE_KEY_REVISION = "search-v6-ordered-path-components-v1"


def _component(name: Any, category: str, *, optional: bool = True) -> None:
    if name is None and optional:
        return
    if not isinstance(name, str) or component_category(name) != category:
        raise ValueError(f"Invalid {category} component: {name!r}")


def selection_structure_payload(document: Mapping[str, Any]) -> list[dict]:
    """Keep Search's ordered components; ignore every numeric/configuration field."""
    paths = document.get("pathways")
    if not isinstance(paths, list) or not paths:
        raise ValueError("Selection requires nonempty serialized pathways")
    result = []
    for pathway in paths:
        if not isinstance(pathway, Mapping):
            raise ValueError("Pathway must be a mapping")
        searches = pathway.get("search")
        if not isinstance(searches, list) or not searches:
            raise ValueError("Selection requires nonempty search steps")
        _component(pathway.get("choose"), "choose")
        _component(pathway.get("update"), "update", optional=False)
        archive = pathway.get("archive", [])
        if not isinstance(archive, list):
            raise ValueError("Archive must be an ordered list")
        for name in archive:
            _component(name, "archive", optional=False)
        steps = []
        for step in searches:
            if not isinstance(step, Mapping):
                raise ValueError("Search step must be a mapping")
            _component(step.get("choose"), "choose")
            _component(step.get("primary"), "search", optional=False)
            _component(step.get("secondary"), "search")
            steps.append({k: step.get(k) for k in ("choose", "primary", "secondary")})
        result.append(
            {
                "choose": pathway.get("choose"),
                "search": steps,
                "update": pathway.get("update"),
                "archive": list(archive),
            }
        )
    return result


def selection_structure_key(document: Mapping[str, Any]) -> str:
    payload = selection_structure_payload(document)
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class SelectionPlan:
    rows: tuple[Mapping[str, Any], ...]
    structure_keys: tuple[str, ...]
    quota_count: int
    fallback_count: int
    structure_key_revision: str = STRUCTURE_KEY_REVISION


def select_validation_candidates(
    ordered_rows: Sequence[Mapping[str, Any]],
    documents: Mapping[str, Mapping[str, Any]],
    *,
    training_pool: int = 100,
    elite_count: int = 20,
    max_per_structure: int = 2,
) -> SelectionPlan:
    """Rank, apply quota, then fill only from the same frozen training pool."""
    if not 0 < elite_count <= training_pool or max_per_structure <= 0:
        raise ValueError("Invalid selection quota or pool size")
    ids = [str(r["algorithm_identity"]) for r in ordered_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate full algorithm identity")

    def rank(row: Mapping[str, Any]) -> tuple[float, int]:
        cost = float(row.get("cost", row.get("training_cost")))
        index = row.get("discovery_index", row.get("proposal", row.get("candidate")))
        if not math.isfinite(cost) or index is None:
            raise ValueError("Missing discovery index or nonfinite training score")
        return cost, int(index)

    pool = sorted(ordered_rows, key=rank)[:training_pool]
    if len(pool) < elite_count:
        raise ValueError("Fewer complete candidates than validation slots")
    keys = {
        str(r["algorithm_identity"]): selection_structure_key(
            documents[str(r["algorithm_identity"])]
        )
        for r in pool
    }
    selected = []
    counts: dict[str, int] = {}
    for row in pool:
        key = keys[str(row["algorithm_identity"])]
        if counts.get(key, 0) < max_per_structure:
            selected.append(row)
            counts[key] = counts.get(key, 0) + 1
        if len(selected) == elite_count:
            break
    quota_count = len(selected)
    selected_ids = {str(r["algorithm_identity"]) for r in selected}
    for row in pool:
        if len(selected) == elite_count:
            break
        if str(row["algorithm_identity"]) not in selected_ids:
            selected.append(row)
            selected_ids.add(str(row["algorithm_identity"]))
    return SelectionPlan(
        tuple(selected),
        tuple(keys[str(r["algorithm_identity"])] for r in selected),
        quota_count,
        len(selected) - quota_count,
    )
