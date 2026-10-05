"""Append-only, portable records for every evaluated algorithm candidate."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ...serialization import algorithm_to_dict


def _finite(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def candidate_record(candidate: Any, index: int, designer: str) -> dict[str, Any]:
    metadata = dict(getattr(candidate, "metadata", {}) or {})
    evaluation_ledger = list(metadata.get("evaluation_ledger", []))
    document = algorithm_to_dict(candidate)
    performance = np.asarray(getattr(candidate, "performance", []), dtype=float)
    ledger_values = np.asarray(
        [entry.get("value", np.nan) for entry in evaluation_ledger], dtype=float
    )
    registered_score = metadata.get("training_score")
    if ledger_values.size:
        valid = bool(np.all(np.isfinite(ledger_values)))
        training_cost = float(np.mean(ledger_values)) if valid else None
    else:
        valid = bool(performance.size and np.all(np.isfinite(performance)))
        training_cost = float(np.mean(performance)) if valid else None
    if registered_score is not None:
        try:
            score = float(registered_score)
        except (TypeError, ValueError):
            score = float("nan")
        valid = valid and math.isfinite(score)
        training_cost = score if valid else None
    return {
        "schema": "autooptlib.design-candidate",
        "schema_version": 1,
        # Keep the executable algorithm schema separate from this record's
        # schema.  A pathway payload is not self-describing now that legacy
        # pathways and Search v6 stream graphs have different semantics.
        "algorithm_schema_version": document["schema_version"],
        "execution_semantics": document.get("execution_semantics"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": int(index),
        "designer": designer,
        "status": (
            "screened_out"
            if metadata.get("racing", {}).get("eliminated", False)
            else "ok"
            if valid
            else "invalid"
        ),
        "training_cost": training_cost,
        "performance": _finite(performance),
        "candidate_metadata": _finite(
            {
                key: value
                for key, value in metadata.items()
                if key != "evaluation_ledger"
            }
        ),
        "evaluation_ledger": _finite(evaluation_ledger),
        "actual_design_fes": sum(
            int(entry.get("evaluations", 0)) for entry in evaluation_ledger
        ),
        "configuration": document.get("configuration"),
        "representation": document["pathways"],
    }


def append_candidate_records(
    path: str | Path | None,
    candidates: Iterable[Any],
    *,
    start: int,
    designer: str,
) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for offset, candidate in enumerate(candidates):
            record = candidate_record(candidate, start + offset, designer)
            handle.write(
                json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def trim_candidate_records(path: str | Path | None, completed: int) -> None:
    """Atomically roll an append-only ledger back to a checkpoint boundary."""

    if path is None:
        return
    target = Path(path)
    if not target.exists():
        if completed:
            raise RuntimeError(
                f"Candidate ledger {target} is missing {completed} checkpointed records."
            )
        return
    lines = [line for line in target.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) < completed:
        raise RuntimeError(
            f"Candidate ledger {target} has {len(lines)} records but the checkpoint "
            f"requires {completed}."
        )
    for index, line in enumerate(lines[:completed], 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Candidate ledger {target} has invalid JSON at record {index}."
            ) from exc
        if int(record.get("candidate", -1)) != index:
            raise RuntimeError(
                f"Candidate ledger {target} is not contiguous at record {index}."
            )
    if len(lines) == completed:
        return
    temporary = target.with_suffix(target.suffix + ".tmp")
    text = "\n".join(lines[:completed])
    temporary.write_text(text + ("\n" if text else ""), encoding="utf-8")
    temporary.replace(target)


__all__ = [
    "append_candidate_records",
    "candidate_record",
    "trim_candidate_records",
]
