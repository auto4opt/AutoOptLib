"""Shared utilities for the design module."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


def _name_variants(name: str) -> list[str]:
    # Attribute lookup order matters when a compatibility namespace exposes
    # both Python-style and MATLAB-style spellings.  Keep the caller's exact
    # spelling first and de-duplicate without using an unordered set.
    variants = [name, name.lower(), name.upper()]
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    variants.append(snake)
    if "_" in name:
        parts = name.split("_")
        camel = parts[0].lower() + "".join(p.title() for p in parts[1:])
        pascal = "".join(p.title() for p in parts)
        variants.extend([camel, pascal])
    else:
        parts = re.findall(r"[A-Za-z][^A-Z]*", name)
        if parts:
            joined = "_".join(part.lower() for part in parts)
            variants.append(joined)
            camel = parts[0].lower() + "".join(p.title() for p in parts[1:])
            pascal = "".join(p.title() for p in parts)
            variants.extend([camel, pascal])
    return list(dict.fromkeys(v for v in variants if v))


def get_flex(
    obj: Any, name: str, default: Any | None = None, *, required: bool = False
) -> Any:
    if obj is None:
        if required:
            raise AttributeError(f"Cannot read {name!r} from None")
        return default
    for candidate in _name_variants(name):
        if isinstance(obj, dict) and candidate in obj:
            return obj[candidate]
        if hasattr(obj, candidate):
            return getattr(obj, candidate)
    # Acronym-heavy legacy names such as ``ProbFE`` cannot be recovered by a
    # generic camel-case conversion of ``prob_fe``.  Compare a punctuation-free
    # form as a final compatibility step.
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    try:
        attributes = obj.keys() if isinstance(obj, dict) else vars(obj).keys()
    except TypeError:
        attributes = ()
    for candidate in attributes:
        if re.sub(r"[^a-z0-9]", "", str(candidate).lower()) == normalized:
            return obj[candidate] if isinstance(obj, dict) else getattr(obj, candidate)
    if required:
        raise AttributeError(f"Attribute {name!r} not found on {obj!r}")
    return default


def ensure_rng(setting: Any) -> np.random.Generator:
    rng = get_flex(setting, "rng", None)
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    random_state = get_flex(setting, "random_state", None)
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, (int, np.integer)):
        return np.random.default_rng(int(random_state))
    return np.random.default_rng()


def to_numpy(array_like: Any, *, dtype: Optional[type] = None) -> np.ndarray | None:
    if array_like is None:
        return None
    return np.asarray(array_like, dtype=dtype)


def ensure_para_slot(paras: list[Any] | None, index: int) -> list[Any]:
    if paras is None:
        raise ValueError("Parameter container cannot be None")
    while len(paras) < index:
        paras.append([None, None])
    entry = paras[index - 1]
    if entry is None:
        entry = [None, None]
        paras[index - 1] = entry
    elif isinstance(entry, tuple):
        entry = list(entry)
        paras[index - 1] = entry
    return entry


def problem_list(problem: Any) -> list[Any]:
    if problem is None:
        return []
    if isinstance(problem, Sequence) and not isinstance(problem, (str, bytes)):
        return list(problem)
    return [problem]


def get_problem_type(problem: Any) -> str | None:
    problems = problem_list(problem)
    if not problems:
        return None
    first = problems[0]
    type_field = get_flex(first, "type", None)
    if isinstance(type_field, Sequence) and not isinstance(type_field, (str, bytes)):
        return str(type_field[0])
    if type_field is not None:
        return str(type_field)
    return None


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def has_global_behavior(entry: Any) -> bool:
    try:
        return bool(entry[1][0])
    except (TypeError, IndexError):
        return False


def has_local_behavior(entry: Any) -> bool:
    try:
        return bool(entry[0][0])
    except (TypeError, IndexError):
        return False


def sample_without(pool: Sequence[int], value: int, rng: np.random.Generator) -> int:
    candidates = [v for v in pool if v != value]
    if not candidates:
        raise ValueError("No alternative available to sample")
    return int(rng.choice(candidates))


def inclusive_range(start: int, end: int) -> list[int]:
    if end < start:
        return []
    return list(range(start, end + 1))


def deep_copy_operator_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.array(matrix, copy=True)


def create_zero_matrix(rows: int) -> np.ndarray:
    return np.zeros((rows, 2), dtype=int)


def set_behavior(entry: list[Any], value: str | None) -> None:
    if len(entry) < 2:
        while len(entry) < 2:
            entry.append(None)
    entry[1] = value


def get_behavior(entry: list[Any]) -> str | None:
    if not entry or len(entry) < 2:
        return None
    return entry[1]


def reinit_parameters(space: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lower = space[:, 0]
    upper = space[:, 1]
    return lower + (upper - lower) * rng.random(lower.shape)


def ceil_divide(numerator: float, denominator: float) -> int:
    if denominator == 0:
        return 0
    return int(math.ceil(numerator / denominator))


def copy_if_array(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return value


def ensure_aux_list(aux: Any, length: int) -> list[Any]:
    if aux is None:
        return [None] * length
    aux_list = list(aux)
    if len(aux_list) < length:
        aux_list.extend([None] * (length - len(aux_list)))
    return aux_list


@dataclass
class SearchStep:
    primary: str
    termination: np.ndarray
    secondary: Optional[str] = None


@dataclass
class SearchParam:
    primary: Optional[np.ndarray]
    secondary: Optional[np.ndarray] = None


@dataclass
class Pathway:
    choose: str
    search: list[SearchStep]
    update: str
    archive: list[str]


@dataclass
class PathwayParam:
    choose: Optional[np.ndarray]
    search: list[SearchParam]
    update: Optional[np.ndarray]
