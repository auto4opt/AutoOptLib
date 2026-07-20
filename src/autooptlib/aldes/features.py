"""Problem-feature extraction for continual ALDes training."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class PBOFeatureResult:
    """A reproducible PBO feature vector and the samples that produced it."""

    features: np.ndarray
    feature_names: tuple[str, ...]
    samples: np.ndarray
    initial_populations: np.ndarray


def _binary_random_walk(
    dimension: int, length: int, rng: np.random.Generator
) -> np.ndarray:
    if dimension <= 0 or length <= 0:
        raise ValueError("dimension and random-walk length must be positive.")
    current = rng.integers(0, 2, size=dimension, dtype=np.int8)
    sample = np.empty((length, dimension), dtype=np.int8)
    for row in range(length):
        sample[row] = current
        column = int(rng.integers(0, dimension))
        current = current.copy()
        current[column] = 1 - current[column]
    return sample


def _feature_mapping(
    decisions: np.ndarray, objectives: np.ndarray, *, seed: int
) -> dict[str, float]:
    try:
        import pandas as pd
        from pflacco.classical_ela_features import (
            calculate_dispersion,
            calculate_ela_meta,
            calculate_information_content,
            calculate_nbc,
        )
    except ImportError as exc:  # pragma: no cover - dependency specific
        raise ImportError(
            "Continual ALDes feature extraction requires pandas and pflacco; "
            "install AutoOptLib with `pip install 'autooptlib[aldes]'`."
        ) from exc

    frame = pd.DataFrame(decisions)
    values: dict[str, Any] = {}
    # Degenerate neighborhoods are expected on discrete random walks. pflacco
    # may emit divide-by-zero warnings for those intermediate ratios; their
    # non-finite outputs are handled explicitly during cross-trial averaging.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        values.update(calculate_information_content(frame, objectives, seed=seed))
        values.update(calculate_ela_meta(frame, objectives))
        values.update(calculate_nbc(frame, objectives))
        values.update(
            calculate_dispersion(frame, objectives, dist_method="hamming")
        )
    result: dict[str, float] = {}
    for name, value in values.items():
        # Runtime measurements are machine-dependent and are not landscape
        # descriptors. Excluding them also makes saved features portable.
        if str(name).endswith("costs_runtime"):
            continue
        try:
            result[str(name)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def extract_pbo_features(
    problem_id: int,
    *,
    instance: int = 1,
    dimension: int = 100,
    trials: int = 5,
    sample_factor: int = 100,
    feature_dim: int = 32,
    population_size: int = 50,
    seed: int | None = None,
) -> PBOFeatureResult:
    """Extract the paper-style random-walk PBO features for continual ALDes.

    Every trial uses a binary random walk of length ``sample_factor *
    dimension``. Numeric, non-runtime pflacco features are averaged across
    trials. Names are sorted before the first ``feature_dim`` values are kept,
    giving a stable schema across runs and library dictionary ordering.
    """

    if trials <= 0 or sample_factor <= 0 or feature_dim <= 0:
        raise ValueError("trials, sample_factor, and feature_dim must be positive.")
    if population_size <= 0:
        raise ValueError("population_size must be positive.")
    try:
        import ioh
    except ImportError as exc:  # pragma: no cover - dependency specific
        raise ImportError(
            "PBO feature extraction requires IOH; install AutoOptLib with "
            "`pip install 'autooptlib[aldes]'`."
        ) from exc

    problem = ioh.get_problem(
        int(problem_id),
        instance=int(instance),
        dimension=int(dimension),
        problem_class=ioh.ProblemClass.PBO,
    )
    root = np.random.default_rng(seed)
    mappings: list[dict[str, float]] = []
    samples: list[np.ndarray] = []
    for _ in range(trials):
        trial_seed = int(root.integers(0, np.iinfo(np.int32).max))
        trial_rng = np.random.default_rng(trial_seed)
        decisions = _binary_random_walk(
            dimension, sample_factor * dimension, trial_rng
        )
        objectives = np.asarray(problem(decisions), dtype=float).reshape(-1)
        mappings.append(_feature_mapping(decisions, objectives, seed=trial_seed))
        samples.append(decisions)

    names = sorted({name for mapping in mappings for name in mapping})
    matrix = np.full((trials, len(names)), np.nan, dtype=float)
    for row, mapping in enumerate(mappings):
        for column, name in enumerate(names):
            matrix[row, column] = mapping.get(name, np.nan)
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=0)
    totals = np.where(finite, matrix, 0.0).sum(axis=0)
    averaged = np.divide(
        totals,
        counts,
        out=np.zeros_like(totals),
        where=counts > 0,
    )

    if len(names) < feature_dim:
        padding = feature_dim - len(names)
        names.extend(f"padding.{index}" for index in range(padding))
        averaged = np.pad(averaged, (0, padding))
    selected_names = tuple(names[:feature_dim])
    selected = np.asarray(averaged[:feature_dim], dtype=np.float32)
    sample_array = np.stack(samples)
    if sample_array.shape[1] < population_size:
        raise ValueError("The feature sample is smaller than population_size.")
    population_indices = np.linspace(
        0, sample_array.shape[1] - 1, population_size, dtype=int
    )
    initial = np.array(sample_array[:, population_indices, :], copy=True)
    return PBOFeatureResult(selected, selected_names, sample_array, initial)


def standardize_features(
    results: Sequence[PBOFeatureResult],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize continual-task features without leaking runtime metadata."""

    if not results:
        raise ValueError("At least one feature result is required.")
    names = results[0].feature_names
    if any(result.feature_names != names for result in results[1:]):
        raise ValueError("All feature results must use the same feature schema.")
    matrix = np.vstack([result.features for result in results]).astype(float)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    return ((matrix - mean) / scale).astype(np.float32), mean, scale


__all__ = [
    "PBOFeatureResult",
    "extract_pbo_features",
    "standardize_features",
]
