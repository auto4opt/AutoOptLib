"""High-level AutoOptLib design workflow backed by an ALDes generator."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..problems.base import validate_constructed_problems
from ..utils.design import Design
from ..utils.general.process import _build_problem_struct, _normalize_setting
from ..utils.space import space
from .codec import decode_sequence


def _load_features(value: Any) -> np.ndarray:
    if hasattr(value, "features"):
        value = value.features
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"ALDes feature file not found: {path}")
        value = np.load(path, allow_pickle=False)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim not in {1, 2}:
        raise ValueError("ALDesFeatures must be a vector or a batch of vectors.")
    return array


def _load_initial_populations(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "initial_populations"):
        return value.initial_populations
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"ALDes initial-population file not found: {path}")
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                return {
                    int(name.rsplit("_", 1)[-1]): np.array(loaded[name], copy=True)
                    for name in loaded.files
                }
            finally:
                loaded.close()
        return loaded
    return value


def design_with_aldes(
    problem_descriptor: Any,
    instance_train: Sequence[Any],
    instance_test: Sequence[Any],
    *,
    setting: Any,
) -> tuple[list[Design], list[Design]]:
    """Generate, score, and test algorithms using a trained ALDes model."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency specific
        raise ImportError(
            "Designer='aldes' requires PyTorch. Install AutoOptLib with "
            "`pip install 'autooptlib[aldes]'`."
        ) from exc

    from .model import ALDesGenerator

    setting = _normalize_setting(setting)
    model_value = getattr(setting, "ALDesModel", None)
    if model_value is None:
        raise ValueError(
            "Designer='aldes' requires ALDesModel (a generator or checkpoint path)."
        )
    if isinstance(model_value, (str, Path)):
        model, _ = ALDesGenerator.load_checkpoint(model_value)
    elif isinstance(model_value, ALDesGenerator):
        model = model_value
    else:
        raise TypeError("ALDesModel must be an ALDesGenerator or checkpoint path.")

    mode = str(getattr(setting, "ALDesMode", "single")).lower()
    expects_features = bool(model.config.condition_on_features)
    if (mode == "continual") != expects_features:
        expected = "continual" if expects_features else "single"
        raise ValueError(
            f"This ALDes checkpoint was trained for {expected!r} mode; "
            f"ALDesMode={mode!r} is incompatible."
        )
    feature_value = getattr(setting, "ALDesFeatures", None)
    initial_value = getattr(setting, "ALDesInitialPopulations", None)
    if mode == "continual":
        if feature_value is None:
            raise ValueError(
                "ALDesMode='continual' requires ALDesFeatures for the target problem."
            )
        features = _load_features(feature_value)
        if initial_value is None and hasattr(feature_value, "initial_populations"):
            initial_value = feature_value
    else:
        if feature_value is not None:
            raise ValueError(
                "ALDesMode='single' does not use ALDesFeatures; omit that option."
            )
        features = None
    setting.InitialPopulations = _load_initial_populations(initial_value)
    model_device = next(model.parameters()).device
    feature_tensor = (
        torch.as_tensor(features, device=model_device) if features is not None else None
    )

    instances = list(instance_train) + list(instance_test)
    problems = _build_problem_struct(problem_descriptor, instances, setting)
    problems, data, _ = problem_descriptor(problems, instances, "construct")
    validate_constructed_problems(problems, data)
    setting = space(problems, setting)

    candidate_count = int(getattr(setting, "ALDesCandidates", 0) or setting.AlgN)
    if candidate_count < int(setting.AlgN):
        raise ValueError("ALDesCandidates must be at least AlgN.")
    temperature = float(getattr(setting, "ALDesTemperature", 1.0))
    greedy = bool(getattr(setting, "ALDesGreedy", False))
    seed = getattr(setting, "Seed", getattr(setting, "seed", None))
    torch_generator = None
    if seed is not None:
        torch_generator = torch.Generator(device=model_device)
        torch_generator.manual_seed(int(seed))
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            feature_tensor,
            candidates=candidate_count,
            temperature=temperature,
            greedy=greedy,
            generator=torch_generator,
        )

    rng = np.random.default_rng(seed)
    setting.rng = rng
    train_indices = rng.permutation(len(instance_train)).tolist()
    test_indices = (rng.permutation(len(instance_test)) + len(instance_train)).tolist()

    candidates: list[Design] = []
    best_trace: list[Design] = []
    best_cost = np.inf
    evaluated: dict[tuple[int, ...], Design] = {}
    train_rng_state = deepcopy(rng.bit_generator.state)
    for row in generated.sequences.detach().cpu().numpy():
        key = tuple(int(token) for token in row)
        if key in evaluated:
            algorithm = deepcopy(evaluated[key])
        else:
            rng.bit_generator.state = deepcopy(train_rng_state)
            algorithm = decode_sequence(row, problems, setting)
            algorithm.evaluate(problems, data, setting, train_indices)
            evaluated[key] = deepcopy(algorithm)
        candidates.append(algorithm)
        cost = float(np.mean(algorithm.performance[train_indices, :]))
        if cost < best_cost:
            best_cost = cost
            best_trace.append(algorithm)

    rng.bit_generator.state = deepcopy(train_rng_state)
    rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)

    candidates.sort(
        key=lambda algorithm: float(np.mean(algorithm.performance[train_indices, :]))
    )
    finalists = candidates[: int(setting.AlgN)]
    test_rng_state = deepcopy(rng.bit_generator.state)
    for algorithm in finalists:
        rng.bit_generator.state = deepcopy(test_rng_state)
        algorithm.evaluate(problems, data, setting, test_indices)
    rng.bit_generator.state = deepcopy(test_rng_state)
    rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
    finalists.sort(
        key=lambda algorithm: float(np.mean(algorithm.performance[test_indices, :]))
    )
    return finalists, best_trace


__all__ = ["design_with_aldes"]
