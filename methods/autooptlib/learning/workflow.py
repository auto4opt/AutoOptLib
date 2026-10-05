"""High-level AutoOptLib workflow for graph-sequence learning models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from problems.base import validate_constructed_problems

from ..utils.design import Design
from ..utils.design._generated import evaluate_generated_designs
from ..utils.design._helpers import get_problem_type
from ..utils.general.process import _build_problem_struct, _normalize_setting
from ..utils.space import space
from .codec import LearningCodec


def _load_initial_populations(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "initial_populations"):
        return value.initial_populations
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(
                f"Learning initial-population file not found: {path}"
            )
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


def design_with_learning(
    problem_descriptor: Any,
    instance_train: Sequence[Any],
    instance_test: Sequence[Any],
    *,
    setting: Any,
) -> tuple[list[Design], list[Design]]:
    """Generate, evaluate, and test graph algorithms with a trained model."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency specific
        raise ImportError(
            "Designer='learning' requires PyTorch. Install AutoOptLib with "
            "`pip install 'autooptlib[learning]'`."
        ) from exc

    from .device import resolve_device
    from .model import LearningGenerator

    setting = _normalize_setting(setting)
    instances = list(instance_train) + list(instance_test)
    problems = _build_problem_struct(problem_descriptor, instances, setting)
    problems, data, _ = problem_descriptor(problems, instances, "construct")
    validate_constructed_problems(problems, data)
    setting = space(problems, setting)

    model_value = getattr(setting, "LearningModel", None)
    if model_value is None:
        raise ValueError(
            "Designer='learning' requires LearningModel "
            "(a LearningGenerator or checkpoint path)."
        )
    device = resolve_device(getattr(setting, "LearningDevice", "auto"))
    if isinstance(model_value, (str, Path)):
        model, _ = LearningGenerator.load_checkpoint(model_value, map_location=device)
    elif isinstance(model_value, LearningGenerator):
        model = model_value.to(device)
    else:
        raise TypeError("LearningModel must be a LearningGenerator or checkpoint path.")
    codec = LearningCodec.from_problem(problems, setting, vocabulary=model.vocabulary)
    if (
        model.grammar.max_pathways != codec.grammar.max_pathways
        or model.grammar.max_search_components != codec.grammar.max_search_components
        or model.grammar.population_size_bins != codec.grammar.population_size_bins
        or model.grammar.offspring_size_bins != codec.grammar.offspring_size_bins
        or model.grammar.configuration_digits != codec.grammar.configuration_digits
        or model.grammar.encode_boundary_handling
        != codec.grammar.encode_boundary_handling
        or model.grammar.population_size_values != codec.grammar.population_size_values
        or model.grammar.offspring_size_values != codec.grammar.offspring_size_values
    ):
        raise ValueError("LearningModel grammar bounds do not match AlgP and AlgQ.")
    setting.InitialPopulations = _load_initial_populations(
        getattr(setting, "LearningInitialPopulations", None)
    )

    if type(setting.AlgN) is not int or setting.AlgN != 1:
        raise ValueError(
            "Designer='learning' performs one deterministic inference; set AlgN=1."
        )
    was_training = model.training
    previous_problem_type = model.active_problem_type
    model.set_problem_type(get_problem_type(problems))
    try:
        model.eval()
        with torch.no_grad():
            generated = model.generate(
                candidates=1,
                greedy=True,
            )
        return evaluate_generated_designs(
            generated.sequences.detach().cpu().numpy(),
            normalize=codec.grammar.normalize,
            decode=codec.decode,
            problems=problems,
            data=data,
            setting=setting,
            train_count=len(instance_train),
            test_count=len(instance_test),
            parallel=True,
        )
    finally:
        model.set_problem_type(previous_problem_type)
        model.train(was_training)


__all__ = ["design_with_learning"]
