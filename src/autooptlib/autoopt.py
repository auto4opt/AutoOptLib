"""High-level entry point mirroring MATLAB's AutoOpt function."""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
from typing import Any, List

from .utils.general import input_handler, normalize_options, output, process
from .utils.general.output import write_experiment_manifest


def _to_arguments(kwargs: dict[str, Any]) -> List[Any]:
    args: List[Any] = []
    for key, value in kwargs.items():
        args.extend([key, value])
    return args


def _load_problem_callable(descriptor: Any):
    if callable(descriptor):
        return descriptor
    if isinstance(descriptor, str):
        module = import_module("autooptlib.problems")
        if hasattr(module, descriptor):
            return getattr(module, descriptor)
    raise ValueError(f"Unsupported problem descriptor: {descriptor!r}")


def autoopt(**kwargs: Any):
    """Design optimization algorithms or execute them with validated options."""
    kwargs = normalize_options(kwargs)
    if "Mode" not in kwargs:
        raise ValueError("AutoOpt requires a Mode parameter ('design' or 'solve').")

    mode = str(kwargs["Mode"]).lower()
    kwargs["Mode"] = mode
    from .programming import _configure_model_options

    _configure_model_options(kwargs)
    manifest_options = dict(kwargs)
    setting = SimpleNamespace(Mode=mode)
    setting._ExplicitOptions = frozenset(kwargs)
    output_directory = kwargs.pop("OutputDir", None)
    arguments = _to_arguments(kwargs)

    problem_descriptor, *instances = input_handler(arguments, setting, "data")
    if mode == "design":
        # Independent instances can be statistical blocks even with one run.
        setting._DesignStatisticalBlocks = len(instances[0])
    setting = input_handler(arguments, setting, "parameter")
    setting = input_handler(arguments, setting, "check")
    problem_callable = _load_problem_callable(problem_descriptor)

    solutions, history = process(problem_callable, *instances, setting=setting)
    output(
        solutions,
        history,
        *instances,
        setting=setting,
        directory=output_directory,
    )

    runtime_stats = getattr(setting, "EvalRuntimeStats", None)
    if runtime_stats is not None:
        manifest_options["RuntimeStats"] = runtime_stats
    for attribute, key in (
        ("EvalRuntimeProfiles", "RuntimeProfiles"),
        ("EvalConcurrencyProfiles", "ConcurrencyProfiles"),
    ):
        value = getattr(setting, attribute, None)
        if value:
            manifest_options[key] = value

    if mode == "design":
        for attribute, key in (
            ("EvalActualDesignFEs", "ActualDesignFEs"),
            ("SearchTrainingFEs", "SearchTrainingFEs"),
            ("SearchEvaluatedCandidates", "SearchEvaluatedCandidates"),
            ("SearchUnusedDesignFEs", "SearchUnusedDesignFEs"),
        ):
            value = getattr(setting, attribute, None)
            if value is not None:
                manifest_options[key] = int(value)
        for attribute in ("SearchEvaluationEngine", "SearchControllerStats"):
            value = getattr(setting, attribute, None)
            if value is not None:
                manifest_options[attribute] = value

    write_experiment_manifest(output_directory, mode=mode, options=manifest_options)
    return solutions, history


__all__ = ["autoopt"]
