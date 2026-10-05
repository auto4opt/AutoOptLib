"""Learning-based design over AutoOptLib's native algorithm graph space."""

from __future__ import annotations

from importlib import import_module

from .codec import LearningCodec, decode_sequence, encode_design
from .dataset import GraphPerformanceDataset, SequenceRecord
from .evaluator import AutoOptEvaluator, EvaluationConfig
from .grammar import LearningGrammar, SequenceValidationError
from .vocabulary import LearningVocabulary

_TORCH_EXPORTS = {
    "ArchiveImitationConfig": ("archive_imitation", "ArchiveImitationConfig"),
    "ArchiveImitationTrainer": ("archive_imitation", "ArchiveImitationTrainer"),
    "describe_device": ("device", "describe_device"),
    "resolve_device": ("device", "resolve_device"),
    "GenerationResult": ("model", "GenerationResult"),
    "GeneratorConfig": ("model", "GeneratorConfig"),
    "LearningGenerator": ("model", "LearningGenerator"),
}


def __getattr__(name: str):
    if name not in _TORCH_EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _TORCH_EXPORTS[name]
    try:
        module = import_module(f".{module_name}", __name__)
    except ImportError as exc:
        if exc.name == "torch":
            raise ImportError(
                "PyTorch is required for learning-based generation and training. "
                "Install AutoOptLib with `pip install 'autooptlib[learning]'`."
            ) from exc
        raise
    value = getattr(module, attribute)
    globals()[name] = value
    return value


__all__ = [
    "ArchiveImitationConfig",
    "ArchiveImitationTrainer",
    "AutoOptEvaluator",
    "EvaluationConfig",
    "GenerationResult",
    "GeneratorConfig",
    "GraphPerformanceDataset",
    "LearningCodec",
    "LearningGenerator",
    "LearningGrammar",
    "LearningVocabulary",
    "SequenceRecord",
    "SequenceValidationError",
    "decode_sequence",
    "describe_device",
    "encode_design",
    "resolve_device",
]
