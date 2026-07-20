"""ALDes: autoregressive learning for metaheuristic algorithm design.

The codec and evaluator require only NumPy and AutoOptLib.  PyTorch is loaded
only when the generator or PPO training APIs are requested.
"""

from __future__ import annotations

from importlib import import_module

from .codec import ComponentInstruction, decode_sequence
from .evaluator import AutoOptEvaluator, EvaluationConfig, evaluate_pbo_actions
from .features import PBOFeatureResult, extract_pbo_features, standardize_features
from .problems import make_pbo_problem
from .vocabulary import (
    BEGIN_INDEX,
    END_INDEX,
    TOKEN_BY_INDEX,
    TOKEN_BY_NAME,
    TOKENS,
    VOCABULARY_SIZE,
    SequenceValidationError,
    allowed_next_tokens,
    normalize_sequence,
    tokens_to_names,
    validate_sequence,
)

_TORCH_EXPORTS = {
    "ALDesGenerator": ("model", "ALDesGenerator"),
    "GenerationResult": ("model", "GenerationResult"),
    "GeneratorConfig": ("model", "GeneratorConfig"),
    "ElasticWeightConsolidation": ("training", "ElasticWeightConsolidation"),
    "PPOConfig": ("training", "PPOConfig"),
    "PPOTrainer": ("training", "PPOTrainer"),
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
                "PyTorch is required for ALDes generation and training. Install "
                "AutoOptLib with `pip install 'autooptlib[aldes]'`."
            ) from exc
        raise
    value = getattr(module, attribute)
    globals()[name] = value
    return value


__all__ = [
    "ALDesGenerator",
    "AutoOptEvaluator",
    "BEGIN_INDEX",
    "ComponentInstruction",
    "END_INDEX",
    "ElasticWeightConsolidation",
    "EvaluationConfig",
    "GenerationResult",
    "GeneratorConfig",
    "PPOConfig",
    "PPOTrainer",
    "PBOFeatureResult",
    "SequenceValidationError",
    "TOKENS",
    "TOKEN_BY_INDEX",
    "TOKEN_BY_NAME",
    "VOCABULARY_SIZE",
    "allowed_next_tokens",
    "decode_sequence",
    "evaluate_pbo_actions",
    "extract_pbo_features",
    "make_pbo_problem",
    "normalize_sequence",
    "tokens_to_names",
    "standardize_features",
    "validate_sequence",
]
