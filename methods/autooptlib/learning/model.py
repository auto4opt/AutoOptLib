"""Autoregressive generator for AutoOptLib graph sequences."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ._checkpoint import atomic_torch_save
from ._generator import AutoregressiveGenerator, GenerationResult, GeneratorConfig
from .device import resolve_device
from .grammar import LearningGrammar
from .vocabulary import LearningVocabulary


class LearningGenerator(AutoregressiveGenerator):
    """Generate grammar-valid sequences over the AutoOptLib graph space."""

    def __init__(
        self,
        vocabulary: LearningVocabulary,
        grammar: LearningGrammar,
        config: GeneratorConfig | None = None,
        *,
        device: Any = "auto",
    ) -> None:
        if grammar.vocabulary != vocabulary:
            raise ValueError("Learning grammar and vocabulary do not match.")
        resolved_config = config or GeneratorConfig()
        required_length = grammar.maximum_length()
        if resolved_config.max_length < required_length:
            raise ValueError(
                f"GeneratorConfig.max_length={resolved_config.max_length} cannot "
                f"represent this grammar; use at least {required_length}."
            )
        self.vocabulary = vocabulary
        self.grammar = grammar
        self.active_problem_type: str | None = None
        super().__init__(
            resolved_config,
            vocabulary_size=vocabulary.size,
            begin_index=vocabulary.begin_index,
            end_index=vocabulary.end_index,
            allowed_next=self._allowed_next,
            generator_name="AutoOptLib learning designer",
        )
        self.to(resolve_device(device))

    def _allowed_next(self, sequence):
        return self.grammar.allowed_next_tokens(
            sequence, required_problem_type=self.active_problem_type
        )

    def set_problem_type(self, problem_type: str | None) -> None:
        """Restrict generation/training to one type in a mixed vocabulary."""

        if problem_type is not None:
            problem_type = str(problem_type).lower()
            if problem_type not in self.vocabulary.supported_problem_types:
                raise ValueError("Problem type is outside the learning vocabulary.")
        self.active_problem_type = problem_type

    def save_checkpoint(self, path: str | Path, **metadata: Any) -> None:
        atomic_torch_save(
            {
                "schema": "autooptlib.learning.generator",
                "schema_version": 2,
                "config": asdict(self.config),
                "vocabulary": self.vocabulary.to_dict(),
                "grammar": {
                    "max_pathways": self.grammar.max_pathways,
                    "max_search_components": self.grammar.max_search_components,
                    "population_size_bins": self.grammar.population_size_bins,
                    "offspring_size_bins": self.grammar.offspring_size_bins,
                    "configuration_digits": self.grammar.configuration_digits,
                    "encode_boundary_handling": self.grammar.encode_boundary_handling,
                    "population_size_values": list(self.grammar.population_size_values),
                    "offspring_size_values": list(self.grammar.offspring_size_values),
                    "graph_semantics": self.grammar.graph_semantics,
                },
                "state_dict": self.state_dict(),
                "metadata": metadata,
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls, path: str | Path, *, map_location: Any = None
    ) -> tuple["LearningGenerator", dict[str, Any]]:
        device = (
            resolve_device("auto")
            if map_location is None
            else torch.device(map_location)
        )
        payload = torch.load(Path(path), map_location=device, weights_only=True)
        if payload.get("schema") != "autooptlib.learning.generator":
            raise ValueError("Not an AutoOptLib learning-generator checkpoint.")
        if payload.get("schema_version") not in {1, 2}:
            raise ValueError("Unsupported learning-generator checkpoint version.")
        config_value = dict(payload["config"])
        if config_value.pop("condition_on_features", False) or any(
            name.startswith("feature_projection.") for name in payload["state_dict"]
        ):
            raise ValueError(
                "Feature-conditioned checkpoints are no longer supported; "
                "train an unconditioned source model."
            )
        config_value.pop("feature_dim", None)
        vocabulary = LearningVocabulary.from_dict(payload["vocabulary"])
        grammar_value = payload["grammar"]
        grammar = LearningGrammar(
            vocabulary,
            max_pathways=grammar_value["max_pathways"],
            max_search_components=grammar_value["max_search_components"],
            population_size_bins=grammar_value.get("population_size_bins", 10),
            offspring_size_bins=grammar_value.get("offspring_size_bins", 10),
            configuration_digits=grammar_value.get("configuration_digits", 3),
            encode_boundary_handling=grammar_value.get(
                "encode_boundary_handling", False
            ),
            population_size_values=tuple(
                grammar_value.get("population_size_values", ())
            ),
            offspring_size_values=tuple(grammar_value.get("offspring_size_values", ())),
            graph_semantics=grammar_value.get("graph_semantics", "legacy_pathway_v1"),
        )
        model = cls(
            vocabulary,
            grammar,
            GeneratorConfig(**config_value),
            device=device,
        )
        model.load_state_dict(payload["state_dict"])
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Learning-generator metadata must be a dictionary.")
        return model, dict(metadata)


__all__ = [
    "GenerationResult",
    "GeneratorConfig",
    "LearningGenerator",
]
