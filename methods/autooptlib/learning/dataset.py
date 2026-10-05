"""Safe graph-performance datasets for learning-based algorithm design."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..utils.design import Design
from ._checkpoint import atomic_text_write
from .codec import LearningCodec
from .grammar import LearningGrammar
from .vocabulary import LearningVocabulary


@dataclass(frozen=True)
class SequenceRecord:
    """One quantized graph sequence with its previously measured cost."""

    sequence: tuple[int, ...]
    performance: float
    metadata: Mapping[str, Any] | None = None


class GraphPerformanceDataset:
    """A versioned collection of graph sequences and preserved performances."""

    def __init__(
        self,
        vocabulary: LearningVocabulary,
        grammar: LearningGrammar,
        records: Iterable[SequenceRecord],
    ) -> None:
        if grammar.vocabulary != vocabulary:
            raise ValueError("Dataset grammar and vocabulary do not match.")
        normalized = []
        for record in records:
            sequence = tuple(grammar.validate(record.sequence))
            if isinstance(record.performance, (bool, np.bool_)):
                raise ValueError("Dataset performances must be numeric, not boolean.")
            performance = float(record.performance)
            if not np.isfinite(performance):
                raise ValueError("Dataset performances must be finite.")
            if record.metadata is not None and not isinstance(record.metadata, Mapping):
                raise TypeError("Dataset metadata must be a mapping or None.")
            normalized.append(
                SequenceRecord(
                    sequence=sequence,
                    performance=performance,
                    metadata=dict(record.metadata or {}),
                )
            )
        if not normalized:
            raise ValueError("A graph-performance dataset cannot be empty.")
        self.vocabulary = vocabulary
        self.grammar = grammar
        self.records = tuple(normalized)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def performances(self) -> np.ndarray:
        return np.asarray([record.performance for record in self.records], dtype=float)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def padded_sequences(self) -> np.ndarray:
        length = max(len(record.sequence) for record in self.records)
        result = np.full(
            (len(self.records), length),
            self.vocabulary.end_index,
            dtype=np.int64,
        )
        for index, record in enumerate(self.records):
            result[index, : len(record.sequence)] = record.sequence
        return result

    def subset(self, indices: Sequence[int]) -> "GraphPerformanceDataset":
        return GraphPerformanceDataset(
            self.vocabulary,
            self.grammar,
            (self.records[int(index)] for index in indices),
        )

    def split(
        self, validation_fraction: float, *, seed: int = 0
    ) -> tuple["GraphPerformanceDataset", "GraphPerformanceDataset | None"]:
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1).")
        if validation_fraction == 0 or len(self) < 2:
            return self, None
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self))
        validation_count = max(1, int(round(len(self) * validation_fraction)))
        validation_count = min(validation_count, len(self) - 1)
        return (
            self.subset(order[validation_count:]),
            self.subset(order[:validation_count]),
        )

    def deduplicate(self, aggregation: str = "best") -> "GraphPerformanceDataset":
        """Combine graphs that collapse to the same quantized sequence."""

        aggregation = str(aggregation).lower()
        if aggregation not in {"best", "mean"}:
            raise ValueError("aggregation must be 'best' or 'mean'.")
        groups: dict[tuple[int, ...], list[SequenceRecord]] = {}
        for record in self.records:
            groups.setdefault(record.sequence, []).append(record)
        combined = []
        for sequence, group in groups.items():
            if aggregation == "best":
                selected = min(group, key=lambda item: item.performance)
                performance = selected.performance
                metadata = selected.metadata
            else:
                performance = float(np.mean([item.performance for item in group]))
                metadata = {"duplicates": len(group)}
            combined.append(SequenceRecord(sequence, performance, metadata))
        return GraphPerformanceDataset(self.vocabulary, self.grammar, combined)

    @classmethod
    def merge(
        cls, datasets: Sequence["GraphPerformanceDataset"]
    ) -> "GraphPerformanceDataset":
        """Merge same- or cross-type datasets into one typed token space."""

        if not datasets:
            raise ValueError("At least one dataset is required.")
        bounds = {
            (
                dataset.grammar.max_pathways,
                dataset.grammar.max_search_components,
                dataset.grammar.population_size_bins,
                dataset.grammar.offspring_size_bins,
                dataset.grammar.configuration_digits,
                dataset.grammar.encode_boundary_handling,
                dataset.grammar.population_size_values,
                dataset.grammar.offspring_size_values,
                dataset.grammar.graph_semantics,
            )
            for dataset in datasets
        }
        if len(bounds) != 1:
            raise ValueError("Merged datasets must use the same AlgP/AlgQ bounds.")
        vocabulary = LearningVocabulary.merge(
            tuple(dataset.vocabulary for dataset in datasets)
        )
        (
            max_pathways,
            max_search_components,
            population_size_bins,
            offspring_size_bins,
            configuration_digits,
            encode_boundary_handling,
            population_size_values,
            offspring_size_values,
            graph_semantics,
        ) = next(iter(bounds))
        grammar = LearningGrammar(
            vocabulary,
            max_pathways=max_pathways,
            max_search_components=max_search_components,
            population_size_bins=population_size_bins,
            offspring_size_bins=offspring_size_bins,
            configuration_digits=configuration_digits,
            encode_boundary_handling=encode_boundary_handling,
            population_size_values=population_size_values,
            offspring_size_values=offspring_size_values,
            graph_semantics=graph_semantics,
        )
        records = []
        for dataset in datasets:
            for record in dataset.records:
                names = [dataset.vocabulary.name(token) for token in record.sequence]
                sequence = tuple(vocabulary.index(name) for name in names)
                records.append(
                    SequenceRecord(
                        sequence,
                        record.performance,
                        record.metadata,
                    )
                )
        return cls(vocabulary, grammar, records)

    @classmethod
    def from_designs(
        cls,
        designs: Sequence[Design],
        codec: LearningCodec,
        *,
        instance_indices: Sequence[int] | None = None,
        aggregation: str = "mean",
        deduplicate: str | None = "best",
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> "GraphPerformanceDataset":
        """Encode evaluated graphs without modifying or re-evaluating them."""

        if not designs:
            raise ValueError("designs cannot be empty.")
        if aggregation not in {"mean", "median", "best"}:
            raise ValueError("aggregation must be 'mean', 'median', or 'best'.")
        indices = None
        if instance_indices is not None:
            raw_indices = list(instance_indices)
            indices = []
            for raw in raw_indices:
                try:
                    value = int(raw)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("instance_indices must contain integers.") from exc
                if isinstance(raw, (bool, np.bool_)) or raw != value:
                    raise ValueError("instance_indices must contain integers.")
                indices.append(value)
            if not indices or len(indices) != len(set(indices)) or min(indices) < 0:
                raise IndexError("instance_indices must be unique nonnegative indices.")
        metadata_values = [None] * len(designs) if metadata is None else list(metadata)
        if len(metadata_values) != len(designs):
            raise ValueError("metadata must provide one entry per design.")
        records = []
        for index, design in enumerate(designs):
            performance = np.asarray(getattr(design, "performance", ()), dtype=float)
            if performance.ndim != 2 or performance.size == 0:
                raise ValueError(
                    "Every design must contain a two-dimensional performance matrix."
                )
            if indices is not None:
                if max(indices) >= performance.shape[0]:
                    raise IndexError(
                        "instance_indices must be inside every performance matrix."
                    )
                performance = performance[indices, :]
            if performance.size == 0 or not np.all(np.isfinite(performance)):
                raise ValueError(
                    "Every selected design performance must be complete and finite."
                )
            if aggregation == "mean":
                cost = float(np.mean(performance))
            elif aggregation == "median":
                cost = float(np.median(performance))
            else:
                cost = float(np.min(performance))
            records.append(
                SequenceRecord(
                    tuple(codec.encode(design)),
                    cost,
                    metadata_values[index],
                )
            )
        dataset = cls(codec.vocabulary, codec.grammar, records)
        return dataset if deduplicate is None else dataset.deduplicate(deduplicate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "autooptlib.learning.dataset",
            "schema_version": 2,
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
            "records": [
                {
                    "sequence": list(record.sequence),
                    "performance": record.performance,
                    "metadata": dict(record.metadata or {}),
                }
                for record in self.records
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphPerformanceDataset":
        if value.get("schema") != "autooptlib.learning.dataset":
            raise ValueError("Not an AutoOptLib graph-performance dataset.")
        if value.get("schema_version") not in {1, 2}:
            raise ValueError("Unsupported graph-performance dataset version.")
        if any(record.get("features") is not None for record in value["records"]):
            raise ValueError("Feature-conditioned datasets are no longer supported.")
        vocabulary = LearningVocabulary.from_dict(value["vocabulary"])
        grammar_value = value["grammar"]
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
        records = (
            SequenceRecord(
                sequence=tuple(record["sequence"]),
                performance=record["performance"],
                metadata=record.get("metadata"),
            )
            for record in value["records"]
        )
        return cls(vocabulary, grammar, records)

    def save(self, path: str | Path) -> Path:
        target = Path(path).resolve()
        payload = json.dumps(
            self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False
        )
        return atomic_text_write(target, payload + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "GraphPerformanceDataset":
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, Mapping):
            raise TypeError("Graph-performance dataset root must be an object.")
        return cls.from_dict(value)


__all__ = ["GraphPerformanceDataset", "SequenceRecord"]
