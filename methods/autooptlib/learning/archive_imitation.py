"""Global-elite archive imitation for learning-based algorithm design."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from ._checkpoint import capture_rng_state, restore_rng_state
from .model import LearningGenerator

ARCHIVE_IMITATION_IMPLEMENTATION = "global-elite-archive-imitation-v3"


@dataclass(frozen=True)
class ArchiveImitationConfig:
    """Optimization settings for global-elite archive imitation.

    ``archive_size`` (mu) and ``candidates`` (lambda) are independent: each
    round merges new candidates with the archive and retains up to mu elites.
    A larger archive fills across batches; only the first batch skips updates.
    """

    learning_rate: float = 5e-5
    final_learning_rate: float = 5e-5
    anneal_steps: int = 100
    adam_epsilon: float = 5e-9
    weight_decay: float = 5e-4
    gradient_norm: float = 1.0
    candidates: int = 10
    archive_size: int = 10
    update_epochs: int = 1
    entropy_coefficient: float = 0.01
    final_entropy_coefficient: float = 0.0
    pairwise_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in ("anneal_steps", "candidates", "archive_size", "update_epochs"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name in (
            "learning_rate",
            "final_learning_rate",
            "adam_epsilon",
            "weight_decay",
            "gradient_norm",
            "entropy_coefficient",
            "final_entropy_coefficient",
            "pairwise_weight",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a real number.")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0 < self.final_learning_rate <= self.learning_rate:
            raise ValueError(
                "final_learning_rate must be positive and cannot exceed learning_rate."
            )
        if self.adam_epsilon <= 0:
            raise ValueError("adam_epsilon must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if self.pairwise_weight < 0:
            raise ValueError("pairwise_weight cannot be negative.")
        if self.gradient_norm <= 0:
            raise ValueError("gradient_norm must be positive.")
        if self.entropy_coefficient < 0 or self.final_entropy_coefficient < 0:
            raise ValueError("entropy coefficients cannot be negative.")
        if self.final_entropy_coefficient > self.entropy_coefficient:
            raise ValueError(
                "final_entropy_coefficient cannot exceed entropy_coefficient."
            )


@dataclass(frozen=True)
class _ArchiveRecord:
    sequence: tuple[int, ...]
    task_costs: tuple[float, ...]
    structure_key: str


class ArchiveImitationTrainer:
    """Fit a generator to a persistent global archive of elite algorithms.

    The first batch only initializes the archive.  Every later batch is merged
    with the current archive, the best ``archive_size`` records are retained,
    and the generator is trained by equal-weight sequence imitation plus an
    annealed entropy bonus evaluated on the newly sampled sequences. Optional
    pairwise ranking learns from the prior archive and all incoming candidates,
    including candidates not retained in the updated archive.
    """

    def __init__(
        self,
        model: LearningGenerator,
        config: ArchiveImitationConfig | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.config = config or ArchiveImitationConfig()
        if optimizer is not None:
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise TypeError("optimizer must be a torch optimizer.")
            model_parameters = [id(parameter) for parameter in model.parameters()]
            optimizer_parameters = [
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            if len(optimizer_parameters) != len(model_parameters) or set(
                optimizer_parameters
            ) != set(model_parameters):
                raise ValueError(
                    "optimizer must cover every model parameter exactly once."
                )
        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=self.config.adam_epsilon,
        )
        self.steps = 0
        self._archive: list[_ArchiveRecord] = []

    @property
    def archive_size(self) -> int:
        return len(self._archive)

    def archive_state(self) -> list[dict[str, Any]]:
        """Return a serializable, order-preserving archive snapshot."""

        return [
            {
                "sequence": list(record.sequence),
                "task_costs": list(record.task_costs),
                "structure_key": record.structure_key,
            }
            for record in self._archive
        ]

    def load_archive_state(self, state: Sequence[dict[str, Any]]) -> None:
        """Restore a validated archive from a training checkpoint."""

        if not isinstance(state, (list, tuple)):
            raise ValueError("Archive checkpoint state must be a sequence.")
        if len(state) > self.config.archive_size:
            raise ValueError("Archive checkpoint exceeds configured archive_size.")
        restored: list[_ArchiveRecord] = []
        seen_sequences: set[tuple[int, ...]] = set()
        seen_structures: set[str] = set()
        task_count: int | None = None
        for item in state:
            if not isinstance(item, dict):
                raise ValueError("Invalid archive checkpoint record.")
            try:
                sequence = tuple(self.model.grammar.validate(item["sequence"]))
                raw_costs = item["task_costs"]
                structure_key = item["structure_key"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Invalid archive checkpoint record.") from exc
            if not isinstance(raw_costs, (list, tuple)) or not raw_costs:
                raise ValueError("Invalid archive checkpoint record.")
            if any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                for value in raw_costs
            ):
                raise ValueError("Invalid archive checkpoint record.")
            try:
                task_costs = tuple(float(value) for value in raw_costs)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Invalid archive checkpoint record.") from exc
            if not all(math.isfinite(value) for value in task_costs):
                raise FloatingPointError("Archive task costs must be finite.")
            if not isinstance(structure_key, str) or not structure_key:
                raise ValueError("Invalid archive checkpoint record.")
            if sequence in seen_sequences or structure_key in seen_structures:
                raise ValueError("Archive checkpoint contains a duplicate candidate.")
            if task_count is None:
                task_count = len(task_costs)
            elif len(task_costs) != task_count:
                raise ValueError("Archive records must cover identical tasks.")
            seen_sequences.add(sequence)
            seen_structures.add(structure_key)
            restored.append(_ArchiveRecord(sequence, task_costs, structure_key))
        self._archive = restored

    def _validated_records(
        self,
        sequences: torch.Tensor,
        task_costs: torch.Tensor | np.ndarray,
        structure_keys: Sequence[str],
    ) -> list[_ArchiveRecord]:
        tensor = torch.as_tensor(sequences)
        # A budget remainder is allowed after a full first batch, even when
        # the configured archive capacity has not yet been reached.
        if tensor.ndim != 2 or not (
            tensor.shape[0] == self.config.candidates
            or (
                0 < tensor.shape[0] < self.config.candidates
                and self.steps > 0
                and bool(self._archive)
            )
        ):
            raise ValueError(
                "sequences must contain exactly ArchiveImitationConfig.candidates "
                "rows for initialization, or a nonempty smaller batch afterward."
            )
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if tensor.dtype not in integer_dtypes:
            raise ValueError("Archive sequences must contain integer token indices.")
        raw_costs = (
            task_costs.detach().cpu().numpy()
            if isinstance(task_costs, torch.Tensor)
            else task_costs
        )
        try:
            raw_values = np.asarray(raw_costs)
        except (TypeError, ValueError) as exc:
            raise ValueError("task_costs must be a rectangular numeric array.") from exc
        if raw_values.dtype.kind not in {"i", "u", "f"}:
            raise ValueError(
                "task_costs must contain real numbers, not coercible values."
            )
        values = np.asarray(raw_values, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if (
            values.ndim != 2
            or values.shape[0] != tensor.shape[0]
            or not values.shape[1]
        ):
            raise ValueError("task_costs must contain one nonempty row per candidate.")
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("Archive task costs must be finite.")
        if self._archive and values.shape[1] != len(self._archive[0].task_costs):
            raise ValueError("Archive task coverage changed during training.")
        if not isinstance(structure_keys, (list, tuple)):
            raise ValueError("structure_keys must be a sequence of strings.")
        if len(structure_keys) != tensor.shape[0] or any(
            not isinstance(key, str) or not key for key in structure_keys
        ):
            raise ValueError("structure_keys must contain one nonempty string per row.")

        records: list[_ArchiveRecord] = []
        for row, costs, structure_key in zip(tensor, values, structure_keys):
            try:
                sequence = tuple(
                    self.model.grammar.validate(row.detach().cpu().numpy())
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Archive sequence violates the learning grammar."
                ) from exc
            records.append(
                _ArchiveRecord(
                    sequence,
                    tuple(float(value) for value in costs),
                    structure_key,
                )
            )
        if len({record.sequence for record in records}) != len(records):
            raise ValueError("Archive candidate batch contains duplicate sequences.")
        if len({record.structure_key for record in records}) != len(records):
            raise ValueError("Archive candidate batch contains duplicate structures.")
        return records

    @staticmethod
    def _finite_mean(values: Sequence[float]) -> float:
        """Compute a finite mean without overflowing on finite inputs."""

        scale = max(abs(value) for value in values)
        if scale == 0:
            return 0.0
        ratio = math.fsum(value / scale for value in values) / len(values)
        return max(-1.0, min(1.0, ratio)) * scale

    @classmethod
    def _mean_task_cost(cls, record: _ArchiveRecord) -> float:
        return cls._finite_mean(record.task_costs)

    @classmethod
    def _record_order(cls, records: list[_ArchiveRecord]) -> list[int]:
        mean_costs = [cls._mean_task_cost(record) for record in records]
        return sorted(
            range(len(records)),
            key=lambda index: (
                float(mean_costs[index]),
                records[index].structure_key,
                records[index].sequence,
            ),
        )

    def _select_archive(self, incoming: list[_ArchiveRecord]) -> list[_ArchiveRecord]:
        # Graph identity is authoritative.  The lower-cost observation wins if a
        # caller accidentally supplies the same graph twice across rounds.
        by_structure: dict[str, _ArchiveRecord] = {}
        for record in [*self._archive, *incoming]:
            previous = by_structure.get(record.structure_key)
            if previous is None or (
                self._mean_task_cost(record),
                record.sequence,
            ) < (self._mean_task_cost(previous), previous.sequence):
                by_structure[record.structure_key] = record
        candidates = list(by_structure.values())
        order = self._record_order(candidates)
        return [candidates[index] for index in order[: self.config.archive_size]]

    def _archive_tensor(self) -> torch.Tensor:
        device = next(self.model.parameters()).device
        rows = [
            torch.as_tensor(record.sequence, dtype=torch.long)
            for record in self._archive
        ]
        return nn.utils.rnn.pad_sequence(
            rows,
            batch_first=True,
            padding_value=self.model.end_index,
        ).to(device)

    def _imitation_loss(self, sequences: torch.Tensor) -> torch.Tensor:
        log_probability = self.model.score(sequences)
        targets = sequences[:, 1:]
        lengths = targets.ne(self.model.end_index).sum(dim=1) + 1
        return -(log_probability / lengths.to(log_probability.dtype)).mean()

    @classmethod
    def _preference_pairs(cls, records: list[_ArchiveRecord]) -> list[tuple[int, int]]:
        """Prefer lower mean cost with wins on at least two thirds of task groups.

        This is a consistency filter, not a statistical significance test.
        Scale-aware 1e-6 ties are ignored. Scaling before subtraction also
        keeps comparisons finite for extreme but valid input costs.
        """
        pairs = []
        for i, left in enumerate(records):
            for j in range(i + 1, len(records)):
                right = records[j]
                scale = max(
                    1.0, *(abs(x) for x in (*left.task_costs, *right.task_costs))
                )
                differences = (
                    np.asarray(right.task_costs) / scale
                    - np.asarray(left.task_costs) / scale
                )
                required = math.ceil(2 * len(differences) / 3)
                if (
                    np.mean(differences) > 1e-6
                    and np.count_nonzero(differences > 1e-6) >= required
                ):
                    pairs.append((i, j))
                elif (
                    np.mean(differences) < -1e-6
                    and np.count_nonzero(differences < -1e-6) >= required
                ):
                    pairs.append((j, i))
        return pairs

    def _pairwise_loss(
        self, sequences: torch.Tensor, pairs: torch.Tensor
    ) -> torch.Tensor:
        """Mean logistic ranking loss on length-normalized sequence log probabilities."""
        scores = self.model.score(sequences)
        lengths = sequences[:, 1:].ne(self.model.end_index).sum(dim=1) + 1
        scores = scores / lengths.to(scores.dtype)
        return torch.nn.functional.softplus(
            -(scores[pairs[:, 0]] - scores[pairs[:, 1]])
        ).mean()

    def _annealed_values(self, progress: float | None) -> tuple[float, float]:
        if progress is None:
            # Step zero initializes the archive. With multiple real updates the
            # first starts at zero and the last reaches one; a single real update
            # uses the final endpoint.
            update_count = max(self.config.anneal_steps - 1, 1)
            progress = (
                float(self.steps > 0)
                if update_count == 1
                else min(max(self.steps - 1, 0) / (update_count - 1), 1.0)
            )
        elif (
            isinstance(progress, (bool, np.bool_))
            or not isinstance(progress, Real)
            or not math.isfinite(progress)
            or not 0 <= progress <= 1
        ):
            raise ValueError("annealing progress must be in [0, 1].")
        learning_rate = self.config.learning_rate + progress * (
            self.config.final_learning_rate - self.config.learning_rate
        )
        entropy_coefficient = self.config.entropy_coefficient + progress * (
            self.config.final_entropy_coefficient - self.config.entropy_coefficient
        )
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return float(learning_rate), float(entropy_coefficient)

    def step(
        self,
        sequences: torch.Tensor,
        task_costs: torch.Tensor | np.ndarray,
        structure_keys: Sequence[str],
        *,
        annealing_progress: float | None = None,
    ) -> dict[str, float]:
        """Merge one new batch into the archive and, after round one, update."""

        was_training = self.model.training
        model_state = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())
        archive_before = list(self._archive)
        steps_before = self.steps
        rng_state_before = capture_rng_state()
        try:
            incoming = self._validated_records(sequences, task_costs, structure_keys)
            previous_keys = tuple(record.structure_key for record in self._archive)
            expected_archive_size = min(
                self.config.archive_size,
                len(set(previous_keys) | {record.structure_key for record in incoming}),
            )
            self._archive = self._select_archive(incoming)
            current_keys = tuple(record.structure_key for record in self._archive)
            archive_changed = previous_keys != current_keys
            if len(self._archive) != expected_archive_size:
                raise RuntimeError(
                    "Archive selection did not retain the expected unique elites."
                )

            learning_rate, entropy_coefficient = self._annealed_values(
                annealing_progress
            )
            device = next(self.model.parameters()).device
            new_sequences = torch.as_tensor(sequences, dtype=torch.long, device=device)
            archive_sequences = self._archive_tensor()
            pairs = []
            pair_sequences = pair_indices = None
            if self.config.pairwise_weight > 0:
                # Include rejected new candidates, not just surviving elites.
                # One observation per graph prevents duplicate pair weighting.
                pool = {}
                for record in [*archive_before, *incoming]:
                    previous = pool.get(record.structure_key)
                    if previous is None or self._mean_task_cost(
                        record
                    ) < self._mean_task_cost(previous):
                        pool[record.structure_key] = record
                records = list(pool.values())
                pairs = self._preference_pairs(records)
                if pairs:
                    pair_sequences = nn.utils.rnn.pad_sequence(
                        [
                            torch.tensor(
                                record.sequence, dtype=torch.long, device=device
                            )
                            for record in records
                        ],
                        batch_first=True,
                        padding_value=self.model.end_index,
                    )
                    pair_indices = torch.tensor(pairs, dtype=torch.long, device=device)
            final_loss = torch.zeros((), device=device)
            final_imitation = torch.zeros((), device=device)
            final_entropy = torch.zeros((), device=device)
            final_entropy_loss = torch.zeros((), device=device)
            final_pairwise = torch.zeros((), device=device)
            update_applied = self.steps > 0 and learning_rate > 0
            if update_applied:
                self.model.train(True)
                for _ in range(self.config.update_epochs):
                    imitation = self._imitation_loss(archive_sequences)
                    entropy = self.model.entropy(new_sequences).mean()
                    entropy_loss = -entropy_coefficient * entropy
                    loss = imitation + entropy_loss
                    pairwise = torch.zeros((), device=device)
                    if pair_indices is not None:
                        pairwise = self._pairwise_loss(pair_sequences, pair_indices)
                        loss = loss + self.config.pairwise_weight * pairwise
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(
                            "Archive-imitation loss became non-finite."
                        )
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gradient_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_norm
                    )
                    if not bool(torch.isfinite(gradient_norm)):
                        self.optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError(
                            "Archive-imitation gradients became non-finite."
                        )
                    self.optimizer.step()
                    if not all(
                        bool(torch.isfinite(parameter).all())
                        for parameter in self.model.parameters()
                    ):
                        raise FloatingPointError(
                            "Archive-imitation parameters became non-finite."
                        )
                    if not self._optimizer_state_is_finite():
                        raise FloatingPointError(
                            "Archive-imitation optimizer state became non-finite."
                        )
                    final_loss = loss.detach()
                    final_imitation = imitation.detach()
                    final_entropy = entropy.detach()
                    final_entropy_loss = entropy_loss.detach()
                    final_pairwise = pairwise.detach()
            self.steps += 1
        except Exception:
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)
            self.optimizer.zero_grad(set_to_none=True)
            self._archive = archive_before
            self.steps = steps_before
            restore_rng_state(rng_state_before)
            raise
        finally:
            self.model.train(was_training)

        archive_mean_costs = [self._mean_task_cost(record) for record in self._archive]
        metrics = {
            "loss": float(final_loss.cpu()),
            "archive_imitation_loss": float(final_imitation.cpu()),
            "entropy": float(final_entropy.cpu()),
            "entropy_loss": float(final_entropy_loss.cpu()),
            "entropy_coefficient": entropy_coefficient,
            "learning_rate": learning_rate,
            "archive_size": float(len(self._archive)),
            "archive_changed": float(archive_changed),
            "archive_update_applied": float(update_applied),
            "archive_mean_task_cost": self._finite_mean(archive_mean_costs),
            "archive_best_mean_task_cost": min(archive_mean_costs),
        }
        if self.config.pairwise_weight > 0:
            metrics.update(
                pairwise_loss=float(final_pairwise.cpu()),
                pairwise_weight=self.config.pairwise_weight,
                pairwise_pairs=float(len(pairs)),
                pairwise_update_applied=float(update_applied and bool(pairs)),
            )
        return metrics

    def consolidate_archive(self, updates: int = 1) -> dict[str, float]:
        """Continue equal-weight imitation of the fixed current archive.

        This diagnostic operation performs no sampling, evaluation, archive
        selection, entropy update, or design-step accounting.  It is intended
        to test whether an already trained policy can absorb its final archive
        when given additional optimizer updates.
        """

        if type(updates) is not int or updates <= 0:
            raise ValueError("updates must be a positive integer.")
        if len(self._archive) != self.config.archive_size:
            raise RuntimeError(
                "Archive consolidation requires a complete elite archive."
            )

        was_training = self.model.training
        model_state = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())
        archive_before = list(self._archive)
        steps_before = self.steps
        rng_state_before = capture_rng_state()
        device = next(self.model.parameters()).device
        final_loss = torch.zeros((), device=device)
        final_gradient_norm = torch.zeros((), device=device)
        try:
            archive_sequences = self._archive_tensor()
            self.model.train(True)
            for _ in range(updates):
                loss = self._imitation_loss(archive_sequences)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        "Archive-consolidation loss became non-finite."
                    )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_norm
                )
                if not bool(torch.isfinite(gradient_norm)):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        "Archive-consolidation gradients became non-finite."
                    )
                self.optimizer.step()
                if not all(
                    bool(torch.isfinite(value).all())
                    for value in self.model.state_dict().values()
                ):
                    raise FloatingPointError(
                        "Archive-consolidation model state became non-finite."
                    )
                if not self._optimizer_state_is_finite():
                    raise FloatingPointError(
                        "Archive-consolidation optimizer state became non-finite."
                    )
                final_loss = loss.detach()
                final_gradient_norm = gradient_norm.detach()
        except Exception:
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)
            self.optimizer.zero_grad(set_to_none=True)
            self._archive = archive_before
            self.steps = steps_before
            restore_rng_state(rng_state_before)
            raise
        finally:
            self.model.train(was_training)

        return {
            "archive_consolidation_loss": float(final_loss.cpu()),
            "archive_consolidation_gradient_norm": float(final_gradient_norm.cpu()),
            "archive_consolidation_updates": float(updates),
            "archive_size": float(len(self._archive)),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }

    def _optimizer_state_is_finite(self) -> bool:
        """Return whether every numeric optimizer-state leaf is finite."""

        def finite(value: Any) -> bool:
            if isinstance(value, torch.Tensor):
                return not (value.is_floating_point() or value.is_complex()) or bool(
                    torch.isfinite(value).all()
                )
            if isinstance(value, np.ndarray):
                return value.dtype.kind not in {"f", "c"} or bool(
                    np.isfinite(value).all()
                )
            if isinstance(value, dict):
                return all(finite(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return all(finite(item) for item in value)
            if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
                return math.isfinite(float(value))
            return True

        return finite(self.optimizer.state_dict())


__all__ = [
    "ARCHIVE_IMITATION_IMPLEMENTATION",
    "ArchiveImitationConfig",
    "ArchiveImitationTrainer",
]
