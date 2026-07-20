"""PPO and continual-learning utilities for ALDes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

from .model import ALDesGenerator


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 5e-5
    final_learning_rate: float = 0.0
    anneal_steps: int = 100
    clip_coefficient: float = 0.2
    update_epochs: int = 5
    gradient_norm: float = 1.0
    candidates: int = 16
    baseline_momentum: float = 0.8
    ewc_weight: float = 200.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.final_learning_rate < 0:
            raise ValueError("final_learning_rate cannot be negative.")
        if self.final_learning_rate > self.learning_rate:
            raise ValueError("final_learning_rate cannot exceed learning_rate.")
        if self.anneal_steps <= 0:
            raise ValueError("anneal_steps must be positive.")
        if self.ewc_weight < 0:
            raise ValueError("ewc_weight cannot be negative.")
        if self.update_epochs <= 0 or self.candidates <= 0:
            raise ValueError("update_epochs and candidates must be positive.")
        if not 0 <= self.clip_coefficient < 1:
            raise ValueError("clip_coefficient must be in [0, 1).")
        if not 0 <= self.baseline_momentum < 1:
            raise ValueError("baseline_momentum must be in [0, 1).")


class ElasticWeightConsolidation:
    """Diagonal Fisher penalty used by ALDes continual training."""

    def __init__(self, model: nn.Module) -> None:
        self.means = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.precision = {
            name: torch.zeros_like(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def accumulate(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.precision and parameter.grad is not None:
                self.precision[name] += parameter.grad.detach().square()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        loss = torch.zeros((), device=next(model.parameters()).device)
        for name, parameter in model.named_parameters():
            if name in self.precision:
                loss = (
                    loss
                    + (
                        self.precision[name] * (parameter - self.means[name]).square()
                    ).sum()
                )
        return loss


class PPOTrainer:
    """Train an ALDes generator from AutoOptLib performance feedback."""

    def __init__(
        self,
        model: ALDesGenerator,
        config: PPOConfig | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.config = config or PPOConfig()
        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(), lr=self.config.learning_rate, weight_decay=5e-4
        )
        self.baseline: torch.Tensor | None = None
        self.steps = 0

    def _anneal_learning_rate(self) -> float:
        progress = min(self.steps / self.config.anneal_steps, 1.0)
        learning_rate = (
            self.config.learning_rate
            + progress
            * (self.config.final_learning_rate - self.config.learning_rate)
        )
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    def step(
        self,
        features: torch.Tensor | None,
        evaluate: Callable[[list[np.ndarray]], np.ndarray],
        *,
        ewc: ElasticWeightConsolidation | None = None,
        ewc_weight: float | None = None,
    ) -> dict[str, float]:
        if ewc_weight is not None and ewc_weight < 0:
            raise ValueError("ewc_weight cannot be negative.")
        was_training = self.model.training
        self.model.eval()
        learning_rate = self._anneal_learning_rate()
        try:
            with torch.no_grad():
                generated = self.model.generate(
                    features, candidates=self.config.candidates
                )
                old_log_probability = generated.log_probabilities.sum(dim=1)
            sequences = [row.cpu().numpy() for row in generated.sequences]
            costs_array = np.asarray(evaluate(sequences), dtype=float).reshape(-1)
            if costs_array.shape[0] != self.config.candidates:
                raise ValueError("Evaluator returned one cost per candidate incorrectly.")
            costs = torch.as_tensor(
                costs_array,
                dtype=old_log_probability.dtype,
                device=old_log_probability.device,
            )
            mean_cost = costs.mean().detach()
            if self.baseline is None:
                self.baseline = mean_cost
            else:
                momentum = self.config.baseline_momentum
                self.baseline = (
                    momentum * self.baseline + (1 - momentum) * mean_cost
                ).detach()
            advantage = costs - self.baseline

            # Keep dropout disabled while computing both the old and new
            # policy probabilities. Gradients still flow in eval mode, and an
            # unchanged policy therefore has an exact initial PPO ratio of 1.
            final_loss = torch.zeros((), device=costs.device)
            for _ in range(self.config.update_epochs):
                new_log_probability = self.model.score(features, generated.sequences)
                ratio = torch.exp(new_log_probability - old_log_probability)
                unclipped = advantage * ratio
                clipped = advantage * torch.clamp(
                    ratio,
                    1 - self.config.clip_coefficient,
                    1 + self.config.clip_coefficient,
                )
                loss = torch.maximum(unclipped, clipped).mean()
                weight = self.config.ewc_weight if ewc_weight is None else ewc_weight
                if ewc is not None and weight:
                    loss = loss + weight * ewc.penalty(self.model)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_norm
                )
                self.optimizer.step()
                final_loss = loss.detach()
        finally:
            self.model.train(was_training)

        self.steps += 1

        return {
            "loss": float(final_loss.cpu()),
            "mean_cost": float(mean_cost.cpu()),
            "baseline": float(self.baseline.cpu()),
            "best_cost": float(costs.min().cpu()),
            "learning_rate": learning_rate,
        }

    def consolidate(
        self,
        features: torch.Tensor | None,
        sequences: torch.Tensor,
        ewc: ElasticWeightConsolidation | None = None,
    ) -> ElasticWeightConsolidation:
        """Estimate a diagonal Fisher term before moving to the next task."""

        state = ewc or ElasticWeightConsolidation(self.model)
        was_training = self.model.training
        self.model.eval()
        self.optimizer.zero_grad()
        try:
            loss = -self.model.score(features, sequences).mean()
            loss.backward()
            state.accumulate(self.model)
        finally:
            self.optimizer.zero_grad()
            self.model.train(was_training)
        return state


__all__ = [
    "ElasticWeightConsolidation",
    "PPOConfig",
    "PPOTrainer",
]
