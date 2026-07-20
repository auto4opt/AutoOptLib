"""Autoregressive PyTorch generator for ALDes token programs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .vocabulary import BEGIN_INDEX, END_INDEX, VOCABULARY_SIZE, allowed_next_tokens


@dataclass(frozen=True)
class GeneratorConfig:
    feature_dim: int = 32
    model_dim: int = 32
    heads: int = 8
    layers: int = 8
    feedforward_dim: int = 2048
    dropout: float = 0.1
    max_length: int = 50
    condition_on_features: bool = False
    position_encoding: str = "sinusoidal"


@dataclass
class GenerationResult:
    sequences: torch.Tensor
    log_probabilities: torch.Tensor
    probabilities: torch.Tensor


class ALDesGenerator(nn.Module):
    """Generate valid algorithms while applying the ALDes grammar mask."""

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        super().__init__()
        self.config = config or GeneratorConfig()
        if self.config.model_dim % self.config.heads:
            raise ValueError("model_dim must be divisible by heads.")
        if self.config.position_encoding not in {"sinusoidal", "learned"}:
            raise ValueError("position_encoding must be 'sinusoidal' or 'learned'.")
        self.token_embedding = nn.Embedding(VOCABULARY_SIZE, self.config.model_dim)
        if self.config.position_encoding == "learned":
            self.position_embedding: nn.Module | None = nn.Embedding(
                self.config.max_length + 1, self.config.model_dim
            )
            self.register_buffer("sinusoidal_positions", None)
        else:
            self.position_embedding = None
            self.register_buffer(
                "sinusoidal_positions",
                self._make_sinusoidal_positions(
                    self.config.max_length + 1, self.config.model_dim
                ),
            )
        self.feature_projection = (
            nn.Linear(self.config.feature_dim, self.config.model_dim)
            if self.config.condition_on_features
            else None
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.model_dim,
            nhead=self.config.heads,
            dim_feedforward=self.config.feedforward_dim,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerEncoder(layer, self.config.layers)
        self.output = nn.Linear(self.config.model_dim, VOCABULARY_SIZE)

    @staticmethod
    def _make_sinusoidal_positions(length: int, dimension: int) -> torch.Tensor:
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-np.log(10_000.0) / dimension)
        )
        encoding = torch.zeros(length, dimension, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * scale)
        if dimension > 1:
            encoding[:, 1::2] = torch.cos(positions * scale[: dimension // 2])
        return encoding

    def _features(self, features: torch.Tensor | None, batch_size: int) -> torch.Tensor:
        if not self.config.condition_on_features or self.feature_projection is None:
            raise RuntimeError("This generator is not configured for problem features.")
        if features is None:
            raise ValueError(
                "Continual ALDes mode requires one problem-feature vector per batch."
            )
        features = torch.as_tensor(
            features,
            dtype=self.feature_projection.weight.dtype,
            device=self.feature_projection.weight.device,
        )
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"Expected {self.config.feature_dim} problem features, "
                f"got {features.shape[-1]}."
            )
        if features.shape[0] == 1 and batch_size > 1:
            features = features.expand(batch_size, -1)
        if features.shape[0] != batch_size:
            raise ValueError("Feature batch size does not match the token batch.")
        return features

    def logits(
        self, features: torch.Tensor | None, tokens: torch.Tensor
    ) -> torch.Tensor:
        tokens = torch.as_tensor(
            tokens, dtype=torch.long, device=self.output.weight.device
        )
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, length).")
        batch_size, length = tokens.shape
        if length > self.config.max_length:
            raise ValueError("Token sequence exceeds max_length.")
        position_offset = 1 if self.config.condition_on_features else 0
        positions = torch.arange(
            position_offset,
            position_offset + length,
            device=tokens.device,
            dtype=torch.long,
        ).unsqueeze(0)
        if self.position_embedding is not None:
            positional = self.position_embedding(positions)
        else:
            positional = self.sinusoidal_positions[positions].to(
                device=tokens.device, dtype=self.token_embedding.weight.dtype
            )
        embedded = self.token_embedding(tokens) + positional
        if self.config.condition_on_features:
            conditioned = self._features(features, batch_size)
            feature_token = self.feature_projection(conditioned).unsqueeze(1)
            hidden = torch.cat((feature_token, embedded), dim=1)
        else:
            if features is not None:
                raise ValueError(
                    "Single-problem ALDes mode does not accept problem features."
                )
            hidden = embedded
        total_length = hidden.shape[1]
        causal_mask = torch.triu(
            torch.ones(
                total_length,
                total_length,
                device=tokens.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        decoded = self.decoder(hidden, mask=causal_mask)
        token_offset = 1 if self.config.condition_on_features else 0
        return self.output(decoded[:, token_offset:, :])

    @staticmethod
    def _grammar_mask(tokens: torch.Tensor) -> torch.Tensor:
        rows = [allowed_next_tokens(row.detach().cpu().numpy()) for row in tokens]
        return torch.as_tensor(np.stack(rows), dtype=torch.bool, device=tokens.device)

    def generate(
        self,
        features: torch.Tensor | None = None,
        *,
        candidates: int = 1,
        temperature: float = 1.0,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> GenerationResult:
        if candidates <= 0:
            raise ValueError("candidates must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        device = self.output.weight.device
        sequences = torch.full(
            (candidates, 1), BEGIN_INDEX, dtype=torch.long, device=device
        )
        if self.config.condition_on_features:
            features = self._features(features, candidates)
        elif features is not None:
            raise ValueError("Single-problem ALDes mode does not accept features.")
        finished = torch.zeros(candidates, dtype=torch.bool, device=device)
        selected_log_probs: list[torch.Tensor] = []
        selected_probs: list[torch.Tensor] = []

        for _ in range(self.config.max_length - 1):
            logits = self.logits(features, sequences)[:, -1, :] / temperature
            grammar = self._grammar_mask(sequences)
            grammar[finished, :] = False
            grammar[finished, END_INDEX] = True
            logits = logits.masked_fill(~grammar, -torch.inf)
            log_probs = torch.log_softmax(logits, dim=-1)
            probabilities = log_probs.exp()
            if greedy:
                selected = probabilities.argmax(dim=-1)
            else:
                selected = torch.multinomial(
                    probabilities, 1, generator=generator
                ).squeeze(1)
            selected_log_probs.append(log_probs.gather(1, selected[:, None]).squeeze(1))
            selected_probs.append(probabilities.gather(1, selected[:, None]).squeeze(1))
            sequences = torch.cat((sequences, selected[:, None]), dim=1)
            finished |= selected.eq(END_INDEX)
            if bool(finished.all()):
                break

        if not bool(finished.all()):
            raise RuntimeError(
                "ALDes generation reached max_length before all sequences ended."
            )
        return GenerationResult(
            sequences=sequences,
            log_probabilities=torch.stack(selected_log_probs, dim=1),
            probabilities=torch.stack(selected_probs, dim=1),
        )

    def score(
        self, features: torch.Tensor | None, sequences: torch.Tensor
    ) -> torch.Tensor:
        """Return grammar-conditioned log probability for each sequence."""

        sequences = torch.as_tensor(
            sequences, dtype=torch.long, device=self.output.weight.device
        )
        if sequences.ndim != 2 or sequences.shape[1] < 2:
            raise ValueError("sequences must have shape (batch, length>=2).")
        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]
        logits = self.logits(features, inputs)
        total = torch.zeros(sequences.shape[0], device=sequences.device)
        active = torch.ones(
            sequences.shape[0], dtype=torch.bool, device=sequences.device
        )
        for position in range(inputs.shape[1]):
            prefix = inputs[:, : position + 1]
            grammar = self._grammar_mask(prefix)
            restricted = logits[:, position, :].masked_fill(~grammar, -torch.inf)
            log_probs = torch.log_softmax(restricted, dim=-1)
            chosen = log_probs.gather(1, targets[:, position, None]).squeeze(1)
            total = total + torch.where(active, chosen, torch.zeros_like(chosen))
            active = active & targets[:, position].ne(END_INDEX)
        return total

    def save_checkpoint(self, path: str | Path, **metadata: Any) -> None:
        torch.save(
            {
                "schema": "autooptlib.aldes.generator",
                "schema_version": 2,
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
                "metadata": metadata,
            },
            Path(path),
        )

    @classmethod
    def load_checkpoint(
        cls, path: str | Path, *, map_location: Any = "cpu"
    ) -> tuple["ALDesGenerator", dict[str, Any]]:
        payload = torch.load(Path(path), map_location=map_location, weights_only=True)
        if payload.get("schema") != "autooptlib.aldes.generator":
            raise ValueError("Not an AutoOptLib ALDes generator checkpoint.")
        schema_version = payload.get("schema_version")
        if schema_version not in {1, 2}:
            raise ValueError("Unsupported AutoOptLib ALDes checkpoint version.")
        config = dict(payload["config"])
        if schema_version == 1:
            # Version 1 always used a feature token and learned positions.
            config.setdefault("condition_on_features", True)
            config.setdefault("position_encoding", "learned")
        model = cls(GeneratorConfig(**config))
        model.load_state_dict(payload["state_dict"])
        return model, dict(payload.get("metadata", {}))


__all__ = ["ALDesGenerator", "GenerationResult", "GeneratorConfig"]
