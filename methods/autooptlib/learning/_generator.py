"""Grammar-constrained autoregressive generator used by learning design."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class GeneratorConfig:
    model_dim: int = 64
    heads: int = 4
    layers: int = 4
    feedforward_dim: int = 256
    dropout: float = 0.1
    max_length: int = 50
    position_encoding: str = "sinusoidal"

    def __post_init__(self) -> None:
        integer_fields = (
            "model_dim",
            "heads",
            "layers",
            "feedforward_dim",
            "max_length",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and lie in [0, 1).")
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads.")
        if self.position_encoding not in {"sinusoidal", "learned"}:
            raise ValueError("position_encoding must be 'sinusoidal' or 'learned'.")


@dataclass
class GenerationResult:
    sequences: torch.Tensor
    log_probabilities: torch.Tensor
    probabilities: torch.Tensor


def _initialize_aldes_weights(module: nn.Module) -> None:
    """Apply the initialization used by the authors' ALDes implementation."""

    weight = getattr(module, "weight", None)
    if weight is not None and weight.dim() > 1:
        nn.init.kaiming_uniform_(weight.data)


class _ALDesLayerNorm(nn.Module):
    """ALDes layer normalization (epsilon and parameters match the source)."""

    def __init__(self, model_dim: int, epsilon: float = 1e-12) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(model_dim))
        self.beta = nn.Parameter(torch.zeros(model_dim))
        self.epsilon = float(epsilon)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(-1, keepdim=True)
        variance = values.var(-1, unbiased=False, keepdim=True)
        return (
            self.gamma * (values - mean) / torch.sqrt(variance + self.epsilon)
            + self.beta
        )


class _ALDesMultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention adapted from ALDes."""

    def __init__(self, model_dim: int, heads: int) -> None:
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(model_dim) // self.heads
        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.concatenate = nn.Linear(model_dim, model_dim)

    def _split(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, model_dim = values.shape
        return values.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self, values: torch.Tensor, causal_mask: torch.Tensor | None
    ) -> torch.Tensor:
        query = self._split(self.query(values))
        key = self._split(self.key(values))
        value = self._split(self.value(values))
        scores = query @ key.transpose(2, 3) / math.sqrt(self.head_dim)
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask[None, None, :, :], -10_000.0)
        attended = torch.softmax(scores, dim=-1) @ value
        batch, _heads, length, _head_dim = attended.shape
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch, length, self.heads * self.head_dim)
        )
        return self.concatenate(attended)


class _ALDesDecoderLayer(nn.Module):
    """The decoder block used in ALDes, including its disabled inner dropout."""

    def __init__(self, model_dim: int, feedforward_dim: int, heads: int) -> None:
        super().__init__()
        self.self_attention = _ALDesMultiHeadAttention(model_dim, heads)
        self.attention_norm = _ALDesLayerNorm(model_dim)
        self.feedforward_first = nn.Linear(model_dim, feedforward_dim)
        self.feedforward_second = nn.Linear(feedforward_dim, model_dim)
        self.feedforward_norm = _ALDesLayerNorm(model_dim)
        self.activation = nn.ReLU()

    def forward(
        self, values: torch.Tensor, causal_mask: torch.Tensor | None
    ) -> torch.Tensor:
        attended = self.self_attention(values, causal_mask)
        values = self.attention_norm(attended + values)
        transformed = self.feedforward_second(
            self.activation(self.feedforward_first(values))
        )
        return self.feedforward_norm(transformed + values)


class AutoregressiveGenerator(nn.Module):
    """Generate token sequences while applying a domain grammar mask."""

    def __init__(
        self,
        config: GeneratorConfig,
        *,
        vocabulary_size: int,
        begin_index: int,
        end_index: int,
        allowed_next: Callable[[np.ndarray], np.ndarray],
        generator_name: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocabulary_size = int(vocabulary_size)
        self.begin_index = int(begin_index)
        self.end_index = int(end_index)
        self.allowed_next = allowed_next
        self.generator_name = str(generator_name)
        if self.vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive.")
        if not 0 <= self.begin_index < self.vocabulary_size:
            raise ValueError("begin_index is outside the vocabulary.")
        if not 0 <= self.end_index < self.vocabulary_size:
            raise ValueError("end_index is outside the vocabulary.")
        self.token_embedding = nn.Embedding(self.vocabulary_size, self.config.model_dim)
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
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.decoder = nn.ModuleList(
            [
                _ALDesDecoderLayer(
                    self.config.model_dim,
                    self.config.feedforward_dim,
                    self.config.heads,
                )
                for _ in range(self.config.layers)
            ]
        )
        self.output = nn.Linear(self.config.model_dim, self.vocabulary_size)
        self.apply(_initialize_aldes_weights)

    def _token_tensor(self, tokens: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(tokens)
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if values.dtype not in integer_dtypes:
            raise ValueError("tokens must contain integer token indices.")
        if values.numel() and (
            bool((values < 0).any()) or bool((values >= self.vocabulary_size).any())
        ):
            raise ValueError("token index is outside the vocabulary.")
        return values.to(dtype=torch.long, device=self.output.weight.device)

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

    def hidden_states(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return decoder states before the token-output projection.

        Exposing the representation keeps downstream predictors separate from
        the generator's policy head.  It does not change V1 generation or PPO
        semantics; :meth:`logits` remains the public policy operation.
        """

        tokens = self._token_tensor(tokens)
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, length).")
        _batch_size, length = tokens.shape
        if length > self.config.max_length:
            raise ValueError("Token sequence exceeds max_length.")
        positions = torch.arange(
            0,
            length,
            device=tokens.device,
            dtype=torch.long,
        ).unsqueeze(0)
        if self.position_embedding is not None:
            positional = self.position_embedding(positions)
        else:
            positional = self.sinusoidal_positions[positions].to(
                device=tokens.device, dtype=self.token_embedding.weight.dtype
            )
        hidden = self.embedding_dropout(self.token_embedding(tokens) + positional)
        causal_mask = torch.triu(
            torch.ones(length, length, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        for layer in self.decoder:
            hidden = layer(hidden, causal_mask)
        return hidden

    def logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden_states(tokens))

    def _grammar_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        rows = [self.allowed_next(row.detach().cpu().numpy()) for row in tokens]
        return torch.as_tensor(np.stack(rows), dtype=torch.bool, device=tokens.device)

    def generate(
        self,
        *,
        candidates: int = 1,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> GenerationResult:
        if type(candidates) is not int or candidates <= 0:
            raise ValueError("candidates must be a positive integer.")
        if type(greedy) is not bool:
            raise ValueError("greedy must be boolean.")
        device = self.output.weight.device
        sequences = torch.full(
            (candidates, 1), self.begin_index, dtype=torch.long, device=device
        )
        finished = torch.zeros(candidates, dtype=torch.bool, device=device)
        selected_log_probs: list[torch.Tensor] = []
        selected_probs: list[torch.Tensor] = []
        for _ in range(self.config.max_length - 1):
            logits = self.logits(sequences)[:, -1, :]
            grammar = self._grammar_mask(sequences)
            grammar[finished, :] = False
            grammar[finished, self.end_index] = True
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
            finished |= selected.eq(self.end_index)
            if bool(finished.all()):
                break
        if not bool(finished.all()):
            raise RuntimeError(
                f"{self.generator_name} generation reached max_length before all "
                "sequences ended."
            )
        return GenerationResult(
            sequences=sequences,
            log_probabilities=torch.stack(selected_log_probs, dim=1),
            probabilities=torch.stack(selected_probs, dim=1),
        )

    def entropy(self, sequences: torch.Tensor) -> torch.Tensor:
        """Return mean active-token entropy for each supplied sequence."""

        sequences = self._token_tensor(sequences)
        if sequences.ndim != 2 or sequences.shape[1] < 2:
            raise ValueError("sequences must have shape (batch, length>=2).")
        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]
        logits = self.logits(inputs)
        total = torch.zeros(sequences.shape[0], device=sequences.device)
        counts = torch.zeros(sequences.shape[0], device=sequences.device)
        active = torch.ones(
            sequences.shape[0], dtype=torch.bool, device=sequences.device
        )
        for position in range(inputs.shape[1]):
            grammar = self._grammar_mask(inputs[:, : position + 1])
            restricted = logits[:, position, :].masked_fill(~grammar, -torch.inf)
            probabilities = torch.softmax(restricted, dim=-1)
            log_probabilities = torch.log_softmax(restricted, dim=-1)
            finite_log_probabilities = torch.nan_to_num(log_probabilities, neginf=0.0)
            token_entropy = -(probabilities * finite_log_probabilities).sum(dim=-1)
            total = total + torch.where(
                active, token_entropy, torch.zeros_like(token_entropy)
            )
            counts = counts + active.to(counts.dtype)
            active = active & targets[:, position].ne(self.end_index)
        return total / counts.clamp_min(1)

    def score(self, sequences: torch.Tensor) -> torch.Tensor:
        """Return grammar-conditioned log probability for each sequence."""

        sequences = self._token_tensor(sequences)
        if sequences.ndim != 2 or sequences.shape[1] < 2:
            raise ValueError("sequences must have shape (batch, length>=2).")
        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]
        logits = self.logits(inputs)
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
            active = active & targets[:, position].ne(self.end_index)
        return total


__all__ = [
    "AutoregressiveGenerator",
    "GenerationResult",
    "GeneratorConfig",
]
