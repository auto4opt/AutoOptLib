"""Random-state persistence shared by learning training checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import torch


def atomic_text_write(path: str | Path, text: str) -> Path:
    """Replace a UTF-8 text file atomically with an isolated temporary file."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return target


def atomic_torch_save(payload: Any, path: str | Path) -> Path:
    """Write a torch payload atomically without sharing a temporary filename."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return target


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    mps_backend = getattr(torch.backends, "mps", None)
    mps = getattr(torch, "mps", None)
    if (
        mps_backend is not None
        and mps_backend.is_available()
        and mps is not None
        and hasattr(mps, "get_rng_state")
    ):
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    # map_location may have moved byte tensors onto the model's accelerator.
    torch.set_rng_state(state["cpu"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    mps_backend = getattr(torch.backends, "mps", None)
    mps = getattr(torch, "mps", None)
    if (
        "mps" in state
        and mps_backend is not None
        and mps_backend.is_available()
        and mps is not None
        and hasattr(mps, "set_rng_state")
    ):
        mps.set_rng_state(state["mps"].cpu())


__all__ = [
    "atomic_text_write",
    "atomic_torch_save",
    "capture_rng_state",
    "restore_rng_state",
]
