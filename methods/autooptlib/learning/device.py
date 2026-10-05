"""Portable accelerator selection for learning-based design."""

from __future__ import annotations

import torch


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Select CUDA/ROCm, Apple MPS, or CPU in that priority order."""

    supplied_device = isinstance(requested, torch.device)
    name = str(requested).strip().lower()
    if name in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name in {"amd", "rocm", "hip"}:
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError(
                "AMD GPU acceleration requires a ROCm-enabled PyTorch build."
            )
        return torch.device("cuda:0")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA/ROCm was requested, but this PyTorch installation cannot "
            "access a compatible GPU."
        )
    if device.type == "mps" and not _mps_available():
        raise RuntimeError(
            "MPS was requested, but this PyTorch installation or Mac does "
            "not provide the MPS backend."
        )
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(
            "LearningDevice must be auto, cpu, cuda, cuda:N, mps, or rocm."
        )
    if (
        not supplied_device
        and device.type in {"cpu", "mps"}
        and device.index is not None
    ):
        raise ValueError(f"{device.type} devices cannot include an index.")
    if supplied_device and device.type in {"cpu", "mps"}:
        # PyTorch may report an active MPS device as ``mps:0``. Canonicalize
        # framework-owned objects while keeping user configuration strict.
        device = torch.device(device.type)
    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        if not 0 <= index < torch.cuda.device_count():
            raise RuntimeError(f"CUDA/ROCm device index {index} is unavailable.")
    return device


def describe_device(device: torch.device) -> str:
    """Return a concise human-readable accelerator description."""

    if device.type == "cuda":
        backend = "AMD ROCm" if torch.version.hip is not None else "NVIDIA CUDA"
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        return f"{backend}: {torch.cuda.get_device_name(index)}"
    if device.type == "mps":
        return "Apple MPS"
    return "CPU"


__all__ = ["describe_device", "resolve_device"]
