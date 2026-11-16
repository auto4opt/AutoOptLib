"""Maintain a tabu archive by copying the latest solutions."""

from __future__ import annotations

from typing import Any


def archive_tabu(*args: Any):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        return solution, None

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["", ""], None

    raise ValueError(f"Unsupported mode: {mode}")
