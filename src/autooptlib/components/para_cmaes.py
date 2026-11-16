"""Wrapper for CMA-ES parameter update (delegates to para_cma)."""

from __future__ import annotations

from typing import Any

from .para_cma import para_cma


def para_cmaes(*args: Any):
    return para_cma(*args)
