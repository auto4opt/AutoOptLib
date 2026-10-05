"""Leak-safe shared NumPy arrays used by process evaluation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SharedArraySpec:
    name: str
    shape: tuple[int, ...]
    dtype: str

    def attach(self) -> tuple[shared_memory.SharedMemory, np.ndarray]:
        memory = shared_memory.SharedMemory(name=self.name)
        array = np.ndarray(self.shape, dtype=np.dtype(self.dtype), buffer=memory.buf)
        return memory, array


class SharedArray:
    """Owner of one reusable shared-memory backed NumPy array."""

    def __init__(self, shape: Sequence[int], dtype: np.dtype | str = np.float64):
        resolved_shape = tuple(int(value) for value in shape)
        resolved_dtype = np.dtype(dtype)
        size = max(1, int(np.prod(resolved_shape, dtype=np.int64)))
        self._memory = shared_memory.SharedMemory(
            create=True, size=size * resolved_dtype.itemsize
        )
        self.array = np.ndarray(
            resolved_shape, dtype=resolved_dtype, buffer=self._memory.buf
        )
        self.array.fill(0)
        self._closed = False

    @property
    def spec(self) -> SharedArraySpec:
        return SharedArraySpec(
            name=self._memory.name,
            shape=tuple(self.array.shape),
            dtype=self.array.dtype.str,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._memory.close()
        try:
            self._memory.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> "SharedArray":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            if hasattr(self, "_closed"):
                self.close()
        except Exception:
            pass
