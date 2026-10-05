"""Shared-memory transport for large NumPy arrays nested in Python objects."""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import cloudpickle
import numpy as np

_TAG = "autooptlib.shared-ndarray.v1"


@dataclass(frozen=True)
class SharedNDArraySpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    order: str
    writeable: bool
    index: int


@dataclass
class SharedPickle:
    blob: bytes
    memories: list[shared_memory.SharedMemory]
    shared_bytes: int

    @property
    def transfer_bytes(self) -> int:
        return len(self.blob) + self.shared_bytes

    def close(self, *, unlink: bool) -> None:
        for memory in self.memories:
            try:
                memory.close()
            except FileNotFoundError:
                pass
            if unlink:
                try:
                    memory.unlink()
                except FileNotFoundError:
                    pass
        self.memories.clear()


class _SharedCloudPickler(cloudpickle.CloudPickler):
    def __init__(self, file, *, threshold: int):
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self.threshold = max(0, int(threshold))
        self.memories: list[shared_memory.SharedMemory] = []
        self.shared_bytes = 0
        self._memo: dict[int, SharedNDArraySpec] = {}

    def persistent_id(self, obj: Any):
        if not isinstance(obj, np.ndarray):
            return None
        if obj.dtype.hasobject or obj.nbytes < self.threshold or obj.nbytes == 0:
            return None
        existing = self._memo.get(id(obj))
        if existing is not None:
            return (_TAG, existing)
        order = "F" if obj.flags.f_contiguous and not obj.flags.c_contiguous else "C"
        contiguous = (
            np.asfortranarray(obj) if order == "F" else np.ascontiguousarray(obj)
        )
        memory = shared_memory.SharedMemory(create=True, size=contiguous.nbytes)
        target = np.ndarray(
            contiguous.shape,
            dtype=contiguous.dtype,
            buffer=memory.buf,
            order=order,
        )
        target[...] = contiguous
        spec = SharedNDArraySpec(
            name=memory.name,
            shape=tuple(contiguous.shape),
            dtype=contiguous.dtype.str,
            order=order,
            writeable=bool(obj.flags.writeable),
            index=len(self.memories),
        )
        self.memories.append(memory)
        self.shared_bytes += contiguous.nbytes
        self._memo[id(obj)] = spec
        return (_TAG, spec)


class _SharedUnpickler(pickle.Unpickler):
    def __init__(self, file, *, copy_arrays: bool, unlink: bool):
        super().__init__(file)
        self.copy_arrays = copy_arrays
        self.unlink = unlink
        self.memories: list[shared_memory.SharedMemory] = []
        self._arrays: dict[int, np.ndarray] = {}

    def persistent_load(self, pid):
        tag, spec = pid
        if tag != _TAG or not isinstance(spec, SharedNDArraySpec):
            raise pickle.UnpicklingError(f"Unsupported persistent object {tag!r}.")
        if spec.index in self._arrays:
            return self._arrays[spec.index]
        memory = shared_memory.SharedMemory(name=spec.name)
        view = np.ndarray(
            spec.shape,
            dtype=np.dtype(spec.dtype),
            buffer=memory.buf,
            order=spec.order,
        )
        if self.copy_arrays:
            array = np.array(view, copy=True, order=spec.order)
            memory.close()
            if self.unlink:
                try:
                    memory.unlink()
                except FileNotFoundError:
                    pass
        else:
            array = view
            self.memories.append(memory)
        if not spec.writeable:
            array.setflags(write=False)
        self._arrays[spec.index] = array
        return array


def dumps_shared(value: Any, threshold: int) -> SharedPickle:
    stream = io.BytesIO()
    pickler = _SharedCloudPickler(stream, threshold=threshold)
    try:
        pickler.dump(value)
    except BaseException:
        payload = SharedPickle(b"", pickler.memories, pickler.shared_bytes)
        payload.close(unlink=True)
        raise
    return SharedPickle(stream.getvalue(), pickler.memories, pickler.shared_bytes)


def loads_shared(
    blob: bytes, *, copy_arrays: bool = False, unlink: bool = False
) -> tuple[Any, list[shared_memory.SharedMemory]]:
    unpickler = _SharedUnpickler(
        io.BytesIO(blob), copy_arrays=copy_arrays, unlink=unlink
    )
    try:
        value = unpickler.load()
    except BaseException:
        for memory in unpickler.memories:
            memory.close()
        raise
    return value, unpickler.memories


__all__ = ["SharedNDArraySpec", "SharedPickle", "dumps_shared", "loads_shared"]
