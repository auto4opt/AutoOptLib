"""Portable CPU topology and worker resource controls."""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import os
import platform
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class NumaNode:
    """One NUMA node and the CPUs visible to the current process."""

    node_id: int
    cpu_ids: tuple[int, ...]


@dataclass(frozen=True)
class CpuTopology:
    """One process-visible logical CPU and its physical topology identity."""

    cpu_id: int
    node_id: int
    socket_id: int
    core_id: int
    thread_index: int

    @property
    def physical_core(self) -> tuple[int, int]:
        return self.socket_id, self.core_id


def available_cpu_ids() -> tuple[int, ...]:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is not None:
        try:
            return tuple(sorted(int(value) for value in getter(0)))
        except OSError:
            pass
    return tuple(range(max(1, os.cpu_count() or 1)))


def _linux_numa_nodes(cpu_ids: Iterable[int]) -> list[NumaNode]:
    wanted = set(int(value) for value in cpu_ids)
    root = Path("/sys/devices/system/node")
    groups: list[NumaNode] = []
    if not root.exists():
        return groups
    for node in sorted(root.glob("node[0-9]*")):
        path = node / "cpulist"
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        values: set[int] = set()
        for token in text.split(","):
            if not token:
                continue
            if "-" in token:
                lower, upper = token.split("-", 1)
                values.update(range(int(lower), int(upper) + 1))
            else:
                values.add(int(token))
        selected = sorted(values & wanted)
        if selected:
            try:
                node_id = int(node.name.removeprefix("node"))
            except ValueError:
                continue
            groups.append(NumaNode(node_id, tuple(selected)))
    return groups


def discover_numa_nodes() -> tuple[NumaNode, ...]:
    """Return the process-visible topology, with a portable single-node fallback."""

    ids = available_cpu_ids()
    return tuple(_linux_numa_nodes(ids) or [NumaNode(0, ids)])


def _read_topology_integer(cpu_id: int, name: str) -> int | None:
    path = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/topology/{name}")
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def discover_cpu_topology() -> tuple[CpuTopology, ...]:
    """Describe visible CPUs, including SMT siblings, without external tools."""

    ids = available_cpu_ids()
    nodes = discover_numa_nodes()
    cpu_to_node = {cpu: node.node_id for node in nodes for cpu in node.cpu_ids}
    raw: list[tuple[int, int, int, int]] = []
    for cpu_id in ids:
        node_id = cpu_to_node.get(cpu_id, 0)
        socket_id = _read_topology_integer(cpu_id, "physical_package_id")
        core_id = _read_topology_integer(cpu_id, "core_id")
        # Portable fallback: treating logical CPUs as distinct physical cores
        # is conservative for binding and preserves previous behaviour.
        raw.append(
            (
                cpu_id,
                node_id,
                node_id if socket_id is None else socket_id,
                cpu_id if core_id is None else core_id,
            )
        )
    siblings: dict[tuple[int, int], list[int]] = {}
    for cpu_id, _node_id, socket_id, core_id in raw:
        siblings.setdefault((socket_id, core_id), []).append(cpu_id)
    indices = {
        cpu_id: index
        for values in siblings.values()
        for index, cpu_id in enumerate(sorted(values))
    }
    return tuple(
        CpuTopology(cpu_id, node_id, socket_id, core_id, indices[cpu_id])
        for cpu_id, node_id, socket_id, core_id in raw
    )


def physical_cpu_ids() -> tuple[int, ...]:
    """Return one visible logical CPU for each physical core."""

    return tuple(
        item.cpu_id for item in discover_cpu_topology() if item.thread_index == 0
    )


def _cpu_model_identifier() -> str:
    value = platform.processor().strip()
    if value:
        return value
    cpuinfo = Path("/proc/cpuinfo")
    try:
        lines = cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    fields = {
        key.strip().lower(): item.strip()
        for line in lines
        if ":" in line
        for key, item in [line.split(":", 1)]
    }
    for key in ("model name", "hardware", "cpu model"):
        if fields.get(key):
            return fields[key]
    return platform.machine()


def hardware_topology_signature() -> str:
    """Stable key preventing concurrency profiles crossing incompatible CPUs."""

    topology = discover_cpu_topology()
    node_shapes = []
    for node_id in sorted({item.node_id for item in topology}):
        items = [item for item in topology if item.node_id == node_id]
        cores: dict[tuple[int, int], int] = {}
        for item in items:
            cores[item.physical_core] = cores.get(item.physical_core, 0) + 1
        node_shapes.append(
            {
                "logical_cpus": len(items),
                "physical_cores": len(cores),
                "threads_per_core": sorted(cores.values()),
            }
        )
    payload = {
        "machine": platform.machine(),
        "processor": _cpu_model_identifier(),
        # Normalize away Slurm CPU and NUMA identifiers so equivalent sockets
        # can reuse a profile even when different shards receive node0/node3.
        "nodes": sorted(
            node_shapes,
            key=lambda value: (
                value["logical_cpus"],
                value["physical_cores"],
                value["threads_per_core"],
            ),
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def available_memory_bytes() -> int:
    """Best-effort currently available physical memory."""

    meminfo = Path("/proc/meminfo")
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        pass
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return max(1, pages * page_size)
    except (AttributeError, OSError, ValueError):
        return 2**63 - 1


def partition_cpu_ids(workers: int, cores_per_worker: int) -> list[tuple[int, ...]]:
    ids = available_cpu_ids()
    nodes = discover_numa_nodes()
    topology = discover_cpu_topology()
    by_node = {
        node.node_id: tuple(
            item.cpu_id
            for item in sorted(
                (item for item in topology if item.node_id == node.node_id),
                key=lambda item: (
                    item.thread_index,
                    item.socket_id,
                    item.core_id,
                    item.cpu_id,
                ),
            )
        )
        for node in nodes
    }
    node_chunks = [
        [
            tuple(by_node[node.node_id][i : i + cores_per_worker])
            for i in range(0, len(by_node[node.node_id]), cores_per_worker)
        ]
        for node in nodes
    ]
    groups: list[tuple[int, ...]] = []
    # Distribute workers between NUMA nodes, while keeping each worker's cores
    # local to one node whenever the topology permits it.
    positions = [0] * len(node_chunks)
    while len(groups) < workers:
        progressed = False
        for index, chunks in enumerate(node_chunks):
            if positions[index] >= len(chunks):
                continue
            selected = chunks[positions[index]]
            positions[index] += 1
            if len(selected) == cores_per_worker or cores_per_worker == 1:
                groups.append(selected)
                progressed = True
                if len(groups) == workers:
                    break
        if not progressed:
            break
    if len(groups) < workers:
        # Small NUMA-node tails may not form a complete local group. Use the
        # remaining available CPUs before falling back to a repeated binding.
        used = {cpu for group in groups for cpu in group}
        ordered = [
            item.cpu_id
            for item in sorted(
                topology,
                key=lambda item: (
                    item.thread_index,
                    item.node_id,
                    item.socket_id,
                    item.core_id,
                    item.cpu_id,
                ),
            )
        ]
        remaining = [cpu for cpu in ordered if cpu not in used]
        while len(groups) < workers and remaining:
            selected = tuple(remaining[:cores_per_worker])
            del remaining[:cores_per_worker]
            groups.append(selected)
    while len(groups) < workers:
        groups.append((ids[len(groups) % len(ids)],))
    return groups


def bind_current_process(cpu_ids: Iterable[int]) -> bool:
    setter = getattr(os, "sched_setaffinity", None)
    if setter is None:
        return False
    try:
        setter(0, set(int(value) for value in cpu_ids))
    except OSError:
        return False
    return True


def bind_numa_memory_policy(node_id: int | None) -> bool:
    """Request node-local allocation before a worker constructs resident data.

    libnuma is optional. CPU affinity remains active when the library or kernel
    policy is unavailable, so this function is deliberately best effort.
    """

    if node_id is None or platform.system() != "Linux":
        return False
    if len(discover_numa_nodes()) <= 1:
        return False
    library_name = ctypes.util.find_library("numa")
    if not library_name:
        return False
    try:
        library = ctypes.CDLL(library_name, use_errno=True)
        library.numa_available.restype = ctypes.c_int
        if library.numa_available() < 0:
            return False
        library.numa_set_preferred.argtypes = [ctypes.c_int]
        library.numa_set_preferred.restype = None
        # First-touch allocations made while unpickling/initializing the worker
        # now prefer memory from its home node whenever capacity permits. CPU
        # affinity remains independently controlled by EvalAffinity.
        library.numa_set_preferred(int(node_id))
    except (AttributeError, OSError):
        return False
    return True


@contextmanager
def limit_native_threads(count: int) -> Iterator[object]:
    """Limit BLAS/OpenMP threads when threadpoolctl is installed."""

    count = max(1, int(count))
    names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = str(count)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        limiter = None
    else:
        limiter = threadpool_limits(limits=count)
    try:
        if limiter is None:
            yield None
        else:
            with limiter:
                yield limiter
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
