"""Load an explicitly calibrated, machine/build-specific native data cost model.

There is no import-time benchmarking, graph JIT, or global numerical cache.
The AOL_* variables below are research controls for independent ablations.
Normal execution uses graph planning with a compatible installed profile, and
preserves the existing dense path when calibration is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from functools import lru_cache, partial
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def machine_key():
    cpu = ""
    if sys.platform == "darwin":
        # Query the same OS identity without starting a subprocess per process.
        import ctypes

        size = ctypes.c_size_t()
        sysctl = ctypes.CDLL(None).sysctlbyname
        name = b"machdep.cpu.brand_string"
        if sysctl(name, None, ctypes.byref(size), None, 0) == 0:
            value = ctypes.create_string_buffer(size.value)
            if sysctl(name, value, ctypes.byref(size), None, 0) == 0:
                cpu = value.value.decode()
            else:
                raise OSError("Cannot read CPU identity")
        else:
            raise OSError("Cannot read CPU identity")
    elif Path("/proc/cpuinfo").is_file():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            None,
        )
        if cpu is None:
            cpu = platform.processor()
    else:
        cpu = platform.processor()
    return dict(
        system=platform.system(),
        machine=platform.machine(),
        cpu=cpu,
        numpy=np.__version__,
        python=list(sys.version_info[:2]),
        abi=3,
    )


@lru_cache(maxsize=4)
def _binary_key(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile_key(library):
    return dict(machine_key(), binary_sha256=_binary_key(library.__file__))


def profile_path():
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "autooptlib" / "data-plan-costs.json"


@lru_cache(maxsize=8)
def _read_profile(path, mtime_ns, size):
    return json.loads(Path(path).read_text())


def load_profile(library, path=None):
    return _load_profile_at(
        library, Path(path) if path is not None else profile_path(), path is not None
    )


def _load_profile_at(library, target, explicit):
    target = Path(target)
    if not target.is_file():
        if explicit:
            raise FileNotFoundError(f"Data-plan cost profile not found: {target}")
        return None
    stat = target.stat()
    model = _read_profile(str(target.resolve()), stat.st_mtime_ns, stat.st_size)
    if model.get("key") != profile_key(library):
        if explicit:
            raise ValueError(
                "Data-plan profile does not match this CPU/NumPy/native build"
            )
        return None
    return model


@lru_cache(maxsize=8)
def _session_model(library, target, explicit):
    # Profiles are immutable for a process until reload_profiles/install_profile.
    # Resolve and inspect the file once, not once per independent solve.
    model = _load_profile_at(library, target, explicit)
    if model is None:
        return None
    return library.DataCostModel(model)


def reload_profiles():
    """Refresh the cost tables used by subsequently constructed sessions.

    Active sessions retain their immutable table. No population or execution
    state is cached, and changing a profile cannot affect a running algorithm.
    """
    _session_model.cache_clear()
    _read_profile.cache_clear()


def configure_session(session, library):
    mode = os.environ.get("AOL_DATA_PLAN_MODE", "graph")
    dump = os.environ.get("AOL_DATA_PLAN_TRACE") == "1"
    if mode != "graph":
        session.configure_data_plan(mode, dump=dump)
        return
    configured = os.environ.get("AOL_DATA_PLAN_PROFILE")
    target = (
        os.path.abspath(configured) if configured is not None else str(profile_path())
    )
    loader = partial(_session_model, library, target, configured is not None)
    if not session.data_plan_eligible:
        # Admission already read these names. Do not read future graph getters
        # or load a table for a dense-only graph; a later legal name change can
        # request the same pinned profile at its original execution boundary.
        session._defer_data_plan_costs(loader, dump=dump)
        return
    session.configure_data_plan(mode, loader(), dump=dump)


def install_profile(model, library, path=None):
    """Install a separately calibrated profile atomically, never recalibrate."""
    if model.get("key") != profile_key(library):
        raise ValueError("Cannot install a profile for another CPU/NumPy/native build")
    target = Path(path) if path is not None else profile_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(model, indent=2) + "\n")
    temporary.replace(target)
    reload_profiles()
    return target
