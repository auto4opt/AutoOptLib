"""Crash-safe append-only records and reproducibility metadata."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(repository: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            return subprocess.check_output(
                arguments,
                cwd=repository,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def environment_record(repository: Path) -> dict[str, Any]:
    packages = {}
    for name in ("autooptlib", "numpy", "ioh", "cma", "torch", "scipy", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "created_utc": utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "packages": packages,
        "git": _git(repository),
    }


class JsonlStore:
    """One locked JSON object per line; safe to resume after interruption."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at {self.path}:{line_number}."
                    ) from exc
                if not isinstance(value, dict):
                    raise TypeError("Every JSONL record must be an object.")
                records.append(value)
        return records

    def completed_ids(self, *, include_failed: bool = True) -> set[str]:
        statuses = {"ok", "failed"} if include_failed else {"ok"}
        return {
            str(record["task_id"])
            for record in self.latest_records()
            if record.get("status") in statuses and "task_id" in record
        }

    def latest_records(self) -> list[dict[str, Any]]:
        """Return the last append-only record for every task identifier.

        Retried failures remain in the JSONL audit trail, while publication
        summaries and resume checks treat the latest attempt as authoritative.
        Records without a task ID are retained individually.
        """

        latest: dict[str, dict[str, Any]] = {}
        unkeyed: list[dict[str, Any]] = []
        for record in self.records():
            if "task_id" not in record:
                unkeyed.append(record)
                continue
            latest[str(record["task_id"])] = record
        return [*latest.values(), *unkeyed]

    def append(self, record: dict[str, Any]) -> None:
        encoded = (
            json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n"
        )
        with self.path.open("a", encoding="utf-8") as handle:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except ImportError:  # pragma: no cover - Windows server only
                fcntl = None
            try:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_json(path: str | Path, document: dict[str, Any]) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


__all__ = ["JsonlStore", "atomic_json", "environment_record", "utc_now"]
