"""Online worker-count selection without duplicate objective evaluations."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable

from .scheduler import TaskObservation


@dataclass(frozen=True)
class ConcurrencyTrial:
    """One pilot wave executed at a fixed worker limit."""

    workers: int
    tasks: int
    wall_seconds: float
    task_seconds: float
    cpu_seconds: float
    mean_cpu_ratio: float

    @property
    def throughput(self) -> float:
        return self.tasks / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def selection_score(self) -> float:
        """Completed black-box evaluations per wall second."""

        return self.throughput

    def as_dict(self) -> dict[str, object]:
        return {
            "workers": self.workers,
            "tasks": self.tasks,
            "wall_seconds": self.wall_seconds,
            "task_seconds": self.task_seconds,
            "cpu_seconds": self.cpu_seconds,
            "mean_cpu_ratio": self.mean_cpu_ratio,
            "throughput_tasks_per_second": self.throughput,
            "selection_score": self.selection_score,
            "selection_work": "completed_tasks",
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ConcurrencyTrial":
        return cls(
            workers=int(value["workers"]),
            tasks=int(value["tasks"]),
            wall_seconds=float(value["wall_seconds"]),
            task_seconds=float(value["task_seconds"]),
            cpu_seconds=float(value["cpu_seconds"]),
            mean_cpu_ratio=float(value["mean_cpu_ratio"]),
        )


@dataclass
class ConcurrencyProfile:
    """Reusable worker-count observations for one workload signature."""

    selected_workers: int = 1
    maps: int = 0
    trials: list[ConcurrencyTrial] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_workers": self.selected_workers,
            "maps": self.maps,
            "trials": [trial.as_dict() for trial in self.trials],
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ConcurrencyProfile":
        trials = value.get("trials", [])
        return cls(
            selected_workers=max(1, int(value.get("selected_workers", 1))),
            maps=max(0, int(value.get("maps", 0))),
            trials=[
                ConcurrencyTrial.from_dict(item)
                for item in trials
                if isinstance(item, dict)
            ],
        )


class AdaptiveConcurrencyController:
    """Explore legal worker counts on real tasks, then reuse the best limit.

    Pilot waves consume distinct tasks from the submitted batch.  No task is
    evaluated twice and result ordering remains the executor's responsibility.
    """

    def __init__(
        self,
        max_workers: int,
        *,
        preferred_workers: int | None = None,
        profile_path: str | os.PathLike[str] | None = None,
        hardware_signature: str = "portable",
        minimum_gain: float = 0.05,
        retune_interval: int = 8,
    ) -> None:
        self.max_workers = max(1, int(max_workers))
        self.preferred_workers = max(
            1,
            min(
                self.max_workers,
                int(preferred_workers or self.max_workers),
            ),
        )
        self.profile_path = (
            None if profile_path is None else Path(profile_path).expanduser()
        )
        self.hardware_signature = str(hardware_signature)
        self.minimum_gain = max(0.0, min(0.5, float(minimum_gain)))
        self.retune_interval = max(1, int(retune_interval))
        self._profiles: dict[str, ConcurrencyProfile] = {}
        self._signature: str | None = None
        self._profile: ConcurrencyProfile | None = None
        self._levels: list[int] = []
        self._level_index = 0
        self._exploring = False
        self._active_limit = self.max_workers
        self._target = 0
        self._dispatched = 0
        self._completed = 0
        self._task_seconds = 0.0
        self._cpu_seconds = 0.0
        self._cpu_ratios: list[float] = []
        self._stage_started = 0.0
        self._accounted_at = 0.0
        self._map_capacity_seconds = 0.0
        self._map_trials = 0
        self._map_max_workers = self.max_workers
        self._load_profiles()

    @staticmethod
    def _candidate_levels(
        max_workers: int, preferred_workers: int | None = None
    ) -> list[int]:
        preferred = max(1, min(max_workers, int(preferred_workers or max_workers)))
        if max_workers > 8:
            # Large machines should not spend an entire first map walking
            # through every power of two. Probe one baseline, the physical-core
            # neighbourhood, and the SMT ceiling when enough tasks exist.
            return sorted(
                {
                    1,
                    max(1, preferred // 2),
                    preferred,
                    max_workers,
                }
            )
        levels = [1]
        value = 2
        while value < max_workers:
            levels.append(value)
            value *= 2
        if max_workers > 1 and levels[-1] != max_workers:
            levels.append(max_workers)
        return levels

    def _levels_that_fit(self, jobs: int) -> list[int]:
        # Preserve at least one task for exploitation after the pilot waves.
        levels: list[int] = []
        used = 0
        maximum = min(self._map_max_workers, jobs)
        preferred = min(self.preferred_workers, maximum)
        for level in self._candidate_levels(maximum, preferred):
            if used + level >= jobs:
                break
            levels.append(level)
            used += level
        return levels

    def begin_map(
        self,
        signature: Hashable,
        jobs: int,
        *,
        max_workers: int | None = None,
    ) -> int:
        if jobs <= 0:
            raise ValueError("jobs must be positive")
        now = time.monotonic()
        self._map_max_workers = max(
            1,
            min(self.max_workers, int(max_workers or self.max_workers), jobs),
        )
        self._signature = repr(signature)
        self._profile = self._profiles.setdefault(self._signature, ConcurrencyProfile())
        self._profile.maps += 1
        self._map_capacity_seconds = 0.0
        self._map_trials = 0
        self._accounted_at = now
        self._levels = self._levels_that_fit(jobs)
        should_retune = (
            not self._profile.trials
            or (self._profile.maps - 1) % self.retune_interval == 0
        )
        if should_retune and len(self._levels) >= 2:
            self._level_index = 0
            self._begin_trial(self._levels[0], now)
        else:
            self._exploring = False
            learned = self._profile.selected_workers if self._profile.trials else jobs
            self._active_limit = min(self._map_max_workers, jobs, max(1, learned))
            self._target = 0
            self._reset_stage(now)
        return self._active_limit

    def _reset_stage(self, now: float) -> None:
        self._dispatched = 0
        self._completed = 0
        self._task_seconds = 0.0
        self._cpu_seconds = 0.0
        self._cpu_ratios = []
        self._stage_started = now

    def _account_capacity(self, now: float) -> None:
        if self._accounted_at > 0:
            self._map_capacity_seconds += max(0.0, now - self._accounted_at) * max(
                1, self._active_limit
            )
        self._accounted_at = now

    def _begin_trial(self, workers: int, now: float) -> None:
        self._account_capacity(now)
        self._exploring = True
        self._active_limit = max(1, min(self._map_max_workers, workers))
        self._target = self._active_limit
        self._reset_stage(now)

    @property
    def active_limit(self) -> int:
        return self._active_limit

    @property
    def selected_workers(self) -> int:
        if self._profile is None:
            return self._active_limit
        return min(
            self._map_max_workers,
            max(1, self._profile.selected_workers),
        )

    @property
    def map_capacity_seconds(self) -> float:
        return self._map_capacity_seconds

    @property
    def map_trials(self) -> int:
        return self._map_trials

    @property
    def exploring(self) -> bool:
        return self._exploring

    def dispatch_slots(self, running: int) -> int:
        slots = max(0, self._active_limit - max(0, int(running)))
        if self._exploring:
            slots = min(slots, max(0, self._target - self._dispatched))
        return slots

    def note_dispatch(self) -> None:
        if self._exploring:
            self._dispatched += 1

    def note_workers_ready(self) -> None:
        """Exclude one-time lazy process startup from a steady-state pilot.

        Startup remains in the map's wall/capacity accounting, but a worker
        count is selected from objective-task execution rather than from how
        quickly the operating system happened to spawn that pilot's workers.
        """

        if not self._exploring or self._dispatched or self._completed:
            return
        now = time.monotonic()
        self._account_capacity(now)
        self._reset_stage(now)

    def note_completion(self, observation: TaskObservation | None) -> None:
        if not self._exploring:
            return
        self._completed += 1
        if observation is not None:
            self._task_seconds += max(0.0, observation.wall_seconds)
            self._cpu_seconds += max(0.0, observation.cpu_seconds)
            self._cpu_ratios.append(observation.cpu_ratio)

    def _record_trial(self, now: float) -> None:
        assert self._profile is not None
        tasks = max(1, self._completed)
        trial = ConcurrencyTrial(
            workers=self._active_limit,
            tasks=tasks,
            wall_seconds=max(1e-12, now - self._stage_started),
            task_seconds=self._task_seconds,
            cpu_seconds=self._cpu_seconds,
            mean_cpu_ratio=(
                sum(self._cpu_ratios) / len(self._cpu_ratios)
                if self._cpu_ratios
                else 0.0
            ),
        )
        self._profile.trials.append(trial)
        self._map_trials += 1

    def _choose(self) -> int:
        assert self._profile is not None
        current_trials = (
            self._profile.trials[-self._map_trials :] if self._map_trials else []
        )
        candidates = current_trials or self._profile.trials
        if not candidates:
            return min(self._map_max_workers, max(1, self._active_limit))
        best_throughput = max(trial.selection_score for trial in candidates)
        threshold = best_throughput * (1.0 - self.minimum_gain)
        # Prefer the smallest count whose throughput is statistically/practically
        # indistinguishable under the configured minimum-gain margin.
        return min(
            trial.workers for trial in candidates if trial.selection_score >= threshold
        )

    def maybe_advance(self, *, running: int, pending: int) -> bool:
        """Finish a pilot wave at a barrier and activate the next limit."""

        if not self._exploring or running > 0:
            return False
        if self._dispatched < self._target and pending > 0:
            return False
        now = time.monotonic()
        self._record_trial(now)
        self._level_index += 1
        if (
            self._level_index < len(self._levels)
            and pending > self._levels[self._level_index]
        ):
            self._begin_trial(self._levels[self._level_index], now)
        else:
            assert self._profile is not None
            selected = self._choose()
            self._profile.selected_workers = selected
            self._account_capacity(now)
            self._exploring = False
            self._active_limit = min(self._map_max_workers, max(1, selected))
            self._target = 0
            self._reset_stage(now)
        return True

    def finish_map(self) -> None:
        now = time.monotonic()
        if self._exploring and self._completed:
            self._record_trial(now)
            assert self._profile is not None
            self._profile.selected_workers = self._choose()
        self._account_capacity(now)
        self._exploring = False
        self._persist_profiles()

    def _load_profiles(self) -> None:
        if self.profile_path is None:
            return
        try:
            document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        if document.get("schema") != "autooptlib.concurrency-profiles":
            return
        hardware = document.get("hardware", {}).get(self.hardware_signature, {})
        profiles = hardware.get("profiles", {}) if isinstance(hardware, dict) else {}
        if not isinstance(profiles, dict):
            return
        self._profiles = {
            str(key): ConcurrencyProfile.from_dict(value)
            for key, value in profiles.items()
            if isinstance(value, dict)
        }

    @staticmethod
    def _merge_profile(
        existing: ConcurrencyProfile, current: ConcurrencyProfile
    ) -> ConcurrencyProfile:
        trials: list[ConcurrencyTrial] = []
        seen: set[str] = set()
        for trial in [*existing.trials, *current.trials]:
            token = json.dumps(trial.as_dict(), sort_keys=True, separators=(",", ":"))
            if token in seen:
                continue
            seen.add(token)
            trials.append(trial)
        selected = (
            current.selected_workers
            if current.maps >= existing.maps
            else existing.selected_workers
        )
        return ConcurrencyProfile(
            selected_workers=selected,
            maps=max(existing.maps, current.maps),
            trials=trials,
        )

    def _persist_profiles(self) -> None:
        if self.profile_path is None:
            return
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.profile_path.with_suffix(self.profile_path.suffix + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock:
            try:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                fcntl = None  # type: ignore[assignment]
            try:
                try:
                    document = json.loads(self.profile_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    document = {}
                if document.get("schema") != "autooptlib.concurrency-profiles":
                    document = {
                        "schema": "autooptlib.concurrency-profiles",
                        "schema_version": 1,
                        "hardware": {},
                    }
                hardware = document.setdefault("hardware", {})
                section = hardware.setdefault(self.hardware_signature, {})
                stored = section.setdefault("profiles", {})
                for key, profile in self._profiles.items():
                    previous = stored.get(key)
                    if isinstance(previous, dict):
                        profile = self._merge_profile(
                            ConcurrencyProfile.from_dict(previous), profile
                        )
                    stored[key] = profile.as_dict()
                    self._profiles[key] = profile
                temporary = self.profile_path.with_name(
                    f".{self.profile_path.name}.{os.getpid()}.tmp"
                )
                temporary.write_text(
                    json.dumps(document, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, self.profile_path)
            finally:
                if fcntl is not None:
                    try:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass

    def profile_snapshot(self) -> dict[str, dict[str, object]]:
        return {
            signature: profile.as_dict()
            for signature, profile in self._profiles.items()
        }


__all__ = [
    "AdaptiveConcurrencyController",
    "ConcurrencyProfile",
    "ConcurrencyTrial",
]
