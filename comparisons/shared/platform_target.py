"""Shared target evaluator and durable FE ledger for irace and Sparkle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

FAILURE_COST = 1e30
CLAIM_HEARTBEAT_INTERVAL_SECONDS = 30.0
CLAIM_LEASE_SECONDS = 300.0
NATIVE_ZERO_EVALUATION_GENERATION_LIMIT = 1000
PARAMETERS = (
    "crossover_rate",
    "crossover_selector",
    "crossover",
    "aftercross_selector",
    "mutation_rate",
    "mutation_selector",
    "mutation",
    "replacement",
    "population_size",
    "offspring_size",
    "boundary_handling",
    "elite_fraction",
    "de_f",
    "de_cr",
    "de_p",
)
INTEGER_PARAMETERS = frozenset(PARAMETERS) - {
    "crossover_rate",
    "mutation_rate",
    "elite_fraction",
    "de_f",
    "de_cr",
    "de_p",
}
DEFAULT_CONFIGURATION: dict[str, float | int] = {
    "crossover_rate": 0.8,
    "crossover_selector": 0,
    "crossover": 0,
    "aftercross_selector": 0,
    "mutation_rate": 0.8,
    "mutation_selector": 0,
    "mutation": 0,
    "replacement": 0,
    "population_size": 20,
    "offspring_size": 20,
    "boundary_handling": 0,
    "elite_fraction": 0.2,
    "de_f": 0.5,
    "de_cr": 0.5,
    "de_p": 0.2,
}

DE_CURRENT_TO_PBEST_MUTATION = 3
ELITE_FRACTION_SELECTOR = 5


class LedgerIdentityError(RuntimeError):
    """Raised when a work ledger belongs to another immutable design request."""


def template_space_fingerprint(
    suite: str,
    *,
    population_size_max: int = 200,
    offspring_size_max: int = 200,
) -> str:
    """Fingerprint the common, executable eoFastGA design-space semantics."""

    if suite not in {"bbob", "pbo"}:
        raise ValueError(f"Unknown external-template suite: {suite!r}")
    payload = {
        # PBO F22 optimum metadata correction changes the training score and
        # therefore invalidates native PBO design ledgers from the old runner.
        "schema_version": 8 if suite == "pbo" else 7,
        "suite": suite,
        "selectors": [0, 1, 2, 4, ELITE_FRACTION_SELECTOR]
        if suite == "bbob"
        else [0, 1, 2, 3, 4],
        "crossovers": [0, 1, 2, 3],
        "mutations": list(range(4 if suite == "bbob" else 9)),
        "replacements": [0, 1, 2, 3],
        "population_size": [4, int(population_size_max)],
        "offspring_size": [1, int(offspring_size_max)],
        "boundary_handling": ["clip", "reflect", "resample"],
        "elite_fraction": [0.0, 1.0],
        "de_current_to_pbest": {"F": [0.0, 1.0], "CR": [0.0, 1.0], "p": [0.0, 1.0]},
        "constraints": [
            "replacement != 0 implies offspring_size <= population_size",
            "DE current-to-pbest disables external crossover and executes as traverse mutation",
        ],
        "zero_evaluation_generation_limit": (NATIVE_ZERO_EVALUATION_GENERATION_LIMIT),
    }
    if suite == "pbo":
        payload["optimum_metadata_revision"] = "f22-once-transformed-v1"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _integer_value(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a boolean.")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be a finite integer, got {value!r}.")
    return int(number)


def _real_value(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real number, not a boolean.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return number


def normalize_configuration(
    values: dict[str, Any], suite: str | None = None
) -> dict[str, float | int]:
    """Return the exact native-runner configuration with stable scalar types.

    Integer parameters are deliberately not rounded or truncated: silently mapping
    two configurator proposals to the same executable configuration corrupts the
    distinct-candidate ledger.
    """

    normalized = dict(DEFAULT_CONFIGURATION)
    for name in PARAMETERS:
        if name not in values or values[name] is None:
            continue
        normalized[name] = (
            _integer_value(name, values[name])
            if name in INTEGER_PARAMETERS
            else _real_value(name, values[name])
        )
    if suite not in {None, "bbob", "pbo"}:
        raise ValueError(f"Unknown suite {suite!r}.")
    if suite == "bbob" and normalized["mutation"] == DE_CURRENT_TO_PBEST_MUTATION:
        # These are semantic constants, not configurator choices. Canonicalize
        # inactive eoFastGA fields before ledger hashing so equivalent DE
        # proposals cannot consume distinct-candidate budget.  Selector 2 is
        # the persistent round-robin traversal shared with choose_traverse;
        # unlike eoSequentialSelect it is not reset for every offspring.
        normalized["crossover_rate"] = 0.0
        normalized["crossover_selector"] = 0
        normalized["crossover"] = 0
        normalized["aftercross_selector"] = 0
        normalized["mutation_rate"] = 1.0
        normalized["mutation_selector"] = 2
    elif suite == "bbob":
        for name in ("de_f", "de_cr", "de_p"):
            normalized[name] = DEFAULT_CONFIGURATION[name]
    elif suite == "pbo":
        normalized["boundary_handling"] = 0
        for name in ("de_f", "de_cr", "de_p"):
            normalized[name] = DEFAULT_CONFIGURATION[name]
    # Test activity after suite-specific canonicalization. In particular, a
    # BBOB DE proposal may arrive with selector 5 but DE fixes that selector to
    # traversal; its now-inactive elite_fraction must not survive in the
    # executable key and consume another candidate slot.
    if suite == "pbo" or not any(
        normalized[name] == ELITE_FRACTION_SELECTOR
        for name in (
            "crossover_selector",
            "aftercross_selector",
            "mutation_selector",
        )
    ):
        normalized["elite_fraction"] = DEFAULT_CONFIGURATION["elite_fraction"]
    return normalized


def configuration_key(suite: str, configuration: dict[str, Any]) -> str:
    encoded = json.dumps(
        {
            "suite": suite,
            "configuration": normalize_configuration(configuration, suite),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def configuration_is_valid(
    suite: str,
    configuration: dict[str, Any],
    *,
    population_size_max: int = 200,
    offspring_size_max: int = 200,
) -> bool:
    """Return whether eoFastGA can execute the normalized configuration."""

    if suite not in {"bbob", "pbo"}:
        return False
    try:
        normalized = normalize_configuration(configuration, suite)
    except (TypeError, ValueError, OverflowError):
        return False
    selectors = (
        {0, 1, 2, 4, ELITE_FRACTION_SELECTOR} if suite == "bbob" else {0, 1, 2, 3, 4}
    )
    if any(
        normalized[name] not in selectors
        for name in (
            "crossover_selector",
            "aftercross_selector",
            "mutation_selector",
        )
    ):
        return False
    if normalized["crossover"] not in {0, 1, 2, 3}:
        return False
    mutations = set(range(4 if suite == "bbob" else 9))
    if normalized["mutation"] not in mutations:
        return False
    if normalized["replacement"] not in {0, 1, 2, 3}:
        return False
    if not 0.0 <= normalized["crossover_rate"] <= 1.0:
        return False
    if not 0.0 <= normalized["mutation_rate"] <= 1.0:
        return False
    if normalized["boundary_handling"] not in {0, 1, 2}:
        return False
    if any(
        not 0.0 <= normalized[name] <= 1.0
        for name in ("elite_fraction", "de_f", "de_cr", "de_p")
    ):
        return False
    if not 4 <= normalized["population_size"] <= int(population_size_max):
        return False
    if not 1 <= normalized["offspring_size"] <= int(offspring_size_max):
        return False
    if (
        normalized["replacement"] != 0
        and normalized["offspring_size"] > normalized["population_size"]
    ):
        return False
    return True


class DesignLedger:
    """Durable exactly-once ledger for concurrent configurator target calls.

    A candidate is counted only after at least one evaluation has committed.
    Expensive work is protected by an atomic claim which also reserves its FE
    before the native runner starts.  A failed owner releases both the candidate
    slot and the FE reservation, while preserving a diagnostic record.
    """

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_key TEXT PRIMARY KEY,
                candidate_id INTEGER NOT NULL UNIQUE,
                configuration_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evaluations (
                candidate_key TEXT NOT NULL,
                task_key TEXT NOT NULL,
                cost REAL NOT NULL,
                evaluations INTEGER NOT NULL,
                PRIMARY KEY (candidate_key, task_key),
                FOREIGN KEY(candidate_key) REFERENCES candidates(candidate_key)
            );
            CREATE TABLE IF NOT EXISTS claims (
                candidate_key TEXT NOT NULL,
                task_key TEXT NOT NULL,
                owner_token TEXT NOT NULL,
                owner_pid INTEGER NOT NULL,
                owner_host TEXT NOT NULL,
                claimed_at REAL NOT NULL,
                reserved_evaluations INTEGER NOT NULL,
                PRIMARY KEY (candidate_key, task_key),
                FOREIGN KEY(candidate_key) REFERENCES candidates(candidate_key)
            );
            CREATE TABLE IF NOT EXISTS failures (
                failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_key TEXT NOT NULL,
                task_key TEXT NOT NULL,
                error TEXT NOT NULL,
                failed_at REAL NOT NULL
            );
            """
        for attempt in range(100):
            try:
                with self.connect() as connection:
                    connection.executescript(schema)
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 99:
                    raise
                time.sleep(0.05)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=600)
        connection.execute("PRAGMA busy_timeout=600000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def request_identity(request: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(
            request, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return {
            "schema_version": 1,
            "request_sha256": hashlib.sha256(encoded).hexdigest(),
            "protocol_fingerprint": request.get("protocol_fingerprint"),
            "template_space_fingerprint": request.get("template_space_fingerprint"),
            "method": request.get("method"),
            "suite": request.get("suite"),
            "function_id": request.get("function_id"),
        }

    def bind_request(self, request: dict[str, Any]) -> None:
        """Bind an empty ledger to one immutable request, or verify its identity."""

        identity = json.dumps(
            self.request_identity(request), sort_keys=True, separators=(",", ":")
        )
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'request_identity'"
            ).fetchone()
            if row is None:
                populated = sum(
                    int(
                        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[
                            0
                        ]
                    )
                    for table in ("candidates", "evaluations", "claims")
                )
                if populated:
                    raise LedgerIdentityError(
                        "Refusing to reuse a populated legacy design ledger without "
                        "an immutable request identity. Start a fresh work directory."
                    )
                connection.execute(
                    "INSERT INTO metadata VALUES ('request_identity', ?)", (identity,)
                )
            elif str(row[0]) != identity:
                raise LedgerIdentityError(
                    "Design-ledger request identity mismatch. The work directory "
                    "belongs to a different protocol, method, suite, or function."
                )

    def admit(
        self,
        suite: str,
        configuration: dict[str, Any],
        candidate_budget: int,
    ) -> tuple[str, int] | None:
        normalized = normalize_configuration(configuration, suite)
        key = configuration_key(suite, normalized)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT candidate_id FROM candidates WHERE candidate_key = ?", (key,)
            ).fetchone()
            if existing is not None:
                return key, int(existing[0])
            # ``admit`` is retained as a low-level registration helper for
            # tooling/tests. Formal evaluation uses ``acquire`` so registration
            # and an FE reservation are one transaction.
            count = int(
                connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
            )
            if count >= candidate_budget:
                return None
            candidate_id = int(
                connection.execute(
                    "SELECT COALESCE(MAX(candidate_id), 0) + 1 FROM candidates"
                ).fetchone()[0]
            )
            connection.execute(
                "INSERT INTO candidates VALUES (?, ?, ?)",
                (
                    key,
                    candidate_id,
                    json.dumps(normalized, sort_keys=True, separators=(",", ":")),
                ),
            )
            return key, candidate_id

    @staticmethod
    def _active_candidates(connection: sqlite3.Connection) -> int:
        return int(
            connection.execute(
                """
                SELECT COUNT(*) FROM candidates AS c
                WHERE EXISTS (
                    SELECT 1 FROM evaluations AS e
                    WHERE e.candidate_key = c.candidate_key
                ) OR EXISTS (
                    SELECT 1 FROM claims AS q
                    WHERE q.candidate_key = c.candidate_key
                )
                """
            ).fetchone()[0]
        )

    @staticmethod
    def _owner_is_alive(host: str, pid: int, claimed_at: float | None = None) -> bool:
        lease_is_current = (
            claimed_at is None or time.time() - float(claimed_at) <= CLAIM_LEASE_SECONDS
        )
        if host != socket.gethostname():
            # Remote PIDs cannot be inspected locally. A heartbeat lease makes
            # abandoned claims recoverable after a worker or host disappears.
            return lease_is_current
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return lease_is_current
        return lease_is_current

    def reclaim_stale_claims(self) -> int:
        """Release expired evaluation leases before enforcing global limits.

        ``acquire`` can reclaim the lease for the exact candidate/task pair it
        is asked to evaluate.  A resumed configurator may never propose that
        same pair again, however, leaving reserved FE that prevents every new
        proposal from fitting under the budget.  Proactive reclamation avoids
        that terminal deadlock while preserving live local or heartbeating
        remote owners.
        """

        reclaimed = 0
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            claims = connection.execute(
                """
                SELECT candidate_key, task_key, owner_pid, owner_host, claimed_at
                FROM claims
                """
            ).fetchall()
            for candidate_key, task_key, owner_pid, owner_host, claimed_at in claims:
                if self._owner_is_alive(
                    str(owner_host), int(owner_pid), float(claimed_at)
                ):
                    continue
                connection.execute(
                    "INSERT INTO failures(candidate_key, task_key, error, failed_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        candidate_key,
                        task_key,
                        "proactively reclaimed expired evaluation owner",
                        time.time(),
                    ),
                )
                connection.execute(
                    "DELETE FROM claims WHERE candidate_key = ? AND task_key = ?",
                    (candidate_key, task_key),
                )
                reclaimed += 1
            if reclaimed:
                connection.execute(
                    """
                    DELETE FROM candidates
                    WHERE NOT EXISTS (
                        SELECT 1 FROM evaluations AS e
                        WHERE e.candidate_key = candidates.candidate_key
                    ) AND NOT EXISTS (
                        SELECT 1 FROM claims AS q
                        WHERE q.candidate_key = candidates.candidate_key
                    )
                    """
                )
        return reclaimed

    def acquire(
        self,
        suite: str,
        configuration: dict[str, Any],
        candidate_budget: int,
        task_key: str,
        reserved_evaluations: int,
        total_fe_cap: int,
    ) -> tuple[str, str, float | None]:
        """Atomically return ``owner``, ``cached``, ``wait``, or ``limit``.

        The second item is the candidate key for every state; the third item is
        populated only for ``cached``.  ``owner`` callers receive the claim token
        as a fourth logical value encoded after a colon in the state string.
        """

        normalized = normalize_configuration(configuration, suite)
        key = configuration_key(suite, normalized)
        owner = uuid.uuid4().hex
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cached = connection.execute(
                "SELECT cost FROM evaluations WHERE candidate_key = ? AND task_key = ?",
                (key, task_key),
            ).fetchone()
            if cached is not None:
                return "cached", key, float(cached[0])

            claim = connection.execute(
                """
                SELECT owner_pid, owner_host, claimed_at FROM claims
                WHERE candidate_key = ? AND task_key = ?
                """,
                (key, task_key),
            ).fetchone()
            if claim is not None:
                if self._owner_is_alive(str(claim[1]), int(claim[0]), float(claim[2])):
                    return "wait", key, None
                connection.execute(
                    "INSERT INTO failures(candidate_key, task_key, error, failed_at) "
                    "VALUES (?, ?, ?, ?)",
                    (key, task_key, "reclaimed dead evaluation owner", time.time()),
                )
                connection.execute(
                    "DELETE FROM claims WHERE candidate_key = ? AND task_key = ?",
                    (key, task_key),
                )

            candidate = connection.execute(
                "SELECT candidate_id FROM candidates WHERE candidate_key = ?", (key,)
            ).fetchone()
            already_active = candidate is not None and bool(
                connection.execute(
                    """
                    SELECT EXISTS(SELECT 1 FROM evaluations WHERE candidate_key = ?)
                        OR EXISTS(SELECT 1 FROM claims WHERE candidate_key = ?)
                    """,
                    (key, key),
                ).fetchone()[0]
            )
            if not already_active and self._active_candidates(connection) >= int(
                candidate_budget
            ):
                return "limit", key, None

            used = int(
                connection.execute(
                    "SELECT COALESCE(SUM(evaluations), 0) FROM evaluations"
                ).fetchone()[0]
            )
            reserved = int(
                connection.execute(
                    "SELECT COALESCE(SUM(reserved_evaluations), 0) FROM claims"
                ).fetchone()[0]
            )
            if used + reserved + int(reserved_evaluations) > int(total_fe_cap):
                return "limit", key, None

            if candidate is None:
                candidate_id = int(
                    connection.execute(
                        "SELECT COALESCE(MAX(candidate_id), 0) + 1 FROM candidates"
                    ).fetchone()[0]
                )
                connection.execute(
                    "INSERT INTO candidates VALUES (?, ?, ?)",
                    (
                        key,
                        candidate_id,
                        json.dumps(normalized, sort_keys=True, separators=(",", ":")),
                    ),
                )
            connection.execute(
                """
                INSERT INTO claims VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    task_key,
                    owner,
                    os.getpid(),
                    socket.gethostname(),
                    time.time(),
                    int(reserved_evaluations),
                ),
            )
            return f"owner:{owner}", key, None

    def heartbeat(self, candidate_key: str, task_key: str, owner_token: str) -> bool:
        """Renew one in-flight claim and report whether ownership still exists."""

        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE claims SET claimed_at = ?
                WHERE candidate_key = ? AND task_key = ? AND owner_token = ?
                """,
                (time.time(), candidate_key, task_key, owner_token),
            )
        return cursor.rowcount == 1

    def complete(
        self,
        candidate_key: str,
        task_key: str,
        owner_token: str,
        cost: float,
        evaluations: int,
    ) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            claim = connection.execute(
                """
                SELECT owner_token, reserved_evaluations FROM claims
                WHERE candidate_key = ? AND task_key = ?
                """,
                (candidate_key, task_key),
            ).fetchone()
            if claim is None or str(claim[0]) != owner_token:
                raise RuntimeError("Evaluation claim was lost before it could commit.")
            if int(claim[1]) != int(evaluations):
                raise RuntimeError(
                    "Committed FE does not match the reserved FE amount."
                )
            existing = connection.execute(
                """
                SELECT cost, evaluations FROM evaluations
                WHERE candidate_key = ? AND task_key = ?
                """,
                (candidate_key, task_key),
            ).fetchone()
            if existing is not None and (
                float(existing[0]) != float(cost)
                or int(existing[1]) != int(evaluations)
            ):
                raise RuntimeError("Conflicting result already exists in the ledger.")
            if existing is None:
                connection.execute(
                    "INSERT INTO evaluations VALUES (?, ?, ?, ?)",
                    (candidate_key, task_key, float(cost), int(evaluations)),
                )
            connection.execute(
                "DELETE FROM claims WHERE candidate_key = ? AND task_key = ?",
                (candidate_key, task_key),
            )

    def fail(
        self, candidate_key: str, task_key: str, owner_token: str, error: str
    ) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            claim = connection.execute(
                """
                SELECT owner_token FROM claims
                WHERE candidate_key = ? AND task_key = ?
                """,
                (candidate_key, task_key),
            ).fetchone()
            if claim is None or str(claim[0]) != owner_token:
                return
            connection.execute(
                "INSERT INTO failures(candidate_key, task_key, error, failed_at) "
                "VALUES (?, ?, ?, ?)",
                (candidate_key, task_key, str(error)[-4000:], time.time()),
            )
            connection.execute(
                "DELETE FROM claims WHERE candidate_key = ? AND task_key = ?",
                (candidate_key, task_key),
            )
            has_work = bool(
                connection.execute(
                    """
                    SELECT EXISTS(SELECT 1 FROM evaluations WHERE candidate_key = ?)
                        OR EXISTS(SELECT 1 FROM claims WHERE candidate_key = ?)
                    """,
                    (candidate_key, candidate_key),
                ).fetchone()[0]
            )
            if not has_work:
                connection.execute(
                    "DELETE FROM candidates WHERE candidate_key = ?", (candidate_key,)
                )

    def cached_cost(self, candidate_key: str, task_key: str) -> float | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT cost FROM evaluations WHERE candidate_key = ? AND task_key = ?",
                (candidate_key, task_key),
            ).fetchone()
        return None if row is None else float(row[0])

    def record(
        self,
        candidate_key: str,
        task_key: str,
        cost: float,
        evaluations: int,
    ) -> None:
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT cost, evaluations FROM evaluations "
                "WHERE candidate_key = ? AND task_key = ?",
                (candidate_key, task_key),
            ).fetchone()
            if existing is not None:
                if float(existing[0]) != float(cost) or int(existing[1]) != int(
                    evaluations
                ):
                    raise RuntimeError(
                        "Conflicting result already exists in the ledger."
                    )
                return
            connection.execute(
                "INSERT INTO evaluations VALUES (?, ?, ?, ?)",
                (candidate_key, task_key, float(cost), int(evaluations)),
            )

    def summary(self) -> dict[str, Any]:
        with self.connect() as connection:
            registered = int(
                connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
            )
            candidates = int(
                connection.execute(
                    "SELECT COUNT(DISTINCT candidate_key) FROM evaluations"
                ).fetchone()[0]
            )
            evaluations, actual_fes = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(evaluations), 0) FROM evaluations"
            ).fetchone()
            inflight, reserved_fes = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(reserved_evaluations), 0) FROM claims"
            ).fetchone()
            failures = int(
                connection.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
            )
        return {
            "different_candidates": candidates,
            "registered_candidates": registered,
            "completed_candidate_tasks": int(evaluations),
            "actual_design_fes": int(actual_fes),
            "inflight_candidate_tasks": int(inflight),
            "reserved_design_fes": int(reserved_fes),
            "failure_attempts": failures,
        }

    def configurations(self) -> dict[str, dict[str, float | int]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT c.candidate_key, c.configuration_json FROM candidates AS c
                WHERE EXISTS (
                    SELECT 1 FROM evaluations AS e
                    WHERE e.candidate_key = c.candidate_key
                )
                """
            ).fetchall()
        return {key: json.loads(value) for key, value in rows}

    def observed_cost(self, candidate_key: str) -> float | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT AVG(cost) FROM evaluations WHERE candidate_key = ?",
                (candidate_key,),
            ).fetchone()
        return None if row is None or row[0] is None else float(row[0])

    def task_costs(self, candidate_key: str) -> dict[str, float]:
        """Return the durable per-instance observations for one candidate."""

        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT task_key, cost FROM evaluations
                WHERE candidate_key = ? ORDER BY task_key
                """,
                (candidate_key,),
            ).fetchall()
        return {str(task_key): float(cost) for task_key, cost in rows}

    def best_on_common_tasks(
        self, candidate_keys: list[str]
    ) -> tuple[str, float, int] | None:
        """Compare candidates only on tasks observed for every candidate.

        Native racing deliberately evaluates unequal task subsets. Comparing
        their raw means would reward candidates that happened to see easier
        tasks, so cross-attempt tie-breaking is restricted to the intersection.
        """

        unique = list(dict.fromkeys(candidate_keys))
        if not unique:
            return None
        wanted = set(unique)
        observations: dict[str, dict[str, float]] = {key: {} for key in unique}
        # One scan is substantially cheaper than opening one SQLite connection
        # per candidate when recovering a large interrupted native race.
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT candidate_key, task_key, cost FROM evaluations"
            ).fetchall()
        for candidate_key, task_key, cost in rows:
            if candidate_key in wanted:
                observations[str(candidate_key)][str(task_key)] = float(cost)
        if any(not values for values in observations.values()):
            return None
        common = set.intersection(*(set(values) for values in observations.values()))
        if not common:
            return None
        means = {
            key: sum(values[task] for task in common) / len(common)
            for key, values in observations.items()
        }
        best = min(unique, key=lambda key: (means[key], key))
        return best, means[best], len(common)

    def best_observed(self) -> tuple[dict[str, float | int], float]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT c.configuration_json, AVG(e.cost) AS mean_cost
                FROM candidates AS c JOIN evaluations AS e
                  ON c.candidate_key = e.candidate_key
                GROUP BY c.candidate_key
                ORDER BY mean_cost ASC, c.candidate_id ASC LIMIT 1
                """
            ).fetchone()
        if row is None:
            raise RuntimeError("No successful candidate evaluation was recorded.")
        return json.loads(row[0]), float(row[1])


def ledger_completion_errors(
    summary: dict[str, Any], request: dict[str, Any]
) -> list[str]:
    """Return violations of the registered external-design completion rule."""

    errors: list[str] = []
    candidate_budget = int(request["candidate_budget"])
    policy = str(request.get("evaluation_policy", "exact"))
    candidates = int(summary.get("different_candidates", -1))
    registered = int(summary.get("registered_candidates", -1))
    if policy == "exact":
        if candidates != candidate_budget:
            errors.append(
                "completed distinct candidates "
                f"{summary.get('different_candidates')} != {candidate_budget}"
            )
        if registered != candidate_budget:
            errors.append(
                "registered candidates "
                f"{summary.get('registered_candidates')} != {candidate_budget}"
            )
    elif policy == "native_instances":
        # Native racing is budgeted by target FE, not by the number of distinct
        # configurations: irace/SMAC decide how often promising configurations
        # are revisited on further instances.  There must nevertheless be no
        # registered-but-never-evaluated candidate left behind.
        if candidates <= 0:
            errors.append("no completed candidate exists")
        if registered != candidates:
            errors.append(
                f"registered candidates {registered} != completed candidates "
                f"{candidates}"
            )
    else:
        # Backward-compatible contract for legacy racing requests.
        if candidates != candidate_budget:
            errors.append(
                "completed distinct candidates "
                f"{summary.get('different_candidates')} != {candidate_budget}"
            )
        if registered != candidate_budget:
            errors.append(
                "registered candidates "
                f"{summary.get('registered_candidates')} != {candidate_budget}"
            )
    if int(summary.get("inflight_candidate_tasks", -1)) != 0:
        errors.append("in-flight candidate evaluations remain")
    if int(summary.get("reserved_design_fes", -1)) != 0:
        errors.append("reserved FE remain")
    completed = int(summary.get("completed_candidate_tasks", -1))
    actual_fes = int(summary.get("actual_design_fes", -1))
    if policy == "exact":
        if completed != candidate_budget:
            errors.append(
                f"completed aggregate evaluations {completed} != {candidate_budget}"
            )
        if actual_fes != int(request["total_fe_cap"]):
            errors.append(f"actual design FE {actual_fes} != {request['total_fe_cap']}")
    elif policy == "native_instances":
        minimum_task_fe = int(request["minimum_task_fe"])
        total_fe_cap = int(request["total_fe_cap"])
        if minimum_task_fe <= 0:
            errors.append("minimum native task FE must be positive")
        if not 0 < actual_fes <= total_fe_cap:
            errors.append(
                f"actual design FE {actual_fes} is outside the registered cap"
            )
        elif total_fe_cap - actual_fes >= minimum_task_fe:
            errors.append(
                "native target budget is not exhausted: "
                f"{total_fe_cap - actual_fes} FE remain and the minimum task costs "
                f"{minimum_task_fe}"
            )
        if completed <= 0:
            errors.append("no native candidate-instance task completed")
    else:
        if completed < candidate_budget:
            errors.append(f"completed candidate tasks {completed} < {candidate_budget}")
        if not 0 < actual_fes <= int(request["total_fe_cap"]):
            errors.append(
                f"actual design FE {actual_fes} is outside the registered cap"
            )
    return errors


def ledger_is_complete(summary: dict[str, Any], request: dict[str, Any]) -> bool:
    return not ledger_completion_errors(summary, request)


def _runner_cost(
    runner: Path,
    suite: str,
    function_id: int,
    task: dict[str, Any],
    run_budget: int,
    configuration: dict[str, Any],
) -> float:
    payload = {
        "schema": "autooptlib.eofastga-config",
        "schema_version": 1,
        "configuration": normalize_configuration(configuration, suite),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as stream:
        json.dump(payload, stream)
        stream.flush()
        completed = subprocess.run(
            [
                str(runner),
                "--config",
                stream.name,
                "--suite",
                suite,
                "--function",
                str(function_id),
                "--dimension",
                str(task["dimension"]),
                "--instance",
                str(task["instance"]),
                "--seed",
                str(task["seed"]),
                "--budget",
                str(run_budget),
                "--cost-only",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Native eoFastGA runner returned no cost output.")
    return float(lines[-1])


class _ClaimHeartbeat:
    """Keep a durable evaluation claim alive while its native runner blocks."""

    def __init__(
        self,
        ledger: DesignLedger,
        candidate_key: str,
        task_key: str,
        owner_token: str,
    ) -> None:
        self._ledger = ledger
        self._candidate_key = candidate_key
        self._task_key = task_key
        self._owner_token = owner_token
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(CLAIM_HEARTBEAT_INTERVAL_SECONDS):
            if not self._ledger.heartbeat(
                self._candidate_key, self._task_key, self._owner_token
            ):
                return

    def __enter__(self) -> "_ClaimHeartbeat":
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=CLAIM_HEARTBEAT_INTERVAL_SECONDS)


def _reported_result(
    cost: float, evaluations: int, report_evaluations: bool
) -> float | tuple[float, int]:
    if report_evaluations:
        # irace requires a strictly positive resource value. Cache hits and
        # rejected calls therefore consume one bookkeeping unit but no ledger FE.
        return float(cost), max(1, int(evaluations))
    return float(cost)


def evaluate_configuration(
    request_path: Path,
    ledger_path: Path,
    runner: Path,
    configuration: dict[str, Any],
    instance_path: Path,
    *,
    report_evaluations: bool = False,
) -> float | tuple[float, int]:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    ledger = DesignLedger(ledger_path)
    ledger.bind_request(request)
    suite = str(request["suite"])
    if not configuration_is_valid(
        suite,
        configuration,
        population_size_max=int(request.get("population_size_max", 200)),
        offspring_size_max=int(request.get("offspring_size_max", 200)),
    ):
        return _reported_result(FAILURE_COST, 1, report_evaluations)
    normalized = normalize_configuration(configuration, suite)
    exact = str(request["evaluation_policy"]) == "exact"
    task_key = "all-training-tasks" if exact else instance_path.resolve().as_posix()
    tasks = (
        list(request["training_tasks"])
        if exact
        else [json.loads(instance_path.read_text(encoding="utf-8"))]
    )
    evaluations = sum(
        int(task.get("budget", request["candidate_run_budget"])) for task in tasks
    )
    while True:
        state, candidate_key, cached = ledger.acquire(
            suite,
            normalized,
            int(request.get("candidate_limit", request["candidate_budget"])),
            task_key,
            evaluations,
            int(request["total_fe_cap"]),
        )
        if state == "cached":
            assert cached is not None
            return _reported_result(cached, 1, report_evaluations)
        if state == "limit":
            return _reported_result(FAILURE_COST, evaluations, report_evaluations)
        if state == "wait":
            # The claim owner is another target process evaluating this exact
            # executable configuration. Polling avoids duplicate native work;
            # a dead local owner is reclaimed atomically on the next iteration.
            time.sleep(0.1)
            continue
        if state.startswith("owner:"):
            owner_token = state.split(":", 1)[1]
            break
        raise RuntimeError(f"Unknown evaluation-claim state: {state!r}")

    try:
        with _ClaimHeartbeat(ledger, candidate_key, task_key, owner_token):
            gaps = [
                _runner_cost(
                    runner,
                    suite,
                    int(request["function_id"]),
                    task,
                    int(task.get("budget", request["candidate_run_budget"])),
                    normalized,
                )
                for task in tasks
            ]
        scores = [
            (
                math.log10(1.0 + max(0.0, gap))
                if suite == "bbob"
                else (
                    gap
                    if task.get("optimum_raw") is None
                    else max(0.0, gap) / max(1.0, abs(float(task["optimum_raw"])))
                )
            )
            for gap, task in zip(gaps, tasks)
        ]
        if not scores or not all(map(math.isfinite, scores)):
            raise FloatingPointError("Native target returned a non-finite score.")
        cost = sum(scores) / len(scores)
        ledger.complete(candidate_key, task_key, owner_token, cost, evaluations)
        return _reported_result(cost, evaluations, report_evaluations)
    except Exception as exc:
        ledger.fail(
            candidate_key,
            task_key,
            owner_token,
            f"{type(exc).__name__}: {exc}",
        )
        raise


def _configuration_from_tokens(tokens: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        name = tokens[index].lstrip("-").replace("-", "_")
        if name in PARAMETERS and index + 1 < len(tokens):
            result[name] = tokens[index + 1]
            index += 2
        else:
            index += 1
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--config-json")
    parser.add_argument(
        "--report-evaluations",
        action="store_true",
        help="Print 'cost FE' for irace's native maxTime budget.",
    )
    args, remainder = parser.parse_known_args(argv)
    raw = (
        json.loads(args.config_json)
        if args.config_json is not None
        else _configuration_from_tokens(remainder)
    )
    try:
        cost = evaluate_configuration(
            args.request,
            args.ledger,
            args.runner,
            raw,
            args.instance,
            report_evaluations=args.report_evaluations,
        )
    except LedgerIdentityError as exc:
        print(f"autooptlib-platform-target: {exc}", file=sys.stderr)
        return 2
    except (
        subprocess.SubprocessError,
        FloatingPointError,
        OSError,
        RuntimeError,
    ) as exc:
        # A native candidate failure is already durably recorded and its slot/FE
        # reservation has been released. Return the configurator's crash cost but
        # leave a useful diagnostic on stderr instead of silently swallowing it.
        print(
            f"autooptlib-platform-target: {type(exc).__name__}: {exc}", file=sys.stderr
        )
        cost = (FAILURE_COST, 1) if args.report_evaluations else FAILURE_COST
    if args.report_evaluations:
        value, evaluations = cost
        print(f"{value:.17g} {int(evaluations)}")
    else:
        print(f"{cost:.17g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
