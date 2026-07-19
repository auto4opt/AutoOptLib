"""Reproducible reference model for discrete RIS passive beamforming.

The model represents the application class described in the AutoOptLib paper.
It uses the zero-forcing and water-filling evaluation implemented by the
historical MATLAB application. Generated channels remain the default, so the
paper's numerical table still requires the archived research instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Mapping

import numpy as np

from ..problems.base import ProblemDefinition, make_problem


@dataclass(frozen=True)
class RISBeamformingInstance:
    """Channel data for a quantized-phase MU-MISO downlink."""

    bs_to_ris: np.ndarray
    direct_channels: np.ndarray
    ris_to_users: np.ndarray
    phase_bits: int = 2
    transmit_power: float = 1.0
    noise_power: float = 1.0
    user_weights: np.ndarray | None = None

    @property
    def n_elements(self) -> int:
        return int(np.asarray(self.bs_to_ris).shape[0])

    def validate(self) -> None:
        g = np.asarray(self.bs_to_ris)
        hd = np.asarray(self.direct_channels)
        hr = np.asarray(self.ris_to_users)
        if g.ndim != 2 or hd.ndim != 2 or hr.ndim != 2:
            raise ValueError("All RIS channel arrays must be two-dimensional.")
        n_elements, n_antennas = g.shape
        n_users = hd.shape[0]
        if n_elements == 0 or n_antennas == 0 or n_users == 0:
            raise ValueError("RIS instances require elements, antennas, and users.")
        if hd.shape[1] != n_antennas or hr.shape != (n_users, n_elements):
            raise ValueError(
                "Incompatible BS-RIS, direct, and RIS-user channel shapes."
            )
        if not isinstance(self.phase_bits, int) or self.phase_bits <= 0:
            raise ValueError("phase_bits must be a positive integer.")
        if self.transmit_power <= 0 or self.noise_power <= 0:
            raise ValueError("transmit_power and noise_power must be positive.")
        if self.user_weights is not None:
            user_weights = np.asarray(self.user_weights, dtype=float).reshape(-1)
            if (
                user_weights.size != n_users
                or np.any(user_weights < 0)
                or not np.all(np.isfinite(user_weights))
                or float(user_weights.sum()) <= 0
            ):
                raise ValueError(
                    "user_weights must contain one finite non-negative value per user "
                    "and have a positive sum."
                )
        for name, channel in (
            ("bs_to_ris", g),
            ("direct_channels", hd),
            ("ris_to_users", hr),
        ):
            if not np.all(np.isfinite(channel)):
                raise ValueError(f"{name} must contain only finite values.")


def _complex_normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.0)


def generate_ris_instance(
    n_elements: int,
    *,
    seed: int,
    n_antennas: int = 4,
    n_users: int = 3,
    phase_bits: int = 2,
    transmit_power: float = 1.0,
    noise_power: float = 1.0,
) -> RISBeamformingInstance:
    """Generate deterministic Rayleigh channels for examples and CI."""
    if min(n_elements, n_antennas, n_users) <= 0:
        raise ValueError("n_elements, n_antennas, and n_users must be positive.")
    rng = np.random.default_rng(seed)
    instance = RISBeamformingInstance(
        bs_to_ris=_complex_normal(rng, (n_elements, n_antennas)),
        direct_channels=0.2 * _complex_normal(rng, (n_users, n_antennas)),
        ris_to_users=_complex_normal(rng, (n_users, n_elements)),
        phase_bits=phase_bits,
        transmit_power=transmit_power,
        noise_power=noise_power,
        user_weights=np.ones(n_users),
    )
    instance.validate()
    return instance


def load_ris_matlab(path: str | Path) -> dict[int, RISBeamformingInstance]:
    """Load the historical ``Beanforming.mat`` records using optional SciPy.

    Returned keys are one-based to match the original MATLAB instance IDs.
    The data file is not bundled in the Apache-licensed wheel; users must
    supply a copy whose licensing is appropriate for their use.
    """
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "Reading MATLAB application data requires "
            "`pip install autooptlib[applications]`."
        ) from exc
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"RIS MATLAB data file not found: {source}")
    document = loadmat(source, simplify_cells=True)
    records = document.get("Data")
    if records is None:
        raise ValueError("MATLAB data file does not contain a 'Data' variable.")
    if isinstance(records, dict):
        records = [records]
    instances: dict[int, RISBeamformingInstance] = {}
    for index, record in enumerate(records, start=1):
        try:
            instance = RISBeamformingInstance(
                bs_to_ris=np.asarray(record["G"]),
                direct_channels=np.asarray(record["Hd"]),
                ris_to_users=np.asarray(record["Hr"]),
                phase_bits=int(record["b"]),
                transmit_power=float(record["PT"]),
                noise_power=1.0,
                user_weights=np.asarray(record["omega"], dtype=float).reshape(-1),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid RIS record {index} in {source}.") from exc
        instance.validate()
        instances[index] = instance
    return instances


def reciprocal_sum_rate(
    phase_indices: np.ndarray, instance: RISBeamformingInstance
) -> float:
    """Return reciprocal downlink sum rate for a quantized RIS phase vector."""
    instance.validate()
    decisions = np.asarray(phase_indices, dtype=int).reshape(-1)
    if decisions.size != instance.n_elements:
        raise ValueError(
            f"Expected {instance.n_elements} RIS phase decisions, got {decisions.size}."
        )
    levels = 2**instance.phase_bits
    if np.any(decisions < 0) or np.any(decisions >= levels):
        raise ValueError(f"RIS phase decisions must be in [0, {levels - 1}].")

    phases = np.exp(1j * 2.0 * np.pi * decisions / levels)
    # Explicit contraction avoids backend-specific complex GEMM warnings
    # observed for one of the larger historical channel matrices.
    reflected = np.asarray(instance.ris_to_users) * phases[np.newaxis, :]
    effective = np.asarray(instance.direct_channels) + np.einsum(
        "kn,nm->km", reflected, np.asarray(instance.bs_to_ris), optimize=False
    )

    # Match the historical MATLAB application: zero-forcing followed by
    # water-filling under the total BS power budget.
    gram = effective @ effective.conj().T
    beamformer_base = effective.conj().T @ np.linalg.pinv(gram)
    power_cost = np.real(np.diag(beamformer_base.conj().T @ beamformer_base))
    if np.any(power_cost <= np.finfo(float).tiny):
        return float(1.0 / np.finfo(float).tiny)
    filled = _water_filling(instance.transmit_power, power_cost)
    allocated = filled / power_cost
    beamformers = beamformer_base * np.sqrt(allocated)[np.newaxis, :]
    received = effective @ beamformers
    powers = np.abs(received) ** 2
    signal = np.diag(powers)
    interference = powers.sum(axis=1) - signal
    sinr = signal / (interference + instance.noise_power)
    user_weights = (
        np.ones(effective.shape[0])
        if instance.user_weights is None
        else np.asarray(instance.user_weights, dtype=float)
    )
    sum_rate = float(np.sum(np.log1p(sinr) * user_weights))
    tiny = float(np.finfo(float).tiny)
    denominator = sum_rate if sum_rate > tiny else tiny
    return 1.0 / denominator


def _water_filling(total_power: float, power_cost: np.ndarray) -> np.ndarray:
    """Port of the water-filling routine shipped with AutoOptLib v1.1."""
    costs = np.asarray(power_cost, dtype=float).reshape(-1)
    count = costs.size
    level = float(np.max(costs))
    used = float(np.sum(level - costs))
    if used > total_power:
        descending = np.sort(costs)[::-1]
        index = 1
        while used > total_power and index < count:
            level = float(descending[index])
            used = float(np.sum(np.maximum(level - costs, 0.0)))
            if used <= total_power:
                level += (total_power - used) / (count - index)
                break
            index += 1
        return np.maximum(level - costs, 0.0)
    level += (total_power - used) / count
    return level - costs


def make_ris_beamforming_problem(
    instances: Mapping[Hashable, RISBeamformingInstance],
) -> ProblemDefinition:
    """Build an AutoOptLib discrete problem from keyed RIS channel data."""
    data = dict(instances)
    if not data:
        raise ValueError("At least one RIS beamforming instance is required.")
    for instance in data.values():
        instance.validate()

    def get_data(instance_id: Hashable) -> RISBeamformingInstance:
        try:
            return data[instance_id]
        except KeyError as exc:
            raise KeyError(f"Unknown RIS instance {instance_id!r}.") from exc

    def bounds(instance_id: Hashable) -> np.ndarray:
        instance = get_data(instance_id)
        upper = 2**instance.phase_bits - 1
        return np.vstack(
            (
                np.zeros(instance.n_elements, dtype=int),
                np.full(instance.n_elements, upper),
            )
        )

    return make_problem(
        reciprocal_sum_rate,
        bounds=bounds,
        problem_type="discrete",
        data_factory=get_data,
        name="ris_beamforming",
    )
