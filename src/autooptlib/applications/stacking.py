"""Transparent reference model for the material-stacking application class.

The industrial study used proprietary material records and company-specific
rules. This module exposes those concepts through a documented data model and
a synthetic generator; it does not claim to reproduce the paper's table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping

import numpy as np

from ..problems.base import ProblemDefinition, make_problem


@dataclass(frozen=True)
class StackingWeights:
    """Soft-preference weights used by :func:`stacking_objective`."""

    heavy_low: float = 1.0
    same_material: float = 1.0
    same_product: float = 0.5
    shared_central: float = 0.5
    frequent_near_shipping: float = 0.5

    def validate(self) -> None:
        values = np.asarray(list(vars(self).values()), dtype=float)
        if np.any(values < 0) or not np.all(np.isfinite(values)):
            raise ValueError(
                "Stacking preference weights must be finite and non-negative."
            )


@dataclass(frozen=True)
class MaterialStackingInstance:
    """Material dimensions and rack rules for one stacking instance."""

    widths: np.ndarray
    heights: np.ndarray
    weights: np.ndarray
    material_groups: np.ndarray
    product_groups: np.ndarray
    access_frequencies: np.ndarray
    shared_material: np.ndarray
    n_levels: int
    n_positions: int
    rack_width: float
    max_level_weight: float
    preferences: StackingWeights = StackingWeights()

    @property
    def n_items(self) -> int:
        return int(np.asarray(self.widths).size)

    @property
    def dimension(self) -> int:
        return 3 * self.n_items

    def validate(self) -> None:
        arrays = {
            "widths": self.widths,
            "heights": self.heights,
            "weights": self.weights,
            "material_groups": self.material_groups,
            "product_groups": self.product_groups,
            "access_frequencies": self.access_frequencies,
            "shared_material": self.shared_material,
        }
        lengths = {np.asarray(value).reshape(-1).size for value in arrays.values()}
        if lengths != {self.n_items} or self.n_items == 0:
            raise ValueError(
                "All stacking item arrays must have the same non-zero length."
            )
        for name in ("widths", "heights", "weights", "access_frequencies"):
            values = np.asarray(arrays[name], dtype=float)
            if np.any(values < 0) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain finite non-negative values.")
        if min(self.n_levels, self.n_positions) <= 0:
            raise ValueError("n_levels and n_positions must be positive.")
        if self.rack_width <= 0 or self.max_level_weight <= 0:
            raise ValueError("rack_width and max_level_weight must be positive.")
        self.preferences.validate()


def _decode(
    decision: np.ndarray, instance: MaterialStackingInstance
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(decision, dtype=int).reshape(-1)
    if values.size != instance.dimension:
        raise ValueError(
            f"Expected {instance.dimension} stacking decisions, got {values.size}."
        )
    n = instance.n_items
    levels = values[:n]
    positions = values[n : 2 * n]
    orientations = values[2 * n :]
    if (
        np.any(levels < 0)
        or np.any(levels >= instance.n_levels)
        or np.any(positions < 0)
        or np.any(positions >= instance.n_positions)
        or np.any((orientations != 0) & (orientations != 1))
    ):
        raise ValueError("Stacking decision is outside the declared discrete bounds.")
    return levels, positions, orientations


def _group_spread(location: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for label in np.unique(labels):
        members = location[np.asarray(labels) == label]
        if members.size > 1:
            total += float(np.var(members))
    return total


def stacking_objective(
    decision: np.ndarray, instance: MaterialStackingInstance
) -> float:
    """Minimize occupied rack area plus documented operational preferences."""
    instance.validate()
    levels, positions, orientations = _decode(decision, instance)
    heights = np.where(orientations == 0, instance.heights, instance.widths)

    level_heights = np.zeros(instance.n_levels, dtype=float)
    for level in range(instance.n_levels):
        members = levels == level
        if np.any(members):
            level_heights[level] = float(np.max(heights[members]))
    occupied_area = float(instance.rack_width * np.sum(level_heights))

    location = levels * instance.n_positions + positions
    scale = max(1.0, float(instance.n_levels * instance.n_positions))
    weights = np.asarray(instance.weights, dtype=float)
    frequency = np.asarray(instance.access_frequencies, dtype=float)
    shared = np.asarray(instance.shared_material, dtype=bool)
    prefs = instance.preferences
    heavy_low = float(np.sum(weights * levels) / max(1.0, weights.sum()))
    same_material = (
        _group_spread(location, np.asarray(instance.material_groups)) / scale
    )
    same_product = _group_spread(location, np.asarray(instance.product_groups)) / scale
    center = (instance.n_positions - 1) / 2.0
    shared_central = float(np.sum(np.abs(positions[shared] - center))) / scale
    frequent_shipping = float(np.sum(frequency * positions) / max(1.0, frequency.sum()))

    return occupied_area + (
        prefs.heavy_low * heavy_low
        + prefs.same_material * same_material
        + prefs.same_product * same_product
        + prefs.shared_central * shared_central
        + prefs.frequent_near_shipping * frequent_shipping
    )


def stacking_constraints(
    decision: np.ndarray, instance: MaterialStackingInstance
) -> np.ndarray:
    """Return per-level width and weight excess; positive values violate."""
    instance.validate()
    levels, _, orientations = _decode(decision, instance)
    widths = np.where(orientations == 0, instance.widths, instance.heights)
    width_excess = np.zeros(instance.n_levels, dtype=float)
    weight_excess = np.zeros(instance.n_levels, dtype=float)
    for level in range(instance.n_levels):
        members = levels == level
        width_excess[level] = float(np.sum(widths[members]) - instance.rack_width)
        weight_excess[level] = float(
            np.sum(np.asarray(instance.weights)[members]) - instance.max_level_weight
        )
    return np.concatenate((width_excess, weight_excess))


def generate_stacking_instance(
    n_items: int,
    *,
    seed: int,
    n_levels: int = 4,
    n_positions: int = 8,
    rack_width: float = 12.0,
    max_level_weight: float = 40.0,
) -> MaterialStackingInstance:
    """Generate a deterministic non-proprietary instance for examples and CI."""
    if min(n_items, n_levels, n_positions) <= 0:
        raise ValueError("n_items, n_levels, and n_positions must be positive.")
    rng = np.random.default_rng(seed)
    instance = MaterialStackingInstance(
        widths=rng.uniform(0.5, 2.0, n_items),
        heights=rng.uniform(0.5, 2.5, n_items),
        weights=rng.uniform(1.0, 10.0, n_items),
        material_groups=rng.integers(0, max(2, n_items // 3), n_items),
        product_groups=rng.integers(0, max(2, n_items // 4), n_items),
        access_frequencies=rng.uniform(0.0, 1.0, n_items),
        shared_material=rng.random(n_items) < 0.25,
        n_levels=n_levels,
        n_positions=n_positions,
        rack_width=rack_width,
        max_level_weight=max_level_weight,
    )
    instance.validate()
    return instance


def make_material_stacking_problem(
    instances: Mapping[Hashable, MaterialStackingInstance],
) -> ProblemDefinition:
    """Build an AutoOptLib discrete problem from keyed stacking records."""
    data = dict(instances)
    if not data:
        raise ValueError("At least one material-stacking instance is required.")
    for instance in data.values():
        instance.validate()

    def get_data(instance_id: Hashable) -> MaterialStackingInstance:
        try:
            return data[instance_id]
        except KeyError as exc:
            raise KeyError(f"Unknown stacking instance {instance_id!r}.") from exc

    def bounds(instance_id: Hashable) -> np.ndarray:
        instance = get_data(instance_id)
        lower = np.zeros(instance.dimension, dtype=int)
        upper = np.concatenate(
            (
                np.full(instance.n_items, instance.n_levels - 1, dtype=int),
                np.full(instance.n_items, instance.n_positions - 1, dtype=int),
                np.ones(instance.n_items, dtype=int),
            )
        )
        return np.vstack((lower, upper))

    return make_problem(
        stacking_objective,
        bounds=bounds,
        problem_type="discrete",
        constraint=stacking_constraints,
        data_factory=get_data,
        name="material_stacking",
    )
