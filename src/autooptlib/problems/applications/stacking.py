"""Transparent reference model for the material-stacking application class.

The industrial study used proprietary material records and company-specific
rules. This module exposes those concepts through a documented data model and
a synthetic generator; it does not claim to reproduce the paper's table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping

import numpy as np

from autooptlib.problems.base import ProblemDefinition, make_problem
from autooptlib.problems.cpu import CPUObjective


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

    level_heights: np.ndarray = np.zeros(instance.n_levels, dtype=float)
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
    width_excess: np.ndarray = np.zeros(instance.n_levels, dtype=float)
    weight_excess: np.ndarray = np.zeros(instance.n_levels, dtype=float)
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
    *, cpu: bool = False, cpu_backend: str = "python",
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
        lower: np.ndarray = np.zeros(instance.dimension, dtype=int)
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
        cpu=stacking_cpu_objective(cpu_backend) if cpu else None,
    )


def _prepare_stacking(instance):
    from copy import deepcopy
    from dataclasses import replace

    instance = deepcopy(instance)
    instance.validate()
    # Preserve the original constraint dtype, while the objective explicitly
    # promotes weights/frequencies to float64 in its scalar reference.
    instance = replace(instance, widths=np.asarray(instance.widths),
                       heights=np.asarray(instance.heights), weights=np.asarray(instance.weights),
                       material_groups=np.asarray(instance.material_groups),
                       product_groups=np.asarray(instance.product_groups),
                       access_frequencies=np.asarray(instance.access_frequencies, dtype=float),
                       shared_material=np.asarray(instance.shared_material, dtype=bool))
    objective_weights = np.asarray(instance.weights, dtype=float)
    material = tuple(np.flatnonzero(instance.material_groups == label)
                     for label in np.unique(instance.material_groups))
    product = tuple(np.flatnonzero(instance.product_groups == label)
                    for label in np.unique(instance.product_groups))
    return (instance, material, product,
            max(1.0, float(instance.n_levels * instance.n_positions)),
            max(1.0, objective_weights.sum()),
            max(1.0, instance.access_frequencies.sum()),
            (instance.n_positions - 1) / 2.0, objective_weights)


def _stacking_workspace(prepared, _capacity):
    instance = prepared[0]
    n=instance.n_items
    selected=np.flatnonzero(instance.shared_material)
    dtype=np.result_type(instance.widths,instance.heights)
    return (np.empty(instance.n_levels),np.empty(n,dtype=bool),np.empty(n,dtype=int),
            np.empty(n,dtype=bool),np.empty(n,dtype=dtype),np.empty(n,dtype=dtype),
            np.empty(n),np.empty(n),np.empty(n,dtype=int),selected,
            np.empty(len(selected),dtype=int),np.empty(len(selected)))


def _stacking_into(X, prepared, workspace, objectives, constraints):
    instance, material, product, scale, total_weight, total_frequency, center, objective_weights = prepared
    (level_heights,members,location,orientation_mask,heights,widths,
     weighted,frequencies,gather,shared_indices,shared_positions,shared_values)=workspace
    n = instance.n_items
    prefs = instance.preferences
    for i, decision in enumerate(X):
        levels, positions, orientations = decision[:n], decision[n:2*n], decision[2*n:]
        # Preserve NumPy's original gather/reduction order and dtype. Share the
        # decode and level masks between objective and constraint calculations.
        np.equal(orientations,0,out=orientation_mask)
        np.copyto(heights,instance.widths);np.copyto(heights,instance.heights,where=orientation_mask)
        np.copyto(widths,instance.heights);np.copyto(widths,instance.widths,where=orientation_mask)
        for level in range(instance.n_levels):
            np.equal(levels, level, out=members)
            level_heights[level] = float(np.max(heights[members])) if np.any(members) else 0.0
            constraints[i, level] = float(np.sum(widths[members]) - instance.rack_width)
            constraints[i, instance.n_levels + level] = float(
                np.sum(instance.weights[members]) - instance.max_level_weight)
        occupied_area = float(instance.rack_width * np.sum(level_heights))
        np.multiply(levels, instance.n_positions, out=location)
        np.add(location, positions, out=location)
        np.multiply(objective_weights,levels,out=weighted)
        heavy_low = float(np.sum(weighted) / total_weight)
        same_material = 0.0
        for indices in material:
            if indices.size > 1:
                values=gather[:len(indices)];np.take(location,indices,out=values,mode='clip')
                same_material += float(np.var(values))
        same_material /= scale
        same_product = 0.0
        for indices in product:
            if indices.size > 1:
                values=gather[:len(indices)];np.take(location,indices,out=values,mode='clip')
                same_product += float(np.var(values))
        same_product /= scale
        np.take(positions,shared_indices,out=shared_positions,mode='clip')
        np.subtract(shared_positions,center,out=shared_values);np.abs(shared_values,out=shared_values)
        shared_central = float(np.sum(shared_values)) / scale
        np.multiply(instance.access_frequencies,positions,out=frequencies)
        frequent_shipping = float(np.sum(frequencies) / total_frequency)
        objectives[i] = occupied_area + (
            prefs.heavy_low * heavy_low + prefs.same_material * same_material
            + prefs.same_product * same_product + prefs.shared_central * shared_central
            + prefs.frequent_near_shipping * frequent_shipping)


def stacking_cpu_objective(backend="python") -> CPUObjective:
    if backend == "native":
        return _StackingNativeCPU(CPUObjective.from_builtin("material_stacking", decision_dtype="int64", prepare=_prepare_stacking_native, independent=True).evaluate_into, prepare=_prepare_stacking_native, backend="native", independent=True)
    if backend != "python":
        raise ValueError("Stacking CPU backend must be python or native")
    """Prepared CPU adapter; the public scalar functions retain all checks."""
    # Constraint count depends on the instance. The factory supplies this via
    # an instance-specific spec below rather than inspecting it in the kernel.
    return _StackingCPU(_stacking_into, prepare=_prepare_stacking,
                        workspace_factory=_stacking_workspace, independent=True)


class _StackingCPU(CPUObjective):
    def bind(self, data, dimension, dtype):
        from dataclasses import replace

        spec = replace(self, n_constraints=2 * data.n_levels)
        return CPUObjective.bind(spec, data, dimension, dtype)


def _prepare_stacking_native(instance):
    prepared = _prepare_stacking(instance)
    inst, material, product, scale, total_weight, total_frequency, center, _weights = prepared
    if any(np.asarray(v).dtype not in (np.dtype('float32'), np.dtype('float64'))
           for v in (inst.widths, inst.heights, inst.weights)):
        raise TypeError("Native stacking requires float32/float64 item arrays; use the python adapter for other dtypes")
    if max(inst.n_levels,inst.n_positions) > 2**53:
        raise ValueError("Native stacking metadata must be exactly representable in float64")
    prefs = list(vars(inst.preferences).values())
    if len(prefs)!=5 or any(type(v) not in (float,int,np.float64) for v in prefs):
        raise TypeError("Native stacking preferences must have double-precision scalar semantics")
    header = np.array([inst.n_levels, inst.n_positions, inst.rack_width, inst.max_level_weight,
        *prefs, len(material), len(product), inst.n_items,
        np.result_type(inst.widths, inst.heights)==np.dtype('float32'),
        inst.weights.dtype==np.dtype('float32'),
        np.result_type(inst.widths,inst.heights)==np.dtype('float32') and np.asarray(np.float32(0)-inst.rack_width).dtype==np.dtype('float32'),
        inst.weights.dtype==np.dtype('float32') and np.asarray(np.float32(0)-inst.max_level_weight).dtype==np.dtype('float32'),
        total_weight,total_frequency,scale,center],dtype=float)
    def groups(values):
        offsets=np.cumsum([0]+[len(x) for x in values])
        return np.concatenate((offsets,np.concatenate(values)))
    return np.concatenate((header,inst.widths,inst.heights,inst.weights,
        inst.access_frequencies,inst.shared_material,groups(material),groups(product)))


class _StackingNativeCPU(CPUObjective):
    def bind(self, data, dimension, dtype):
        from dataclasses import replace
        return CPUObjective.bind(replace(self,n_constraints=2*data.n_levels),data,dimension,dtype)
