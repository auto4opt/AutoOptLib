"""Small CPU-kernel examples with explicitly ordered scalar references."""
import numpy as np

from .base import make_problem
from .cpu import CPUObjective


def ordered_sphere(x, _data):
    total = 0.0
    for j in range(len(x)):
        total += x[j] * x[j]
    return float(total)


def _sphere_into(X, _prepared, _workspace, objectives, _constraints):
    for i in range(len(X)):
        total = 0.0
        for j in range(X.shape[1]):
            total += X[i, j] * X[i, j]
        objectives[i] = total


def make_ordered_sphere_problem(*, cpu=False, backend="numba"):
    if cpu and backend == "native":
        return make_problem(ordered_sphere,bounds=(-5.,5.),cpu=CPUObjective.from_builtin("ordered_sum_squares",independent=True))
    return make_problem(ordered_sphere, bounds=(-5.0, 5.0),
                        cpu=CPUObjective(_sphere_into, backend=backend, independent=True)
                        if cpu else None)


def _residual_data(dimension):
    rng = np.random.default_rng(47)
    # Instance generation has its own RNG; evaluation never consumes one.
    return rng.normal(size=(dimension, dimension)), rng.normal(size=dimension)


def ordered_residual(x, data):
    matrix, target = data
    total = 0.0
    for row in range(len(target)):
        value = 0.0
        for col in range(len(x)):
            value += matrix[row, col] * x[col]
        residual = value - target[row]
        total += residual * residual
    return float(total)


def residual_constraint(x, _data):
    total = 0.0
    for j in range(len(x)):
        total += x[j]
    return np.array([total - 1.0, -total - 1.0])


def _residual_workspace(prepared, _capacity):
    return np.empty(len(prepared[1]))


def _residual_into(X, prepared, workspace, objectives, constraints):
    matrix, target = prepared
    for i in range(len(X)):
        for row in range(len(target)):
            value = 0.0
            for col in range(X.shape[1]):
                value += matrix[row, col] * X[i, col]
            workspace[row] = value - target[row]
        total = 0.0
        for row in range(len(target)):
            total += workspace[row] * workspace[row]
        objectives[i] = total
        total = 0.0
        for col in range(X.shape[1]):
            total += X[i, col]
        constraints[i, 0], constraints[i, 1] = total - 1.0, -total - 1.0


def make_ordered_residual_problem(*, cpu=False, backend="numba"):
    if cpu and backend == "native":
        return make_problem(ordered_residual,bounds=(-5.,5.),constraint=residual_constraint,
            data_factory=_residual_data,cpu=CPUObjective.from_builtin("ordered_residual",
                n_constraints=2,prepare=_residual_native_data,independent=True))
    return make_problem(
        ordered_residual, bounds=(-5.0, 5.0), constraint=residual_constraint,
        data_factory=_residual_data,
        cpu=CPUObjective(_residual_into, workspace_factory=_residual_workspace,
                         n_constraints=2, backend=backend, independent=True)
        if cpu else None)


def _residual_native_data(data):
    matrix,target=data
    return np.concatenate((matrix.reshape(-1),target))
