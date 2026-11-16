"""Exact three-point segment crossover."""

from ._n_point_crossover import execute_n_point_crossover


def cross_point_three(*args):
    return execute_n_point_crossover(args, 3)
