"""Exact five-point segment crossover."""

from ._n_point_crossover import execute_n_point_crossover


def cross_point_five(*args):
    return execute_n_point_crossover(args, 5)
