"""ParadisEO eoNPtsBitXover(5) compatibility component for Search v6.0."""

from ._fastga_bit_crossover import execute_fastga_bit_crossover


def cross_fastga_five(*args):
    return execute_fastga_bit_crossover(args, "five")
