"""ParadisEO eo1PtBitXover compatibility component for Search v6.0."""

from ._fastga_bit_crossover import execute_fastga_bit_crossover


def cross_fastga_one(*args):
    return execute_fastga_bit_crossover(args, "one")
