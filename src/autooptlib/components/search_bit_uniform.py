"""Uniform mutation strength from ParadisEO ``eoFastGA``."""

from ._bit_mutation import execute_bit_mutation, uniform_strength


def search_bit_uniform(*args):
    return execute_bit_mutation(
        args,
        component="search_bit_uniform",
        strength=uniform_strength,
        behavior=[[], ["GS", "large"]],
    )
