"""Shifted standard bit mutation from ParadisEO ``eoFastGA``."""

from ._bit_mutation import execute_bit_mutation, shifted_strength


def search_bit_shifted(*args):
    return execute_bit_mutation(
        args,
        component="search_bit_shifted",
        strength=shifted_strength,
        behavior=[[], ["GS", "large"]],
    )
