"""Flip exactly three distinct bits, matching ParadisEO ``eoFastGA``."""

from ._bit_mutation import execute_bit_mutation, fixed_strength


def search_bit_three(*args):
    return execute_bit_mutation(
        args,
        component="search_bit_three",
        strength=fixed_strength(3),
        behavior=[["LS", "small"], []],
    )
