"""Flip exactly five distinct bits, matching ParadisEO ``eoFastGA``."""

from ._bit_mutation import execute_bit_mutation, fixed_strength


def search_bit_five(*args):
    return execute_bit_mutation(
        args,
        component="search_bit_five",
        strength=fixed_strength(5),
        behavior=[["LS", "small"], []],
    )
