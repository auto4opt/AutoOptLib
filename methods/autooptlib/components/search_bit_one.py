"""Flip exactly one distinct bit, matching ParadisEO ``eoFastGA``."""

from ._bit_mutation import execute_bit_mutation, fixed_strength


def search_bit_one(*args):
    return execute_bit_mutation(
        args,
        component="search_bit_one",
        strength=fixed_strength(1),
        behavior=[["LS", "small"], []],
    )
