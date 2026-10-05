"""Conditional standard bit mutation from ParadisEO ``eoFastGA``."""

from ._bit_mutation import conditional_strength, execute_bit_mutation


def search_bit_conditional(*args):
    return execute_bit_mutation(
        args,
        component="search_bit_conditional",
        strength=conditional_strength,
        behavior=[["LS", "small"], []],
    )
