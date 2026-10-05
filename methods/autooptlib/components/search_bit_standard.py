"""Standard bit mutation from ParadisEO ``eoFastGA``."""

from ._bit_mutation import execute_bit_mutation, standard_strength


def search_bit_standard(*args):
    return execute_bit_mutation(
        args,
        component="search_bit_standard",
        strength=standard_strength,
        behavior=[["LS", "small"], []],
    )
