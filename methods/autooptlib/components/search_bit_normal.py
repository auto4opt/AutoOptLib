"""Normal mutation strength from ParadisEO ``eoFastGA``."""

from ._bit_mutation import execute_bit_mutation, normal_strength


def search_bit_normal(*args):
    return execute_bit_mutation(
        args,
        component="search_bit_normal",
        strength=normal_strength,
        behavior=[[], ["GS", "large"]],
    )
