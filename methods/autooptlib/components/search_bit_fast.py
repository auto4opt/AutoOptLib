"""Power-law fast bit mutation from ParadisEO ``eoFastGA``."""

from ._bit_mutation import execute_bit_mutation, fast_strength


def search_bit_fast(*args):
    return execute_bit_mutation(
        args,
        component="search_bit_fast",
        strength=fast_strength,
        behavior=[[], ["GS", "large"]],
    )
