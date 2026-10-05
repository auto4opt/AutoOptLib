"""Native eoFastGA current-to-pbest compatibility wrapper for Search v6.0."""

from __future__ import annotations

import numpy as np

from ._fastga_bit_crossover import STREAM_INVALIDATION_KEY
from .search_de_current_to_pbest import search_de_current_to_pbest


def search_fastga_de_current_to_pbest(*args):
    mode = args[-1]
    if mode != "execute":
        return search_de_current_to_pbest(*args)

    auxiliary = args[3] if len(args) > 3 and isinstance(args[3], dict) else {}
    targets = args[0]
    try:
        target_count = len(targets)
    except TypeError:
        target_count = np.atleast_2d(np.asarray(targets)).shape[0]
    raw_indices = auxiliary.get("_target_population_indices")
    direct_population_selection = raw_indices is not None and (
        np.asarray(raw_indices).reshape(-1).size == target_count
    )
    if direct_population_selection:
        # Native CurrentToPBestMutation searches for the first population clone
        # instead of trusting the selector's index.  The stream executor only
        # exposes target indices when this stage selected directly from the
        # current population, which is the exact native contract.
        auxiliary["_fastga_match_first_target"] = True
    else:
        # Defense in depth for old/corrupt serialized graphs.  audit4's grammar
        # rejects this exact component after stage zero, but falling back to the
        # general current-to-pbest semantics is safer than crashing a resumed
        # experiment if such an artifact reaches the component directly.
        auxiliary.pop("_fastga_match_first_target", None)
    forwarded = list(args)
    if len(forwarded) > 3:
        forwarded[3] = auxiliary
    try:
        decisions, state = search_de_current_to_pbest(*forwarded)
    finally:
        auxiliary.pop("_fastga_match_first_target", None)
    state = state if isinstance(state, dict) else auxiliary
    state.pop("_fastga_match_first_target", None)
    # The native CurrentToPBestMutation returns true unconditionally.
    count = np.atleast_2d(np.asarray(decisions)).shape[0]
    state[STREAM_INVALIDATION_KEY] = np.ones(count, dtype=bool)
    return decisions, state
