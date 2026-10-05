"""Search V6 method settings; experiment schedules are supplied explicitly.

Experiment schedules must be supplied explicitly; this module only defines
the retained Search V6 baseline settings.
"""

SEARCH_SETTINGS = {
    "graph_semantics": "stream_graph_v2",
    "structure_mutation_rate": 0.3,
    "archive_size": 1,
    "stagnation_generations": 3,
    "restart_fraction": 0.5,
    "max_attempts": 50,
    "improvement_rate": 0.05,
    "parameter_action_probability": 0.5,
    "action_probability_gain": 0.3,
    "action_reward_ewma": 0.2,
    "parameter_cma_initial_sigma": 1,
    "stream_event_budget_multiplier": 4,
    "parameter_cma_offspring": 5,
    "structure_candidates_per_action": 3,
    "post_structure_cma_generations": 1,
    "parameter_cma_block_generations": 3,
    "population_size_max": 100,
    "offspring_size_max": 100,
}


def require_search_protocol(protocol):
    design = protocol.design
    if design.get("search", {}).get("graph_semantics") != "stream_graph_v2":
        raise ValueError("Only Search V6 stream_graph_v2 is supported.")
    if design.get("autooptlib_evaluation_policy", {}).get("aol_search") != "exact":
        raise ValueError("Search V6 requires exact evaluation.")
    retired = {"staged_exact_selection", "budget_extension", "search_warm_start"}
    if retired.intersection(design) or "search_method_followup" in protocol.document:
        raise ValueError("Retired Search experiment overrides are not supported.")


validate_contract = require_search_protocol
