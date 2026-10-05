"""Frozen Learning method (2026-09-30), independent of experiment schedules."""

from copy import deepcopy

VERSION = "learning-transfer-frozen-20260930"
SOURCE_CANDIDATES = 1000
SOURCE_SETTINGS = {
    "adam_epsilon": 5e-09,
    "archive_size": 10,
    "archive_updates": 5,
    "attention_heads": 4,
    "batch_size": 10,
    "dropout": 0.1,
    "entropy_coefficient": 0.01,
    "feedforward_dimension": 256,
    "final_entropy_coefficient": 0.0,
    "final_learning_rate": 5e-05,
    "gradient_norm": 1.0,
    "graph_semantics": "stream_graph_v2",
    "layers": 4,
    "learning_rate": 5e-05,
    "max_length": 50,
    "max_pathways": 2,
    "max_search_stages": 2,
    "model_dimension": 64,
    "update_method": "archive_imitation",
    "weight_decay": 0.0005,
}
TARGET_SETTINGS = {
    **SOURCE_SETTINGS,
    "archive_size": 15,
    "pairwise_weight": 0.0,
    "training_greedy_candidate": True,
}


def learning_settings(candidate_budget, *, transfer=False):
    """Return independent settings for source training or target fine-tuning."""
    if type(candidate_budget) is not int or candidate_budget < 10:
        raise ValueError("candidate_budget must be an integer of at least 10")
    settings = deepcopy(TARGET_SETTINGS if transfer else SOURCE_SETTINGS)
    settings["full_batches"], settings["last_batch_size"] = divmod(
        candidate_budget, settings["batch_size"]
    )
    return settings
