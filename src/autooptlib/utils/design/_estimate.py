"""Python port of the MATLAB Estimate routine."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _call_use_embed(surrogate: Any, algs: Sequence[Any], setting: Any):
    if surrogate is None:
        raise ValueError("Surrogate must be provided")
    if hasattr(surrogate, "UseEmbed") and callable(getattr(surrogate, "UseEmbed")):
        return surrogate.UseEmbed(algs, setting)
    if hasattr(surrogate, "use_embed") and callable(getattr(surrogate, "use_embed")):
        return surrogate.use_embed(algs, setting)
    raise AttributeError("Surrogate does not provide a use_embed/UseEmbed method")


def _predict(model: Any, data: Any):
    if model is None:
        raise ValueError("Surrogate model is required for prediction")
    if hasattr(model, "predict"):
        features = np.asarray(data)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return model.predict(features)
    raise AttributeError("Surrogate model lacks a predict method")


def estimate(
    self, problem: Any, setting: Any, ind_instance: Sequence[int], surrogate: Any
):
    """Estimate the designed algorithm's performance using a surrogate model."""
    algs = [self]
    embed_algs = _call_use_embed(surrogate, algs, setting)
    model = getattr(surrogate, "model", None)
    features = embed_algs[0]
    prediction = np.asarray(_predict(model, features))
    for instance_index in ind_instance:
        self.performance_approx[instance_index, :] = prediction.reshape(1, -1)
    return self
