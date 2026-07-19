"""Surrogate model utilities (translation of MATLAB Approximate.m)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Iterable, List, Sequence

import numpy as np

from ._embedding import embedding
from ._helpers import ensure_rng

if TYPE_CHECKING:
    from . import Design


class _KNNModel:
    """Lightweight k-NN regressor used as a surrogate model."""

    def __init__(self, k: int = 15):
        self.k = k
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        y = np.asarray(y, dtype=float).reshape(-1)
        self._x = x
        self._y = y

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if self._x is None or self._y is None or self._x.size == 0:
            return np.zeros(x.shape[0], dtype=float)
        dists = np.linalg.norm(self._x[None, :, :] - x[:, None, :], axis=2)
        k = min(self.k, self._x.shape[0])
        idx = np.argsort(dists, axis=1)[:, :k]
        weights = np.exp(-dists[np.arange(dists.shape[0])[:, None], idx])
        weights_sum = weights.sum(axis=1, keepdims=True)
        weights_sum[weights_sum == 0] = 1.0
        preds = np.sum(weights * self._y[idx], axis=1) / weights_sum.ravel()
        return preds


def _mean_performance(alg: "Design") -> float:
    values = alg.ave_perform_all()
    if values.size == 0:
        return np.inf
    return float(np.mean(values))


def _mean_performance_approx(alg: "Design") -> float:
    values = alg.ave_perform_approx_all()
    if values.size == 0:
        return np.inf
    return float(np.mean(values))


class Approximate:
    """Python translation of MATLAB Approximate class."""

    def __init__(
        self, problem: Any, data: Any, setting: Any, ind_instance: Sequence[int]
    ):
        from . import Design

        self.data: List["Design"] = []
        self.embedding = None
        self.model: Any = None
        self.rand_seed: np.ndarray
        self.exact_g: np.ndarray

        alg_p = int(getattr(setting, "AlgP", getattr(setting, "alg_p", 1)))
        alg_q = int(getattr(setting, "AlgQ", getattr(setting, "alg_q", 1)))
        total = (alg_q + 2) * 3 if alg_p == 1 else (alg_p + 2) * 3
        self.rand_seed = ensure_rng(setting).permutation(total)

        exact_budget = int(getattr(setting, "Surro", getattr(setting, "surro", 0)))
        alg_n = int(getattr(setting, "AlgN", getattr(setting, "alg_n", 1)))
        alg_fe = int(getattr(setting, "AlgFE", getattr(setting, "alg_fe", alg_n)))
        alg_gmax = max(1, int(math.ceil(alg_fe / alg_n)))
        step = alg_gmax / max(exact_budget, 1) if exact_budget else alg_gmax
        self.exact_g = np.arange(1, alg_gmax + 1, max(1, int(round(step))))

        seeds = list(ind_instance)
        train_algs_exact: List["Design"] = []
        # Surrogate initialization is part of AlgFE.  The previous fixed pool
        # of 500 exact algorithms made small experiments silently perform
        # thousands of additional objective evaluations.
        training_count = min(alg_fe, max(alg_n, exact_budget))
        self.initial_evaluations = training_count
        for _ in range(training_count):
            alg = Design(problem, setting)
            alg.evaluate(problem, data, setting, seeds)
            train_algs_exact.append(alg)
        self.data = train_algs_exact

        embedding_count = max(30, 3 * training_count)
        train_algs_embed = [Design(problem, setting) for _ in range(embedding_count)]
        self.embedding = self.GetEmbed(train_algs_embed, setting)
        self.model = self.GetModel(self.data, setting)

    # Methods mirroring MATLAB class ---------------------------------------
    def GetEmbed(self, data: Iterable["Design"], setting: Any):
        return embedding(list(data), setting, self, "get")

    def UseEmbed(self, data: Iterable["Design"], setting: Any) -> np.ndarray:
        return embedding(list(data), setting, self, "use")

    def use_embed(self, data: Iterable["Design"], setting: Any) -> np.ndarray:
        return self.UseEmbed(data, setting)

    def GetModel(self, data: Iterable["Design"], setting: Any) -> Any:
        algorithms = list(data)
        embed_algs = self.UseEmbed(algorithms, setting)
        labels = np.array([_mean_performance(alg) for alg in algorithms], dtype=float)
        try:
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(
                n_estimators=1000,
                min_samples_leaf=5,
                random_state=getattr(setting, "Seed", getattr(setting, "seed", None)),
                n_jobs=1,
            )
        except ImportError:
            # Keep approximate mode usable in the NumPy-only core install;
            # ``autooptlib[surrogate]`` activates the reference RF model.
            model = _KNNModel()
        model.fit(embed_algs, labels)
        return model

    def UpdateModel(self, algs: Sequence["Design"], setting: Any):
        n = len(algs)
        actual = np.array([_mean_performance(alg) for alg in algs], dtype=float)
        approx = np.array([_mean_performance_approx(alg) for alg in algs], dtype=float)
        ind_new: set[int] = set()
        for i in range(n):
            for j in range(i + 1, n):
                res = (actual[i] < actual[j]) ^ (approx[i] < approx[j])
                if res:
                    ind_new.add(i)
                    ind_new.add(j)
        new_data = [algs[i] for i in sorted(ind_new)]
        self.data.extend(new_data)
        self.model = self.GetModel(self.data, setting)
        return self

    def UseModel(self, algs: Sequence["Design"], setting: Any) -> np.ndarray:
        features = self.UseEmbed(algs, setting)
        return (
            self.model.predict(features) if self.model else np.zeros(features.shape[0])
        )
