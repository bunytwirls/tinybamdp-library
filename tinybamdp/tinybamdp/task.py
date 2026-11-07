"""Multi-game Bernoulli bandit task, simulation and likelihood.

A *subject dataset* is G independent games of T trials on K arms.  In every
game the arm reward rates are redrawn from an environment prior Beta(2, 2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .models import Model


@dataclass
class BanditData:
    choices: np.ndarray        # (G, T) int
    rewards: np.ndarray        # (G, T) int in {0, 1}
    thetas: Optional[np.ndarray] = None   # (G, K) true reward rates
    K: int = 2

    @property
    def G(self):
        return self.choices.shape[0]

    @property
    def T(self):
        return self.choices.shape[1]

    @property
    def n_trials(self):
        return self.choices.size

    def subset(self, games) -> "BanditData":
        """A dataset restricted to the given game indices."""
        th = None if self.thetas is None else self.thetas[games]
        return BanditData(self.choices[games], self.rewards[games], th, self.K)


def n_trials(data) -> int:
    """Number of choices in a dataset or a list of datasets (mixed-horizon designs)."""
    return sum(d.n_trials for d in data) if isinstance(data, (list, tuple)) else data.n_trials


def sample_task(G: int, K: int, rng: np.random.Generator, a: float = 2.0, b: float = 2.0) -> np.ndarray:
    return rng.beta(a, b, size=(G, K))


def simulate(model: Model, prm: Dict[str, float], thetas: np.ndarray, T: int,
             rng: np.random.Generator) -> BanditData:
    G, K = thetas.shape
    st = model.init(prm, G, K, T)
    C = np.empty((G, T), dtype=np.int64)
    R = np.empty((G, T), dtype=np.int64)
    g = np.arange(G)
    for t in range(T):
        p = model.probs(prm, st, t)
        # vectorised categorical sampling
        c = (rng.random((G, 1)) > np.cumsum(p, axis=1)).sum(axis=1)
        c = np.minimum(c, K - 1)
        r = (rng.random(G) < thetas[g, c]).astype(np.int64)
        model.update(prm, st, t, c, r)
        C[:, t], R[:, t] = c, r
    return BanditData(C, R, thetas, K)


def log_likelihood(model: Model, prm: Dict[str, float], data) -> float:
    """Log-likelihood of a dataset, or the sum over a list of datasets of different horizons."""
    if isinstance(data, (list, tuple)):
        return float(sum(log_likelihood(model, prm, d) for d in data))
    if model.value_params:                       # fast path: values cached across tau
        return _value_softmax_ll(model, prm, data)
    G, T, K = data.G, data.T, data.K
    st = model.init(prm, G, K, T)
    g = np.arange(G)
    ll = 0.0
    for t in range(T):
        c, r = data.choices[:, t], data.rewards[:, t]
        p = model.probs(prm, st, t)[g, c]
        ll += np.log(np.maximum(p, 1e-300)).sum()
        model.update(prm, st, t, c, r)
    return float(ll)


def _value_softmax_ll(model: Model, prm: Dict[str, float], data: BanditData) -> float:
    key = (id(data),) + tuple(prm[k] for k in model.value_params)
    cache = model.__dict__.setdefault("_vcache", {})
    hit = cache.get(key)
    if hit is not None and hit[0] is data:       # the entry holds `data`, so its id cannot be reused
        V = hit[1]
    else:
        if len(cache) > 8:
            cache.clear()
        V = model.values(prm, data)
        cache[key] = (data, V)
    x = prm["tau"] * V
    x = x - x.max(-1, keepdims=True)
    lse = np.log(np.exp(x).sum(-1))
    chosen = np.take_along_axis(x, data.choices[..., None], -1)[..., 0]
    return float(np.maximum(chosen - lse, np.log(1e-300)).sum())


def simulate_design(model: Model, prm: Dict[str, float], design, K: int, rng: np.random.Generator,
                    env_prior=(2.0, 2.0)):
    """Simulate a mixed-horizon design given as [(G1, T1), (G2, T2), ...] -> list of BanditData."""
    return [simulate(model, prm, sample_task(G, K, rng, *env_prior), T, rng) for G, T in design]


def choice_probs(model: Model, prm: Dict[str, float], data: BanditData) -> np.ndarray:
    """(G, T, K) model choice probabilities along the observed history."""
    G, T, K = data.G, data.T, data.K
    st = model.init(prm, G, K, T)
    out = np.empty((G, T, K))
    for t in range(T):
        out[:, t] = model.probs(prm, st, t)
        model.update(prm, st, t, data.choices[:, t], data.rewards[:, t])
    return out


__all__ = ["BanditData", "n_trials", "sample_task", "simulate", "simulate_design", "log_likelihood", "choice_probs"]
