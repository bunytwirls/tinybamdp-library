"""Conjugate and discrete belief states.

All beliefs are immutable value objects: ``update`` returns a new belief, which
makes them safe to use as (part of) dictionary keys in memoised planners.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class BetaBelief:
    """Beta(a, b) belief over a Bernoulli rate (Beta-Bernoulli conjugacy)."""

    a: float = 1.0
    b: float = 1.0

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    @property
    def var(self) -> float:
        n = self.a + self.b
        return self.a * self.b / (n * n * (n + 1.0))

    def predictive(self) -> float:
        """P(next outcome = 1)."""
        return self.mean

    def update(self, outcome: int) -> "BetaBelief":
        return BetaBelief(self.a + outcome, self.b + (1 - outcome))


@dataclass(frozen=True)
class DirichletBelief:
    """Dirichlet(alpha) belief over a categorical distribution."""

    alpha: Tuple[float, ...]

    @staticmethod
    def uniform(k: int, conc: float = 1.0) -> "DirichletBelief":
        return DirichletBelief(tuple([conc] * k))

    @property
    def mean(self) -> np.ndarray:
        a = np.asarray(self.alpha, dtype=float)
        return a / a.sum()

    def predictive(self) -> np.ndarray:
        return self.mean

    def update(self, outcome: int) -> "DirichletBelief":
        a = list(self.alpha)
        a[outcome] += 1
        return DirichletBelief(tuple(a))


def discrete_filter(b: np.ndarray, T_a: np.ndarray, O_a: np.ndarray, o: int):
    """One step of the exact Bayes filter for a discrete POMDP.

    b    : (S,) current belief
    T_a  : (S, S') transition matrix for the chosen action
    O_a  : (S', O) observation matrix for the chosen action
    o    : observed symbol
    Returns (posterior belief, P(o | b, a)).
    """
    pred = b @ T_a                      # predict
    unnorm = pred * O_a[:, o]           # correct
    z = unnorm.sum()
    if z <= 0:
        raise ValueError("Observation has zero probability under the model.")
    return unnorm / z, z


__all__ = ["BetaBelief", "DirichletBelief", "discrete_filter"]
