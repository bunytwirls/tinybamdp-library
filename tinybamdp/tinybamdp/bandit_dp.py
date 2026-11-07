"""Exact Bayes-adaptive planning for finite-horizon Bernoulli bandits.

The hyperstate of a K-armed Beta-Bernoulli bandit is the vector of
(success, failure) counts per arm.  For a game of horizon T, the reachable
hyperstates are the non-negative integer 2K-vectors whose entries sum to n <= T,
so the BAMDP is a finite layered DAG that can be solved exactly by backward
induction.

**Arm symmetry.**  All arms share one prior, so two hyperstates that differ
only by a permutation of the arms have the same value (with the Q-values
permuted accordingly).  The graph therefore stores one *canonical* state per
permutation class: the arms' (s, f) pairs sorted in ascending order.  This
divides the state count by up to K! (K=4, T=15: 490,314 -> 25,261 states; solve ~90 ms -> ~1.5 ms).
:meth:`BanditGraph.lookup` maps arbitrary count vectors to canonical states and
returns Q-values in the caller's original arm order.

The graph depends only on (K, T); it is built once and cached.  Each call to
:func:`solve` then costs one vectorised pass per layer, cheap enough to sit
inside a likelihood used for maximum-likelihood fitting.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from math import comb
from typing import List, Tuple

import numpy as np


def _compositions(n: int, d: int) -> np.ndarray:
    """All non-negative integer d-vectors summing to n (stars and bars)."""
    if d == 1:
        return np.array([[n]], dtype=np.int64)
    out = []
    for bars in combinations(range(n + d - 1), d - 1):
        prev, row = -1, []
        for b in bars:
            row.append(b - prev - 1)
            prev = b
        row.append(n + d - 2 - prev)
        out.append(row)
    return np.asarray(out, dtype=np.int64)


class BanditGraph:
    """Layered graph of canonical (arm-sorted) hyperstates for K arms, horizon T."""

    def __init__(self, K: int, T: int):
        self.K, self.T = K, T
        self.pb = T + 1                      # pair code: s * (T + 1) + f
        self.sb = (T + 1) ** 2               # state code base over sorted pair codes
        self.sw = self.sb ** np.arange(K, dtype=np.int64)
        self.states: List[np.ndarray] = []   # (M_n, 2K) canonical (s1, f1, ..., sK, fK)
        self.codes: List[np.ndarray] = []    # sorted canonical state codes
        self._pairs: List[np.ndarray] = []   # (M_n, K) sorted pair codes
        for n in range(T + 1):
            full = _compositions(n, 2 * K)
            pairs = np.sort(full[:, 0::2] * self.pb + full[:, 1::2], axis=1)
            pairs = np.unique(pairs, axis=0)
            c = pairs @ self.sw
            order = np.argsort(c)
            pairs, c = pairs[order], c[order]
            st = np.empty((len(pairs), 2 * K), dtype=np.int64)
            st[:, 0::2], st[:, 1::2] = pairs // self.pb, pairs % self.pb
            self._pairs.append(pairs)
            self.codes.append(c)
            self.states.append(st)
        # succ[n][:, j, o]: index in layer n+1 after pulling sorted position j with outcome o (1 = success)
        self.succ: List[np.ndarray] = []
        for n in range(T):
            P = self._pairs[n]
            nxt = np.empty((len(P), K, 2), dtype=np.int64)
            for j in range(K):
                for o, step in ((1, self.pb), (0, 1)):
                    Q = P.copy()
                    Q[:, j] += step
                    Q.sort(axis=1)
                    nxt[:, j, o] = np.searchsorted(self.codes[n + 1], Q @ self.sw)
            self.succ.append(nxt)
        self.S = [s[:, 0::2].astype(float) for s in self.states]
        self.F = [s[:, 1::2].astype(float) for s in self.states]

    @property
    def n_states(self) -> int:
        """Number of canonical hyperstates actually stored."""
        return int(sum(len(c) for c in self.codes))

    @property
    def n_full_states(self) -> int:
        """Number of hyperstates without the symmetry reduction: C(T + 2K, 2K)."""
        return comb(self.T + 2 * self.K, 2 * self.K)

    def canonical(self, n: int, counts: np.ndarray):
        """(index in layer n, arm order) for count vectors ``counts`` (G, 2K).

        ``order[g, j]`` is the original arm sitting at sorted position j.
        """
        counts = np.asarray(counts)
        pairs = counts[:, 0::2] * self.pb + counts[:, 1::2]
        order = np.argsort(pairs, axis=1, kind="stable")
        sp = np.take_along_axis(pairs, order, 1)
        return np.searchsorted(self.codes[n], sp @ self.sw), order

    def lookup(self, n: int, counts: np.ndarray, Qn: np.ndarray) -> np.ndarray:
        """Q-values (G, K) for count vectors ``counts`` in the caller's arm order."""
        idx, order = self.canonical(n, counts)
        out = np.empty((len(idx), self.K))
        np.put_along_axis(out, order, Qn[idx], 1)
        return out


@lru_cache(maxsize=32)
def bandit_graph(K: int, T: int) -> BanditGraph:
    return BanditGraph(K, T)


def solve(K: int, T: int, a0: float, b0: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Bayes-optimal Q-values for every canonical hyperstate.

    Each arm has an independent Beta(a0, b0) prior; reward is 1 for success.
    Returns (Q, V): Q[n] has shape (|layer n|, K), indexed by *sorted arm
    position*; V[n] has shape (|layer n|,).  Use ``bandit_graph(K, T).lookup``
    to obtain Q-values for arbitrary count vectors in their own arm order.
    """
    g = bandit_graph(K, T)
    V = [None] * (T + 1)
    Q = [None] * T
    V[T] = np.zeros(len(g.codes[T]))
    for n in range(T - 1, -1, -1):
        S, F = g.S[n], g.F[n]
        p = (a0 + S) / (a0 + b0 + S + F)                 # predictive P(success)
        Vn1 = V[n + 1]
        nxt = g.succ[n]
        q = p * (1.0 + Vn1[nxt[:, :, 1]]) + (1.0 - p) * Vn1[nxt[:, :, 0]]
        Q[n] = q
        V[n] = q.max(axis=1)
    return Q, V


__all__ = ["BanditGraph", "bandit_graph", "solve"]
