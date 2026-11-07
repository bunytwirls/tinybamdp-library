"""Generic finite-horizon Bayes-adaptive MDP with unknown tabular transitions.

The agent knows the reward function R[s, a, s'] but not the transition
probabilities.  Each (s, a) pair belongs to a *group* (``tie``); all pairs in a
group share one Dirichlet belief over next states.  The default is one group
per (s, a), i.e. the classic BAMDP of Duff (2002); tying lets you express
tasks such as bandits, in which an arm behaves the same from every state.

Hyperstate = (physical state, Dirichlet counts of every group).  The
Bayes-optimal Q-function is computed by memoised recursion over hyperstates,
which is exact but exponential in the horizon, so it is intended for small,
cognitively-sized tasks.  ``bandit_dp`` contains a vectorised special case.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

Counts = Tuple[Tuple[int, ...], ...]


class TabularBAMDP:
    def __init__(self, R: np.ndarray, prior: np.ndarray, tie: Optional[np.ndarray] = None):
        """
        R     : (S, A, S) reward for transition s --a--> s'
        prior : (S, A, S) Dirichlet concentrations; zeros mark impossible s'
        tie   : (S, A) integer group ids (default: one group per (s, a))
        """
        self.R = np.asarray(R, float)
        self.S, self.A, _ = self.R.shape
        prior = np.asarray(prior, float)
        self.tie = np.arange(self.S * self.A).reshape(self.S, self.A) if tie is None else np.asarray(tie)
        self.n_groups = int(self.tie.max()) + 1
        self.g_prior = np.zeros((self.n_groups, self.S))
        for s in range(self.S):
            for a in range(self.A):
                g = self.tie[s, a]
                if self.g_prior[g].sum() == 0:
                    self.g_prior[g] = prior[s, a]
                elif not np.allclose(self.g_prior[g], prior[s, a]):
                    raise ValueError("Tied (s, a) pairs must share the same prior.")
        self._memo: Dict = {}

    # ---- beliefs
    def empty_counts(self) -> Counts:
        return tuple((0,) * self.S for _ in range(self.n_groups))

    def predictive(self, s: int, a: int, counts: Counts) -> np.ndarray:
        g = self.tie[s, a]
        alpha = self.g_prior[g] + np.asarray(counts[g])
        return alpha / alpha.sum()

    def posterior_mean_P(self, counts: Counts) -> np.ndarray:
        P = np.empty((self.S, self.A, self.S))
        for s in range(self.S):
            for a in range(self.A):
                P[s, a] = self.predictive(s, a, counts)
        return P

    def bump(self, counts: Counts, s: int, a: int, s2: int) -> Counts:
        g = self.tie[s, a]
        row = list(counts[g])
        row[s2] += 1
        return counts[:g] + (tuple(row),) + counts[g + 1:]

    # ---- Bayes-optimal planning
    def q_values(self, s: int, counts: Counts, h: int) -> np.ndarray:
        """Bayes-optimal Q(s, counts, a) with h steps to go (a read-only cached array)."""
        key = (s, counts, h)
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        q = np.zeros(self.A)
        if h > 0:
            for a in range(self.A):
                p = self.predictive(s, a, counts)
                for s2 in np.nonzero(p)[0]:
                    v_next = self.q_values(s2, self.bump(counts, s, a, s2), h - 1).max() if h > 1 else 0.0
                    q[a] += p[s2] * (self.R[s, a, s2] + v_next)
        q.flags.writeable = False          # cached: callers must not modify it in place
        self._memo[key] = q
        return q

    def value(self, s: int, counts: Counts, h: int) -> float:
        return float(self.q_values(s, counts, h).max())

    # ---- certainty-equivalent planning (posterior-mean MDP, no information value)
    def ce_q_values(self, s: int, counts: Counts, h: int) -> np.ndarray:
        P = self.posterior_mean_P(counts)
        r = (P * self.R).sum(-1)                          # (S, A)
        V = np.zeros(self.S)
        Q = r.copy()
        for _ in range(h):
            Q = r + P @ V
            V = Q.max(1)
        return Q[s]

    # ---- simulation against a true environment
    def rollout(self, P_true: np.ndarray, s0: int, H: int, rng: np.random.Generator,
                agent: str = "bayes") -> float:
        s, counts, ret = s0, self.empty_counts(), 0.0
        for h in range(H, 0, -1):
            q = self.q_values(s, counts, h) if agent == "bayes" else self.ce_q_values(s, counts, h)
            a = int(rng.choice(np.flatnonzero(np.isclose(q, q.max()))))
            s2 = int(rng.choice(self.S, p=P_true[s, a]))
            ret += self.R[s, a, s2]
            counts = self.bump(counts, s, a, s2)
            s = s2
        return ret

    def clear_cache(self):
        self._memo.clear()


def bandit_as_bamdp(K: int, a0: float, b0: float) -> TabularBAMDP:
    """K-armed Bernoulli bandit as a 2-state tied BAMDP (state 1 = 'just won').

    Every arm k is one tied group whose Dirichlet over {lose, win} equals
    Beta(a0, b0) on winning; reward 1 for landing in the win state.
    """
    S, A = 2, K
    R = np.zeros((S, A, S))
    R[:, :, 1] = 1.0
    prior = np.zeros((S, A, S))
    prior[:, :, 0], prior[:, :, 1] = b0, a0
    tie = np.tile(np.arange(K), (S, 1))
    return TabularBAMDP(R, prior, tie)


__all__ = ["TabularBAMDP", "bandit_as_bamdp"]
