"""Finite-horizon discrete POMDPs: belief filtering and point-based value iteration.

Value functions are represented by alpha-vectors.  Backups are performed at a
fixed set of belief points (PBVI; Pineau et al., 2003), so the result is a lower
bound on the optimal value function that is exact *at* the backed-up points.
On 2-state problems with a dense belief grid (e.g. the tiger task) the gap is
negligible; the tests check known optimal values at specific beliefs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .beliefs import discrete_filter


@dataclass
class TabularPOMDP:
    T: np.ndarray   # (A, S, S') transitions
    O: np.ndarray   # (A, S', O) observation probabilities
    R: np.ndarray   # (A, S) expected immediate reward
    gamma: float = 1.0
    state_names: Optional[List[str]] = None
    action_names: Optional[List[str]] = None
    obs_names: Optional[List[str]] = None

    @property
    def nS(self):
        return self.T.shape[1]

    @property
    def nA(self):
        return self.T.shape[0]

    @property
    def nO(self):
        return self.O.shape[2]

    def update(self, b, a, o):
        return discrete_filter(b, self.T[a], self.O[a], o)

    # -------------------------------------------------------- planning
    def solve(self, H: int, beliefs: np.ndarray):
        """Point-based backups for horizons 1..H.

        Returns a list ``layers`` where layers[h-1] = (alphas (n, S), actions (n,))
        is the value function with h steps to go.
        """
        alphas, acts = np.zeros((1, self.nS)), np.zeros(1, dtype=int)
        layers = []
        # g[a, o, i, s] = sum_s' T[a,s,s'] O[a,s',o] alpha_i(s')
        for _ in range(H):
            g = np.einsum("ast,ato,it->aois", self.T, self.O, alphas)
            # for each belief point b, action a and observation o pick best i
            score = np.einsum("aois,bs->aobi", g, beliefs)
            best = score.argmax(-1)                                    # (A, O, B)
            a_idx = np.arange(self.nA)[:, None, None]
            o_idx = np.arange(self.nO)[None, :, None]
            chosen = g[a_idx, o_idx, best]                              # (A, O, B, S)
            cand = self.R[:, None, :] + self.gamma * chosen.sum(1)      # (A, B, S)
            vals = np.einsum("abs,bs->ab", cand, beliefs)
            a_best = vals.argmax(0)                                     # (B,)
            new = cand[a_best, np.arange(len(beliefs))]
            new, uniq = np.unique(np.round(new, 10), axis=0, return_index=True)
            alphas, acts = new, a_best[uniq]
            layers.append((alphas, acts))
        return layers

    @staticmethod
    def evaluate(layer, b):
        alphas, acts = layer
        v = alphas @ b
        i = int(v.argmax())
        return float(v[i]), int(acts[i])

    def simulate(self, layers, b0, s0, rng):
        """Run the planned policy once; returns list of (belief, action, obs, reward)."""
        b, s, trace = np.asarray(b0, float), s0, []
        for h in range(len(layers), 0, -1):
            _, a = self.evaluate(layers[h - 1], b)
            r = self.R[a, s]
            s = int(rng.choice(self.nS, p=self.T[a, s]))
            o = int(rng.choice(self.nO, p=self.O[a, s]))
            trace.append((b.copy(), a, o, r))
            b, _ = self.update(b, a, o)
        return trace


def belief_grid_2(n: int = 201) -> np.ndarray:
    p = np.linspace(0, 1, n)
    return np.stack([p, 1 - p], 1)


def tiger(listen_acc: float = 0.85, r_listen: float = -1.0, r_good: float = 10.0,
          r_bad: float = -100.0) -> TabularPOMDP:
    """Classic tiger problem (Kaelbling, Littman & Cassandra, 1998).

    States: tiger-left (0), tiger-right (1).  Actions: listen, open-left, open-right.
    Opening a door resets the problem uniformly.
    """
    T = np.zeros((3, 2, 2))
    T[0] = np.eye(2)
    T[1] = T[2] = 0.5
    O = np.zeros((3, 2, 2))
    O[0] = [[listen_acc, 1 - listen_acc], [1 - listen_acc, listen_acc]]
    O[1] = O[2] = 0.5
    R = np.array([[r_listen, r_listen], [r_bad, r_good], [r_good, r_bad]])
    return TabularPOMDP(T, O, R, 1.0, ["tiger-left", "tiger-right"],
                        ["listen", "open-left", "open-right"], ["hear-left", "hear-right"])


__all__ = ["TabularPOMDP", "belief_grid_2", "tiger"]
