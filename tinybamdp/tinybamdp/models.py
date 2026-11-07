"""Cognitive models of choice in multi-game Bernoulli bandit tasks.

Every model exposes the same three-method agent interface, used both for
simulation and for likelihood evaluation, so one loop serves both:

    state = model.init(params, G, K, T)
    p     = model.probs(params, state, t)          # (G, K) choice probabilities
    model.update(params, state, t, choice, reward) # in-place belief update

All computation is vectorised over the G independent games of a subject.

Parameters are optimised in an unconstrained space ``z``; each model provides
``to_params(z)`` / ``to_z(params)`` and box bounds on z.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .bandit_dp import bandit_graph, solve


# --------------------------------------------------------------- transforms
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    return np.log(p) - np.log1p(-p)


def softmax(v: np.ndarray, tau: float) -> np.ndarray:
    x = tau * v
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class Param:
    name: str
    kind: str          # "unit" (0,1) via logit, or "pos" (0,inf) via log
    lo: float          # fitting bounds in natural space
    hi: float
    s_lo: float = None # simulation range (defaults to the fitting bounds)
    s_hi: float = None

    def sample_range(self):
        return (self.s_lo if self.s_lo is not None else self.lo,
                self.s_hi if self.s_hi is not None else self.hi)

    def to_nat(self, z):
        return _sig(z) if self.kind == "unit" else np.exp(z)

    def to_z(self, x):
        return _logit(x) if self.kind == "unit" else np.log(x)


class Model:
    name: str = "base"
    params: List[Param] = []

    # ---- parameter handling
    @property
    def n_params(self) -> int:
        return len(self.params)

    @property
    def param_names(self) -> List[str]:
        return [p.name for p in self.params]

    def to_params(self, z: Sequence[float]) -> Dict[str, float]:
        return {p.name: float(p.to_nat(zi)) for p, zi in zip(self.params, z)}

    def to_z(self, params: Dict[str, float]) -> np.ndarray:
        return np.array([p.to_z(params[p.name]) for p in self.params])

    def z_bounds(self) -> List[Tuple[float, float]]:
        return [(float(p.to_z(p.lo)), float(p.to_z(p.hi))) for p in self.params]

    def sample_params(self, rng: np.random.Generator) -> Dict[str, float]:
        """Uniform in z-space within the simulation range (log-uniform for positive params).

        Simulation ranges are narrower than fitting bounds so that simulated
        agents behave non-randomly, as recommended by Wilson & Collins (2019).
        """
        z = []
        for p in self.params:
            lo, hi = p.sample_range()
            z.append(rng.uniform(p.to_z(lo), p.to_z(hi)))
        return self.to_params(z)

    # ---- optional fast path ------------------------------------------------
    #: Parameters the decision values depend on.  A model that sets this and
    #: implements ``values(prm, data) -> (G, T, K)`` is a *value-softmax* model:
    #: its likelihood is softmax(tau * values), so the values are computed once
    #: per setting of ``value_params`` and reused for every tau (including the
    #: optimiser's finite-difference steps in tau).
    value_params: Tuple[str, ...] = ()

    def values(self, prm, data):
        raise NotImplementedError

    # ---- agent interface (overridden)
    def init(self, prm, G, K, T):
        raise NotImplementedError

    def probs(self, prm, st, t):
        raise NotImplementedError

    def update(self, prm, st, t, c, r):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(self.param_names)})"


# ------------------------------------------------------------ count helper
class _CountsMixin:
    """Keeps per-arm (success, failure) counts in state['counts'] (G, 2K)."""

    def _init_counts(self, st, G, K):
        st["counts"] = np.zeros((G, 2 * K), dtype=np.int64)

    def update(self, prm, st, t, c, r):
        g = np.arange(len(c))
        st["counts"][g, 2 * c + (1 - r)] += 1


def counts_before(data) -> np.ndarray:
    """(G, T, 2K) success/failure counts per arm *before* each trial (cached on ``data``)."""
    cached = getattr(data, "_counts_before", None)
    if cached is not None:
        return cached
    G, T, K = data.G, data.T, data.K
    inc = np.zeros((G, T, 2 * K), dtype=np.int64)
    gi, ti = np.meshgrid(np.arange(G), np.arange(T), indexing="ij")
    inc[gi, ti, 2 * data.choices + (1 - data.rewards)] = 1
    out = np.concatenate([np.zeros((G, 1, 2 * K), np.int64), np.cumsum(inc, axis=1)[:, :-1]], axis=1)
    data._counts_before = out
    return out


def _beta_ab(prm):
    """(m, n) -> (a, b): prior mean m and prior strength n = a + b."""
    return prm["m"] * prm["n"], (1.0 - prm["m"]) * prm["n"]


# ------------------------------------------------------------------ models
class RescorlaWagner(Model):
    """Model-free delta-rule learner (tabular Q-learning for a 1-state task)."""

    name = "RW"
    params = [Param("eta", "unit", 0.02, 0.98, 0.05, 0.8), Param("tau", "pos", 0.5, 100.0, 2.0, 20.0)]

    def init(self, prm, G, K, T):
        return {"Q": np.full((G, K), 0.5)}

    def probs(self, prm, st, t):
        return softmax(st["Q"], prm["tau"])

    def update(self, prm, st, t, c, r):
        g = np.arange(len(c))
        st["Q"][g, c] += prm["eta"] * (r - st["Q"][g, c])


class WSLS(Model):
    """Noisy win-stay/lose-shift (Wilson & Collins, 2019)."""

    name = "WSLS"
    params = [Param("eps", "unit", 0.01, 0.99, 0.05, 0.5)]

    def init(self, prm, G, K, T):
        return {"K": K, "last_c": np.zeros(G, dtype=np.int64), "last_r": np.zeros(G, dtype=np.int64)}

    def probs(self, prm, st, t):
        G = len(st["last_c"])
        K = st["K"]
        if t == 0:
            return np.full((G, K), 1.0 / K)
        eps = prm["eps"]
        g = np.arange(G)
        win = st["last_r"] == 1
        p = np.empty((G, K))
        # lose: shift uniformly among the other arms
        p[:] = (1 - eps) / (K - 1) + eps / K
        p[g, st["last_c"]] = eps / K
        # win: stay
        pw = np.full((G, K), eps / K)
        pw[g, st["last_c"]] = 1 - eps + eps / K
        p[win] = pw[win]
        return p

    def update(self, prm, st, t, c, r):
        st["last_c"], st["last_r"] = c.copy(), r.copy()


class KnowledgeGradient(_CountsMixin, Model):
    """Bayesian learner + one-step-lookahead Knowledge Gradient policy.

    Decision value for arm k on trial t (0-indexed) of a horizon-T game:
        D_k = mu_k + (T - t - 1) * KG_k
    (Zhang & Yu, 2013).  On the last trial D_k = mu_k, i.e. greedy.
    """

    name = "KG"
    params = [Param("m", "unit", 0.05, 0.95, 0.3, 0.7), Param("n", "pos", 0.2, 50.0, 1.0, 10.0),
              Param("tau", "pos", 0.5, 100.0, 3.0, 30.0)]

    def init(self, prm, G, K, T):
        st = {"T": T}
        self._init_counts(st, G, K)
        return st

    @staticmethod
    def kg_from_moments(mu, up, dn, remaining, scale=1.0):
        """Decision values mu + remaining * scale * KG from posterior means.

        mu : (G, K) current posterior means
        up : (G, K) mean of arm k after a hypothetical success on k
        dn : (G, K) mean of arm k after a hypothetical failure on k
        """
        K = mu.shape[1]
        if K > 1:
            order = np.argsort(mu, axis=1)
            top = np.take_along_axis(mu, order[:, -1:], 1)
            second = np.take_along_axis(mu, order[:, -2:-1], 1)
            is_top = np.arange(K)[None, :] == order[:, -1:]
            other = np.where(is_top, second, top)      # best mean among the other arms
        else:
            other = np.full_like(mu, -np.inf)
        best = np.maximum(mu, other)
        kg = mu * np.maximum(up, other) + (1 - mu) * np.maximum(dn, other) - best
        return mu + remaining * scale * kg

    @staticmethod
    def kg_values(a, b, S, F, remaining):
        A, B = a + S, b + F
        N = A + B
        return KnowledgeGradient.kg_from_moments(A / N, (A + 1) / (N + 1), A / (N + 1), remaining)

    def probs(self, prm, st, t):
        a, b = _beta_ab(prm)
        cnt = st["counts"]
        v = self.kg_values(a, b, cnt[:, 0::2], cnt[:, 1::2], st["T"] - t - 1)
        return softmax(v, prm["tau"])

    value_params = ("m", "n")

    def values(self, prm, data):
        """KG decision values for every trial at once (no loop over trials)."""
        a, b = _beta_ab(prm)
        cnt = counts_before(data).astype(float)                 # (G, T, 2K)
        G, T, _ = cnt.shape
        rem = np.broadcast_to((T - 1 - np.arange(T))[None, :, None], (G, T, 1)).reshape(-1, 1)
        v = self.kg_values(a, b, cnt[..., 0::2].reshape(G * T, -1), cnt[..., 1::2].reshape(G * T, -1), rem)
        return v.reshape(G, T, -1)


class BayesOptimal(_CountsMixin, Model):
    """Bayesian learner + exact Bayes-adaptive (BAMDP) planning + softmax.

    Q-values are computed by backward induction over the full hyperstate
    graph (see bandit_dp.solve) once per parameter setting and cached.
    """

    name = "OPT"
    params = [Param("m", "unit", 0.05, 0.95, 0.3, 0.7), Param("n", "pos", 0.2, 50.0, 1.0, 10.0),
              Param("tau", "pos", 0.5, 100.0, 3.0, 30.0)]

    def __init__(self):
        self._cache = {}          # (K, T) -> ((m, n), Q); one entry per horizon in use

    def _q(self, K, T, prm):
        key = (prm["m"], prm["n"])
        hit = self._cache.get((K, T))
        if hit is None or hit[0] != key:
            a, b = _beta_ab(prm)
            hit = (key, solve(K, T, a, b)[0])
            self._cache[(K, T)] = hit
        return hit[1]

    def init(self, prm, G, K, T):
        st = {"Q": self._q(K, T, prm), "graph": bandit_graph(K, T)}
        self._init_counts(st, G, K)
        return st

    def probs(self, prm, st, t):
        return softmax(st["graph"].lookup(t, st["counts"], st["Q"][t]), prm["tau"])

    value_params = ("m", "n")

    def values(self, prm, data):
        """Bayes-optimal Q-values (G, T, K) along the observed history."""
        Q = self._q(data.K, data.T, prm)
        g = bandit_graph(data.K, data.T)
        cnt = counts_before(data)
        return np.stack([g.lookup(t, cnt[:, t], Q[t]) for t in range(data.T)], axis=1)


class KnowledgeGradientDBM(Model):
    """Knowledge Gradient on a Dynamic Belief Model (forgetful Bayes).

    The learner assumes each arm's reward rate persists with probability gamma
    and is otherwise redrawn from the prior Beta(m n, (1 - m) n) (Yu & Cohen,
    2009; Zhang & Yu, 2013).  Beliefs are kept on a grid of M rates.  Between
    trials every arm's belief relaxes towards the prior:  b <- gamma b + (1 - gamma) p0.
    Because that relaxation is the same affine map for all arms, the KG term
    for next-trial means is gamma times the KG term on current posteriors.
    gamma -> 1 recovers the fixed-belief KG model (up to grid error).
    """

    name = "KG-DBM"
    params = [Param("m", "unit", 0.05, 0.95, 0.3, 0.7), Param("n", "pos", 0.2, 50.0, 1.0, 10.0),
              Param("gamma", "unit", 0.3, 0.999, 0.6, 0.95), Param("tau", "pos", 0.5, 100.0, 3.0, 30.0)]
    M = 101
    _x = (np.arange(M) + 0.5) / M

    def _prior(self, prm):
        a, b = _beta_ab(prm)
        x = self._x
        lp = (a - 1) * np.log(x) + (b - 1) * np.log1p(-x)
        p = np.exp(lp - lp.max())
        return p / p.sum()

    def init(self, prm, G, K, T):
        p0 = self._prior(prm)
        return {"b": np.broadcast_to(p0, (G, K, self.M)).copy(), "p0": p0, "T": T}

    def probs(self, prm, st, t):
        x, b = self._x, st["b"]
        e1 = b @ x
        e2 = b @ (x * x)
        mu = e1
        up = e2 / e1
        dn = (e1 - e2) / (1 - e1)
        v = KnowledgeGradient.kg_from_moments(mu, up, dn, st["T"] - t - 1, prm["gamma"])
        return softmax(v, prm["tau"])

    def update(self, prm, st, t, c, r):
        g = np.arange(len(c))
        b, x = st["b"], self._x
        lik = np.where(r[:, None] == 1, x[None, :], 1 - x[None, :])
        post = b[g, c] * lik
        b[g, c] = post / post.sum(1, keepdims=True)
        gam = prm["gamma"]
        st["b"] = gam * b + (1 - gam) * st["p0"]


MODELS = {m.name: m for m in (RescorlaWagner(), WSLS(), KnowledgeGradient(),
                                KnowledgeGradientDBM(), BayesOptimal())}


def get_model(name: str) -> Model:
    """Fresh instance (models with caches should not be shared across processes)."""
    return type(MODELS[name])()


def register_model(model: Model) -> None:
    """Add a user-defined model (anything implementing the agent interface)."""
    MODELS[model.name] = model


__all__ = ["counts_before", "Param", "Model", "RescorlaWagner", "WSLS", "KnowledgeGradient", "KnowledgeGradientDBM",
           "BayesOptimal", "MODELS", "get_model", "register_model", "softmax"]
