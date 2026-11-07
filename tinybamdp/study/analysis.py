"""Study-specific analyses built on tinybamdp: policy disagreement, KG/OPT
equivalence, likelihood surfaces and behavioural signatures."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from tinybamdp.bandit_dp import bandit_graph, solve
from tinybamdp.models import KnowledgeGradient, get_model
from tinybamdp.task import BanditData, log_likelihood, sample_task, simulate


def _tied(v: np.ndarray) -> np.ndarray:
    """Rows whose two largest entries are (numerically) equal."""
    if v.shape[1] < 2:
        return np.zeros(len(v), bool)
    s = np.sort(v, axis=1)
    return np.isclose(s[:, -1], s[:, -2])


def policy_disagreement(T: int, m: float = 0.5, n: float = 4.0, K: int = 2,
                        G: int = 2000, tau: float = 10.0, seed: int = 0,
                        env_prior: Tuple[float, float] = (2.0, 2.0)) -> np.ndarray:
    """Per-trial fraction of visited hyperstates where KG and OPT argmax differ.

    Hyperstates are visited by simulating the OPT agent; both policies use the
    same prior (m, n).  States where *either* policy is tied are excluded,
    because an argmax over a tie is arbitrary.  Returns an array of length T.
    """
    rng = np.random.default_rng(seed)
    opt = get_model("OPT")
    prm = {"m": m, "n": n, "tau": tau}
    d = simulate(opt, prm, sample_task(G, K, rng, *env_prior), T, rng)
    a, b = m * n, (1 - m) * n
    Q, _ = solve(K, T, a, b)
    g = bandit_graph(K, T)
    counts = np.zeros((G, 2 * K), dtype=np.int64)
    out = np.zeros(T)
    gi = np.arange(G)
    for t in range(T):
        q = g.lookup(t, counts, Q[t])
        v = KnowledgeGradient.kg_values(a, b, counts[:, 0::2], counts[:, 1::2], T - t - 1)
        valid = ~(_tied(q) | _tied(v))
        out[t] = ((q.argmax(1) != v.argmax(1)) & valid).mean()
        c, r = d.choices[:, t], d.rewards[:, t]
        counts[gi, 2 * c + (1 - r)] += 1
    return out


def kg_opt_gap_difference(K: int, T: int, a: float, b: float) -> float:
    """Max |(Q_k - Q_0) - (D_k - D_0)| over *all* hyperstates and arms.

    Zero means KG and Bayes-optimal values differ only by a per-state constant,
    so their softmax choice probabilities are identical for every tau.
    """
    Q, _ = solve(K, T, a, b)
    g = bandit_graph(K, T)
    mx = 0.0
    for n in range(T):
        s = g.states[n].astype(float)
        v = KnowledgeGradient.kg_values(a, b, s[:, 0::2], s[:, 1::2], T - n - 1)
        mx = max(mx, float(np.abs((Q[n] - Q[n][:, :1]) - (v - v[:, :1])).max()))
    return mx


def equivalence_table(Ks: Sequence[int] = (2, 3, 4), Ts: Sequence[int] = (2, 3, 4, 5),
                      n_priors: int = 200, seed: int = 0, lo: float = 0.1, hi: float = 10.0):
    """Max gap difference over ``n_priors`` log-uniform random Beta priors per (K, T)."""
    rng = np.random.default_rng(seed)
    priors = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(n_priors, 2)))
    priors[:3] = [(1, 1), (2, 2), (0.5, 0.5)]
    out = {}
    for K in Ks:
        for T in Ts:
            out[(K, T)] = max(kg_opt_gap_difference(K, T, a, b) for a, b in priors)
    return out


def kg_opt_gap_difference_hetero(a: Sequence[float], b: Sequence[float], T: int) -> float:
    """Like :func:`kg_opt_gap_difference`, but every arm k has its *own* prior Beta(a_k, b_k).

    Equivalently: any K independent Beta posteriors with T choices left.  Uses an
    exact memoised recursion (no arm symmetry), so keep K and T small.
    """
    from functools import lru_cache
    from itertools import product

    a, b = np.asarray(a, float), np.asarray(b, float)
    K = len(a)

    @lru_cache(maxsize=None)
    def V(counts, h):
        return 0.0 if h == 0 else float(Q(counts, h).max())

    @lru_cache(maxsize=None)
    def Q(counts, h):
        c = np.asarray(counts)
        q = np.empty(K)
        for k in range(K):
            s, f = c[2 * k], c[2 * k + 1]
            p = (a[k] + s) / (a[k] + b[k] + s + f)
            win, lose = list(counts), list(counts)
            win[2 * k] += 1
            lose[2 * k + 1] += 1
            q[k] = p * (1 + V(tuple(win), h - 1)) + (1 - p) * V(tuple(lose), h - 1)
        return q

    mx = 0.0
    for n in range(T):                               # every reachable hyperstate with T - n choices left
        for counts in product(range(n + 1), repeat=2 * K):
            if sum(counts) != n:
                continue
            q = Q(counts, T - n)
            c = np.asarray(counts, float)[None, :]
            v = KnowledgeGradient.kg_values(a[None, :], b[None, :], c[:, 0::2], c[:, 1::2], T - n - 1)[0]
            mx = max(mx, float(np.abs((q - q[0]) - (v - v[0])).max()))
    return mx


def equivalence_table_hetero(Ks: Sequence[int] = (2, 3), Ts: Sequence[int] = (2, 3, 4),
                             n_priors: int = 100, seed: int = 1, lo: float = 0.1, hi: float = 10.0):
    """Max gap difference over random *per-arm* priors (log-uniform a_k, b_k) per (K, T)."""
    rng = np.random.default_rng(seed)
    out = {}
    for K in Ks:
        priors = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(n_priors, 2, K)))
        for T in Ts:
            out[(K, T)] = max(kg_opt_gap_difference_hetero(pa, pb, T) for pa, pb in priors)
    return out


def likelihood_surface(model_name: str, data, fixed: Dict[str, float],
                       x_name: str, x_vals, y_name: str, y_vals) -> np.ndarray:
    """Log-likelihood on a 2-D grid of two parameters, others held at ``fixed``.

    ``data`` may be a list of datasets (mixed-horizon design)."""
    m = get_model(model_name)
    Z = np.empty((len(y_vals), len(x_vals)))
    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            Z[i, j] = log_likelihood(m, {**fixed, x_name: x, y_name: y}, data)
    return Z


def behavioural_signatures(data: BanditData, a: float = 1.0, b: float = 1.0) -> Dict[str, float]:
    """Zhang & Yu-style signatures: win-stay, lose-shift, P(best mean arm), P(least-tried arm).

    "Best mean" uses posterior means under a Beta(a, b) prior (default uniform).
    """
    C, R = data.choices, data.rewards
    G, T = C.shape
    K = data.K
    stay = C[:, 1:] == C[:, :-1]
    win = R[:, :-1] == 1
    S = np.zeros((G, K))
    N = np.zeros((G, K))
    greedy, least = [], []
    gi = np.arange(G)
    for t in range(T):
        if t > 0:
            mu = (S + a) / (N + a + b)
            amb = ~_tied(mu)
            greedy.append(C[amb, t] == mu[amb].argmax(1))
            ambn = ~_tied(-N)
            least.append(C[ambn, t] == N[ambn].argmin(1))
        S[gi, C[:, t]] += R[:, t]
        N[gi, C[:, t]] += 1

    def cat(xs):
        xs = [x for x in xs if len(x)]
        return float(np.concatenate(xs).mean()) if xs else np.nan

    return {"win-stay": float(stay[win].mean()), "lose-shift": float((~stay)[~win].mean()),
            "choose best mean": cat(greedy), "choose least tried": cat(least)}


__all__ = ["policy_disagreement", "kg_opt_gap_difference", "equivalence_table",
           "kg_opt_gap_difference_hetero", "equivalence_table_hetero",
           "likelihood_surface", "behavioural_signatures"]
