import math

import numpy as np
import pytest

from tinybamdp import bandit_dp, bamdp, beliefs, fitting, models, pomdp, task


# ------------------------------------------------------------------ beliefs
def test_beta_posterior_closed_form():
    b = beliefs.BetaBelief(2.0, 3.0)
    for o in [1, 1, 0, 1]:
        b = b.update(o)
    assert (b.a, b.b) == (5.0, 4.0)
    assert math.isclose(b.mean, 5 / 9)


def test_dirichlet_matches_beta():
    d = beliefs.DirichletBelief((3.0, 2.0)).update(1).update(1)   # (fail, win)
    b = beliefs.BetaBelief(2.0, 3.0).update(1).update(1)
    assert math.isclose(d.mean[1], b.mean)


def test_discrete_filter_bayes_rule():
    T = np.eye(2)
    O = np.array([[0.85, 0.15], [0.15, 0.85]])
    post, z = beliefs.discrete_filter(np.array([0.5, 0.5]), T, O, 0)
    assert np.allclose(post, [0.85, 0.15]) and math.isclose(z, 0.5)


# --------------------------------------------------------------- bandit DP
def test_graph_size_stars_and_bars():
    g = bandit_dp.bandit_graph(2, 20)
    assert g.n_full_states == math.comb(24, 4)
    assert g.n_states < g.n_full_states          # arm symmetry reduces the graph


def test_known_two_step_value():
    _, V = bandit_dp.solve(2, 2, 1.0, 1.0)
    assert math.isclose(V[0][0], 0.5 + 0.5 * 2 / 3 + 0.5 * 0.5)


def test_horizon_one_is_prior_mean():
    _, V = bandit_dp.solve(3, 1, 1.3, 0.7)
    assert math.isclose(V[0][0], 1.3 / 2.0)


@pytest.mark.parametrize("a0,b0", [(1.0, 1.0), (1.3, 0.7), (0.5, 2.0)])
def test_vectorised_dp_equals_generic_bamdp(a0, b0):
    K, T = 2, 5
    Q, _ = bandit_dp.solve(K, T, a0, b0)
    m = bandit_dp.bandit_graph(K, T)
    gen = bamdp.bandit_as_bamdp(K, a0, b0)
    rng = np.random.default_rng(0)
    for n in range(T):
        for i in rng.choice(len(m.codes[n]), size=min(5, len(m.codes[n])), replace=False):
            s = m.states[n][i]
            counts = tuple((int(s[2 * k + 1]), int(s[2 * k])) for k in range(K))  # (lose, win)
            assert np.allclose(gen.q_values(0, counts, T - n), Q[n][i])


def _policy_value(K, T, a0, b0, policy_values):
    """Exact expected return of a deterministic policy on the hyperstate graph."""
    g = bandit_dp.bandit_graph(K, T)
    V = np.zeros(len(g.codes[T]))
    for n in range(T - 1, -1, -1):
        S, F = g.S[n], g.F[n]
        p = (a0 + S) / (a0 + b0 + S + F)
        q = p * (1 + V[g.succ[n][:, :, 1]]) + (1 - p) * V[g.succ[n][:, :, 0]]
        a = policy_values(S, F, n).argmax(1)
        V = q[np.arange(len(q)), a]
    return V[0]


@pytest.mark.parametrize("T", [3, 6, 10])
def test_optimal_dominates_kg_and_greedy(T):
    K, a0, b0 = 2, 2.0, 2.0
    _, V = bandit_dp.solve(K, T, a0, b0)
    kg = _policy_value(K, T, a0, b0, lambda S, F, n: models.KnowledgeGradient.kg_values(a0, b0, S, F, T - n - 1))
    greedy = _policy_value(K, T, a0, b0, lambda S, F, n: (a0 + S) / (a0 + b0 + S + F))
    assert V[0][0] >= kg - 1e-12 and V[0][0] >= greedy - 1e-12


def test_kg_is_greedy_on_last_trial():
    S, F = np.array([[3.0, 1.0]]), np.array([[1.0, 0.0]])
    v = models.KnowledgeGradient.kg_values(1.0, 1.0, S, F, 0)
    assert np.allclose(v, (1 + S) / (2 + S + F))


# ------------------------------------------------------------- likelihoods
@pytest.mark.parametrize("name", list(models.MODELS))
def test_probabilities_normalised(name):
    rng = np.random.default_rng(1)
    m = models.get_model(name)
    prm = m.sample_params(rng)
    d = task.simulate(m, prm, task.sample_task(6, 2, rng), 8, rng)
    P = task.choice_probs(m, prm, d)
    assert np.allclose(P.sum(-1), 1.0) and (P >= 0).all()


def test_wsls_full_noise_is_uniform():
    rng = np.random.default_rng(2)
    m = models.get_model("WSLS")
    d = task.simulate(m, {"eps": 0.5}, task.sample_task(5, 2, rng), 6, rng)
    ll = task.log_likelihood(m, {"eps": 0.99999999}, d)
    assert math.isclose(ll, d.n_trials * math.log(0.5), rel_tol=1e-6)


def test_mle_recovers_wsls_noise():
    rng = np.random.default_rng(3)
    m = models.get_model("WSLS")
    d = task.simulate(m, {"eps": 0.2}, task.sample_task(200, 2, rng), 10, rng)
    f = fitting.fit("WSLS", d, n_starts=2, rng=rng)
    assert abs(f.params["eps"] - 0.2) < 0.05


# ------------------------------------------------------------------ POMDP
def test_tiger_known_values():
    P = pomdp.tiger()
    layers = P.solve(2, pomdp.belief_grid_2(201))
    v1, a1 = P.evaluate(layers[0], np.array([0.5, 0.5]))
    assert math.isclose(v1, -1.0) and a1 == 0                 # listen
    v1c, a1c = P.evaluate(layers[0], np.array([1.0, 0.0]))
    assert math.isclose(v1c, 10.0) and a1c == 2               # open right
    v2, _ = P.evaluate(layers[1], np.array([0.5, 0.5]))
    assert math.isclose(v2, -2.0)


# ------------------------------------------------------------ generic BAMDP
def test_bayes_beats_certainty_equivalent_on_average():
    """Information value: Bayes-adaptive planning should not lose to CE on average."""
    gen = bamdp.bandit_as_bamdp(2, 1.0, 1.0)
    rng = np.random.default_rng(4)
    diffs = []
    for _ in range(300):
        th = rng.beta(1, 1, size=2)
        P = np.zeros((2, 2, 2))
        P[:, :, 1], P[:, :, 0] = th, 1 - th
        seed = int(rng.integers(1 << 30))
        rb = gen.rollout(P, 0, 6, np.random.default_rng(seed), "bayes")
        rc = gen.rollout(P, 0, 6, np.random.default_rng(seed), "ce")
        diffs.append(rb - rc)
    assert np.mean(diffs) > -0.1


@pytest.mark.parametrize("T", [2, 3])
@pytest.mark.parametrize("a0,b0", [(1.0, 1.0), (2.0, 2.0), (0.7, 3.0), (0.3, 0.3)])
def test_kg_equals_optimal_choice_gap_for_short_two_arm_games(T, a0, b0):
    """For 2 arms and T <= 3 the KG and Bayes-optimal value gaps coincide (identical softmax)."""
    Q, _ = bandit_dp.solve(2, T, a0, b0)
    g = bandit_dp.bandit_graph(2, T)
    for n in range(T):
        s = g.states[n].astype(float)
        v = models.KnowledgeGradient.kg_values(a0, b0, s[:, 0::2], s[:, 1::2], T - n - 1)
        assert np.allclose(Q[n][:, 0] - Q[n][:, 1], v[:, 0] - v[:, 1])


# ------------------------------------------------------------- v0.2 additions


def test_kg_dbm_reduces_to_kg_when_gamma_is_one():
    rng = np.random.default_rng(5)
    kg, dbm = models.get_model("KG"), models.get_model("KG-DBM")
    prm = {"m": 0.5, "n": 4.0, "tau": 8.0}
    d = task.simulate(kg, prm, task.sample_task(30, 3, rng), 8, rng)
    P1 = task.choice_probs(kg, prm, d)
    P2 = task.choice_probs(dbm, {**prm, "gamma": 1.0}, d)
    assert np.abs(P1 - P2).max() < 5e-3          # grid discretisation error only


def test_kg_dbm_forgets_towards_prior():
    m = models.get_model("KG-DBM")
    prm = {"m": 0.5, "n": 2.0, "gamma": 0.5, "tau": 5.0}
    st = m.init(prm, 1, 2, 10)
    for _ in range(20):
        m.update(prm, st, 0, np.array([0]), np.array([1]))
    mean_arm0 = float(st["b"][0, 0] @ m._x)
    assert 0.5 < mean_arm0 < 0.85                # bounded by forgetting (no forgetting -> ~0.95)


def test_tie_aware_classification_at_T2():
    rng = np.random.default_rng(6)
    opt = models.get_model("OPT")
    d = task.simulate(opt, {"m": 0.5, "n": 3.0, "tau": 8.0}, task.sample_task(40, 2, rng), 2, rng)
    fits = {name: fitting.fit(name, d, 3, rng) for name in ("KG", "OPT")}
    w, margin = fitting.classify(fits, ["KG", "OPT"], tie_eps=1e-4)
    assert w == fitting.TIE and margin < 1e-4


def test_wilson_ci_and_inversion():
    lo, hi = fitting.wilson_ci(50, 100)
    assert lo < 0.5 < hi and math.isclose(lo + hi, 1.0, rel_tol=1e-9)
    M = np.array([[0.8, 0.2, 0.0], [0.4, 0.6, 0.0]])          # trailing tie column
    inv = fitting.inversion_matrix(M)
    assert np.allclose(inv.sum(0), 1.0) and math.isclose(inv[0, 0], 0.8 / 1.2)


def test_mixed_design_likelihood_is_sum():
    rng = np.random.default_rng(7)
    m = models.get_model("OPT")
    prm = {"m": 0.5, "n": 4.0, "tau": 6.0}
    blocks = task.simulate_design(m, prm, [(5, 3), (3, 8)], 2, rng)
    total = task.log_likelihood(m, prm, blocks)
    assert math.isclose(total, sum(task.log_likelihood(m, prm, b) for b in blocks))
    assert task.n_trials(blocks) == 5 * 3 + 3 * 8


def test_cross_validation_is_worse_than_in_sample():
    rng = np.random.default_rng(8)
    m = models.get_model("RW")
    d = task.simulate(m, {"eta": 0.3, "tau": 8.0}, task.sample_task(10, 2, rng), 10, rng)
    f = fitting.fit("RW", d, 3, rng)
    cv = fitting.cross_validate("RW", d, f, 0, rng)
    assert np.isfinite(cv) and cv >= f.nll - 1e-6


def test_bamdp_cache_is_read_only():
    gen = bamdp.bandit_as_bamdp(2, 1.0, 1.0)
    q = gen.q_values(0, gen.empty_counts(), 3)
    with pytest.raises(ValueError):
        q[0] = 99.0


def test_fit_reports_convergence():
    rng = np.random.default_rng(9)
    m = models.get_model("WSLS")
    d = task.simulate(m, {"eps": 0.3}, task.sample_task(20, 2, rng), 6, rng)
    f = fitting.fit("WSLS", d, 2, rng)
    assert isinstance(f.converged, bool) and 0 <= f.n_failed <= 2


def test_symmetric_lookup_matches_generic_bamdp_in_original_arm_order():
    """Canonical (sorted-arm) states must return Q-values in the caller's arm order."""
    K, T, a0, b0 = 3, 5, 1.3, 0.8
    Q, _ = bandit_dp.solve(K, T, a0, b0)
    g = bandit_dp.bandit_graph(K, T)
    gen = bamdp.bandit_as_bamdp(K, a0, b0)
    rng = np.random.default_rng(10)
    for _ in range(40):
        n = int(rng.integers(0, T))
        counts = rng.multinomial(n, np.ones(2 * K) / (2 * K))[None, :]
        q = g.lookup(n, counts, Q[n])[0]
        c = tuple((int(counts[0, 2 * k + 1]), int(counts[0, 2 * k])) for k in range(K))
        assert np.allclose(q, gen.q_values(0, c, T - n))


@pytest.mark.parametrize("name,K", [("KG", 2), ("KG", 3), ("OPT", 2), ("OPT", 4)])
def test_fast_value_path_matches_sequential_loop(name, K):
    rng = np.random.default_rng(11)
    m = models.get_model(name)
    prm = {"m": 0.45, "n": 3.0, "tau": 7.0}
    d = task.simulate(m, prm, task.sample_task(12, K, rng), 6, rng)
    P = task.choice_probs(m, prm, d)                      # sequential agent loop
    ll_loop = float(np.log(np.take_along_axis(P, d.choices[..., None], -1)).sum())
    assert math.isclose(task.log_likelihood(m, prm, d), ll_loop, rel_tol=1e-9)
