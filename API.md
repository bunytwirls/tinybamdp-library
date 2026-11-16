# tinybamdp API reference (version 0)

This page lists everything the `tinybamdp` library offers, with parameters, defaults and return values. The library depends only
on NumPy and SciPy. The separate `study` package (needs Matplotlib) holds the analyses and figures of the identifiability study
and is listed at the end.

Conventions used throughout:

* `G` games per subject, `T` trials per game, `K` arms.
* Arms are numbered `0 … K-1`; rewards are `0` or `1`.
* `rng` is a `numpy.random.Generator` (for example `np.random.default_rng(0)`).
* Model parameters are passed as a plain dict, for example `{"m": 0.5, "n": 4.0, "tau": 10.0}`.

Contents: [Top-level imports](#top-level-imports) · [`beliefs`](#tinybamdpbeliefs) · [`bandit_dp`](#tinybamdpbandit_dp) ·
[`bamdp`](#tinybamdpbamdp) · [`pomdp`](#tinybamdppomdp) · [`models`](#tinybamdpmodels) · [`task`](#tinybamdptask) ·
[`fitting`](#tinybamdpfitting) · [`study`](#the-study-package-not-part-of-the-installed-library) ·
[Data](#data-simulated-and-your-own) · [References](#references)

---

## Top-level imports

```python
import tinybamdp
tinybamdp.__version__        # "0"
```

| Name | What it is |
|---|---|
| `bandit_dp`, `bamdp`, `beliefs`, `fitting`, `models`, `pomdp`, `task` | the seven modules below |
| `MODELS` | dict of the built-in models: `"RW"`, `"WSLS"`, `"KG"`, `"KG-DBM"`, `"OPT"` |
| `get_model(name)` | a fresh instance of a model (see [`models`](#tinybamdpmodels)) |
| `register_model(model)` | add your own model to `MODELS` |
| `BanditData` | container for one subject's choices and rewards |
| `sample_task`, `simulate`, `simulate_design`, `log_likelihood` | task generation, simulation and likelihood (see [`task`](#tinybamdptask)) |

---

## `tinybamdp.beliefs`

Immutable belief states. `update` returns a new object, so beliefs can be used as dictionary keys.

### `BetaBelief(a=1.0, b=1.0)`
Beta(a, b) belief over the success rate of a Bernoulli arm.

| Member | Returns |
|---|---|
| `.a`, `.b` | the two Beta parameters |
| `.mean` | `a / (a + b)` |
| `.var` | `a b / ((a + b)² (a + b + 1))` |
| `.predictive()` | P(next outcome = 1), equal to `.mean` |
| `.update(outcome)` | new `BetaBelief(a + outcome, b + 1 - outcome)`; `outcome` is 0 or 1 |

### `DirichletBelief(alpha)`
Dirichlet belief over a categorical distribution; `alpha` is a tuple of concentrations.

| Member | Returns |
|---|---|
| `DirichletBelief.uniform(k, conc=1.0)` | Dirichlet with `k` equal concentrations `conc` |
| `.mean`, `.predictive()` | normalised `alpha` (NumPy array) |
| `.update(outcome)` | new belief with `alpha[outcome] + 1` |

### `discrete_filter(b, T_a, O_a, o)`
One exact Bayes-filter step for a discrete POMDP.

* `b`: (S,) current belief; `T_a`: (S, S') transition matrix of the chosen action; `O_a`: (S', O) observation matrix of the
  chosen action; `o`: observed symbol.
* Returns `(posterior, P(o | b, a))`. Raises `ValueError` if the observation has probability 0.

---

## `tinybamdp.bandit_dp`

Exact Bayes-optimal planning for finite-horizon K-armed Bernoulli bandits whose arms share one Beta(a0, b0) prior. The hyperstate
is the count vector `(s_1, f_1, …, s_K, f_K)`. States that differ only by a permutation of arms are stored once (sorted,
"canonical"), which divides the state count by up to K!; for K = 4, T = 15 there are 25,261 canonical states out of 490,314.

### `solve(K, T, a0, b0)`
Backward induction over the whole hyperstate graph.

* Returns `(Q, V)`: `Q[n]` has shape `(number of canonical states with n observations, K)` and is indexed by **sorted arm
  position**; `V[n]` has shape `(number of states,)`. `V[0][0]` is the Bayes-optimal expected total reward of a whole game.
* To get Q-values for arbitrary counts in their own arm order, use `bandit_graph(K, T).lookup(n, counts, Q[n])`.
* Reward is 1 per success. About 1.2–1.5 ms for K = 4, T = 15 once the graph is built (measured on a 2-core machine).

### `bandit_graph(K, T)`
Returns the cached `BanditGraph` for `(K, T)` (an `lru_cache` holding up to 32 graphs). Building the K = 4, T = 15 graph takes
under a second; it is built once and reused.

### `BanditGraph(K, T)`

| Member | Meaning |
|---|---|
| `.K`, `.T` | arms and horizon |
| `.states[n]` | (M_n, 2K) canonical count vectors with `n` observations |
| `.codes[n]` | sorted integer codes of those states |
| `.succ[n]` | (M_n, K, 2) index in layer `n+1` after pulling sorted position `j` with outcome `o` (1 = success) |
| `.S[n]`, `.F[n]` | (M_n, K) success and failure counts as floats |
| `.n_states` | number of canonical states stored |
| `.n_full_states` | number of hyperstates without the symmetry reduction, C(T + 2K, 2K) |
| `.canonical(n, counts)` | `(index in layer n, arm order)` for count vectors `counts` of shape (G, 2K) |
| `.lookup(n, counts, Qn)` | (G, K) Q-values for `counts` in the caller's arm order |

---

## `tinybamdp.bamdp`

Generic finite-horizon Bayes-adaptive MDP with known rewards and unknown tabular transitions (Duff, 2002). Exact, but the
number of hyperstates grows exponentially with the horizon, so it is meant for small tasks.

### `TabularBAMDP(R, prior, tie=None)`

| Argument | Shape | Meaning |
|---|---|---|
| `R` | (S, A, S) | reward for the transition s →a→ s' |
| `prior` | (S, A, S) | Dirichlet concentrations over next states; zeros mark impossible s' |
| `tie` | (S, A) int | group id per (s, a); all pairs in a group share one Dirichlet belief. Default: one group per pair. Tied pairs must have the same prior, or a `ValueError` is raised. |

Counts are tuples of tuples (one row of next-state counts per group), so they can be dictionary keys.

| Method | Returns |
|---|---|
| `.empty_counts()` | all-zero counts |
| `.predictive(s, a, counts)` | (S,) posterior predictive over next states |
| `.posterior_mean_P(counts)` | (S, A, S) posterior-mean transition tensor |
| `.bump(counts, s, a, s2)` | new counts after observing s →a→ s2 |
| `.q_values(s, counts, h)` | (A,) Bayes-optimal Q-values with `h` steps to go (memoised; the returned array is read-only) |
| `.value(s, counts, h)` | `max(q_values(...))` |
| `.ce_q_values(s, counts, h)` | (A,) certainty-equivalent Q-values: plans in the posterior-mean MDP and ignores learning |
| `.rollout(P_true, s0, H, rng, agent="bayes")` | total reward of one H-step episode against true transitions `P_true` (S, A, S); `agent="bayes"` uses `q_values`, any other value uses `ce_q_values`; ties are broken at random |
| `.clear_cache()` | empties the memo table (`._memo`, whose size is the number of hyperstates visited) |

### `bandit_as_bamdp(K, a0, b0)`
A K-armed Bernoulli bandit written as a 2-state tied BAMDP (state 1 = "just won"); every arm is one tied group with prior
Beta(a0, b0) on winning. Used in the tests to check `bandit_dp.solve` against the generic solver.

---

## `tinybamdp.pomdp`

Finite-horizon discrete POMDPs with alpha-vector value functions, computed by point-based value iteration (PBVI; Pineau et al.,
2003). The result is exact at the belief points that are backed up and a lower bound elsewhere.

### `TabularPOMDP(T, O, R, gamma=1.0, state_names=None, action_names=None, obs_names=None)`

| Argument | Shape | Meaning |
|---|---|---|
| `T` | (A, S, S') | transition probabilities |
| `O` | (A, S', O) | observation probabilities |
| `R` | (A, S) | expected immediate reward |
| `gamma` | float | discount factor (1.0 = undiscounted) |

| Member | Returns |
|---|---|
| `.nS`, `.nA`, `.nO` | numbers of states, actions, observations |
| `.update(b, a, o)` | `(posterior belief, P(o | b, a))` via `beliefs.discrete_filter` |
| `.solve(H, beliefs)` | list `layers` of length H; `layers[h-1] = (alphas (n, S), actions (n,))` is the value function with h steps to go. `beliefs` is an (N, S) array of belief points to back up at. |
| `TabularPOMDP.evaluate(layer, b)` | `(value, best action)` at belief `b` |
| `.simulate(layers, b0, s0, rng)` | runs the planned policy once from belief `b0` and true state `s0`; returns a list of `(belief, action, observation, reward)` |

### `belief_grid_2(n=201)`
(n, 2) evenly spaced belief points for a 2-state problem.

### `tiger(listen_acc=0.85, r_listen=-1.0, r_good=10.0, r_bad=-100.0)`
The tiger problem (Kaelbling et al., 1998). States: tiger-left (0), tiger-right (1). Actions: listen (0), open-left (1),
open-right (2). Observations: hear-left, hear-right. Opening a door resets the tiger uniformly.

---

## `tinybamdp.models`

Cognitive models of choice in multi-game Bernoulli bandit tasks. Every model implements one agent interface, used both to
simulate and to compute the likelihood, vectorised over the G games of a subject:

```python
state = model.init(params, G, K, T)
p     = model.probs(params, state, t)            # (G, K) choice probabilities on trial t (0-based)
model.update(params, state, t, choices, rewards) # in-place update; choices and rewards are (G,) int arrays
```

### Built-in models

`m` is the prior mean and `n` the prior strength of a Beta(m·n, (1 − m)·n) prior; `tau` is the softmax precision (inverse
temperature). "Fit bounds" are the box bounds used by `fitting.fit`; "simulation range" is where `sample_params` draws from
(narrower, so simulated agents are not random; Wilson & Collins, 2019).

| Name | Class | Parameters: fit bounds (simulation range) | Decision rule |
|---|---|---|---|
| `"RW"` | `RescorlaWagner` | `eta` 0.02–0.98 (0.05–0.8), `tau` 0.5–100 (2–20) | delta rule Q ← Q + η(r − Q), Q starts at 0.5 each game (Rescorla & Wagner, 1972); softmax on Q |
| `"WSLS"` | `WSLS` | `eps` 0.01–0.99 (0.05–0.5) | uniform on trial 0; then win-stay / lose-shift with lapse ε (Robbins, 1952; Wilson & Collins, 2019) |
| `"KG"` | `KnowledgeGradient` | `m` 0.05–0.95 (0.3–0.7), `n` 0.2–50 (1–10), `tau` 0.5–100 (3–30) | softmax on D_k = μ_k + (T − t − 1)·KG_k (Frazier et al., 2008; Zhang & Yu, 2013) |
| `"KG-DBM"` | `KnowledgeGradientDBM` | `m`, `n` as KG; `gamma` 0.3–0.999 (0.6–0.95); `tau` as KG | Dynamic Belief Model learner (Yu & Cohen, 2009): beliefs on a grid of 101 rates relax towards the prior, b ← γb + (1 − γ)p₀; softmax on γ-scaled KG values (Zhang & Yu, 2013) |
| `"OPT"` | `BayesOptimal` | `m`, `n`, `tau` as KG | softmax on exact Bayes-optimal Q-values from `bandit_dp.solve` |

Notes:

* `OPT` solves the full hyperstate graph once per `(m, n)` and `(K, T)` and caches it on the instance. The graph grows as
  C(T + 2K, 2K)/K! (roughly), so very long games with many arms become slow; the tutorial limits OPT with 4 arms to T ≤ 15.
* `KG-DBM` with `gamma → 1` reduces to `KG` up to grid error.

### Functions

| Function | Returns |
|---|---|
| `get_model(name)` | a **fresh** instance of the model registered under `name` (models with caches should not be shared between processes) |
| `register_model(model)` | adds `model` to `MODELS` under `model.name`. `get_model` builds new instances with `type(model)()`, so the class must be constructible without arguments. With `fitting.run_recovery(n_jobs > 1)` and the `"spawn"` start method, register the model at import time of a module the workers import. |
| `softmax(v, tau)` | softmax of `tau * v` along the last axis |
| `counts_before(data)` | (G, T, 2K) success/failure counts per arm before each trial (cached on `data`) |

### `Param(name, kind, lo, hi, s_lo=None, s_hi=None)`
A model parameter. `kind="unit"` for (0, 1) parameters (optimised on the logit scale), `kind="pos"` for positive ones (log
scale). `lo`, `hi` are the fitting bounds; `s_lo`, `s_hi` the simulation range (default: the fitting bounds).
Methods: `sample_range()`, `to_nat(z)`, `to_z(x)`.

### `Model` (base class for your own models)

| Member | Meaning |
|---|---|
| `name` | string key used in `MODELS` |
| `params` | list of `Param` |
| `n_params`, `param_names` | derived from `params` |
| `to_params(z)`, `to_z(params)`, `z_bounds()` | conversion between the dict of natural values and the unconstrained vector `z` |
| `sample_params(rng)` | random parameters, uniform in z within the simulation range |
| `init`, `probs`, `update` | the agent interface (must be overridden) |
| `value_params`, `values(prm, data)` | optional fast path: if a model sets `value_params` (the parameters its decision values depend on) and implements `values` returning (G, T, K) values, the likelihood is computed as softmax(τ · values) and the values are reused across τ. `KG` and `OPT` use it with `value_params = ("m", "n")`. |

Static helpers on `KnowledgeGradient`: `kg_from_moments(mu, up, dn, remaining, scale=1.0)` (decision values from current means
`mu` and the means after a hypothetical success `up` or failure `dn`) and `kg_values(a, b, S, F, remaining)` (decision values from
Beta prior `a, b` and count arrays `S, F` of shape (G, K)).

---

## `tinybamdp.task`

The multi-game bandit task: simulation and likelihood.

### `BanditData(choices, rewards, thetas=None, K=2)`
One subject's data.

| Field / member | Meaning |
|---|---|
| `choices` | (G, T) int array of chosen arms, 0 … K − 1 |
| `rewards` | (G, T) int array of 0/1 outcomes |
| `thetas` | optional (G, K) true reward rates (known only for simulated data) |
| `K` | number of arms. **Defaults to 2: set it explicitly for tasks with more arms.** |
| `.G`, `.T`, `.n_trials` | games, trials per game, total choices |
| `.subset(games)` | a `BanditData` restricted to the given game indices or boolean mask |

### Functions

| Function | Returns |
|---|---|
| `sample_task(G, K, rng, a=2.0, b=2.0)` | (G, K) reward rates drawn from the environment prior Beta(a, b) |
| `simulate(model, prm, thetas, T, rng)` | `BanditData` of one simulated subject playing `len(thetas)` games of `T` trials |
| `simulate_design(model, prm, design, K, rng, env_prior=(2.0, 2.0))` | list of `BanditData` for a mixed-horizon design given as `[(G1, T1), (G2, T2), ...]` |
| `log_likelihood(model, prm, data)` | log-likelihood of a `BanditData`, or the sum over a list of them (mixed horizons) |
| `choice_probs(model, prm, data)` | (G, T, K) model choice probabilities along the observed history |
| `n_trials(data)` | number of choices in a dataset or a list of datasets |

---

## `tinybamdp.fitting`

Maximum-likelihood fitting, cross-validation, and parameter and model recovery, following Wilson and Collins (2019).

### Constants

| Name | Value | Meaning |
|---|---|---|
| `TIE` | `"tie"` | label returned by `classify` when the best two models are within `tie_eps` |
| `WARM_START` | `{"OPT": "KG"}` | in `run_recovery`, OPT is also started from the KG estimate (the two share parameters) |
| `MP_CONTEXT` | `None` | start method for worker processes in `run_recovery(n_jobs > 1)`; `None` = platform default. Set `"spawn"` in notebooks. |

### `FitResult`
Dataclass with fields `model`, `params` (dict), `nll`, `bic` (= 2·NLL + k·ln N), `aic` (= 2·NLL + 2k), `n_failed` (restarts whose
optimiser did not report success), `converged` (did the best restart report success?), `cv_nll` (held-out NLL, if computed).

### Functions

| Function | Parameters (defaults) | Returns |
|---|---|---|
| `fit(model_name, data, n_starts=5, rng=None, maxiter=200, init=None)` | `data`: `BanditData` or list of them; `init`: optional parameter dict used as an extra start | `FitResult` from L-BFGS-B with `n_starts` random restarts in z-space |
| `cross_validate(model_name, data, full_fit=None, n_starts=0, rng=None)` | `data`: a single `BanditData`; `full_fit` defaults to `fit(..., 3 starts)` | leave-one-game-out held-out NLL (Zhang & Yu, 2013); each fold starts from the full-data estimate plus `n_starts` random starts |
| `classify(fits, models, criterion="bic", tie_eps=1e-6)` | `fits`: dict name → `FitResult`; `criterion`: any numeric `FitResult` field such as `"bic"`, `"aic"`, `"cv_nll"` | `(winner or "tie", margin between best and second best)` |
| `run_recovery(gen_models, fit_models, n_subjects, G, T, K=2, n_starts=4, seed=0, n_jobs=1, progress=None, env_prior=(2.0, 2.0), cv=False)` | simulates `n_subjects` per generating model with random parameters and fits every fit model; `progress(done, total)` is called after each subject; `cv=True` also fills `cv_nll` | list of `(generating model, true params, {fit model: FitResult})` |
| `confusion_matrix(results, models, criterion="bic", tie_eps=None)` | | P(best-fitting model ∣ generating model), rows = generating; with `tie_eps` an extra last column holds P(tie) |
| `inversion_matrix(M)` | `M` from `confusion_matrix` | P(generating model ∣ best-fitting model), assuming equal numbers per generating model |
| `recovery_summary(results, models, criterion="bic", tie_eps=1e-6, thresholds=(2.0, 6.0))` | | per generating model: `n`, `acc`, `acc_lo`, `acc_hi` (95% Wilson CI), `tie`, `median_margin`, `decisive>X` (share with margin > X), `acc|decisive>X` |
| `wilson_ci(k, n, z=1.96)` | | Wilson score interval for k successes out of n |
| `ci_halfwidth(p, n, z=1.96)` | | half-width of the Wilson interval for proportion p with n subjects |
| `n_for_halfwidth(h, p=0.5, z=1.96)` | | subjects per model for a half-width of about h |
| `failure_rate(results)` | | share of fits whose best restart did not converge |
| `param_table(results, model)` | | dict parameter → (true values, recovered values) for self-fits of `model` |
| `recovery_stats(results, model, log_params=("n", "tau"))` | | dict parameter → Pearson r, Spearman ρ and median bias (log scale for `log_params`) |

---

## The `study` package (not part of the installed library)

Lives in the repository at `tinybamdp/study/` and needs Matplotlib. `pip install tinybamdp` from Git installs only `tinybamdp`.

### `study.analysis`

| Function | Returns |
|---|---|
| `policy_disagreement(T, m=0.5, n=4.0, K=2, G=2000, tau=10.0, seed=0, env_prior=(2.0, 2.0))` | (T,) per-trial share of hyperstates (visited by a simulated OPT agent) where the KG and OPT argmax differ; states where either policy is tied are excluded |
| `kg_opt_gap_difference(K, T, a, b)` | max over all hyperstates and arms of the difference between OPT and KG value gaps; 0 means identical choice probabilities for every τ |
| `equivalence_table(Ks=(2, 3, 4), Ts=(2, 3, 4, 5), n_priors=200, seed=0, lo=0.1, hi=10.0)` | dict (K, T) → max gap difference over random shared Beta priors |
| `kg_opt_gap_difference_hetero(a, b, T)` | as above, but each arm has its own prior Beta(a_k, b_k); exact memoised recursion, so keep K and T small |
| `equivalence_table_hetero(Ks=(2, 3), Ts=(2, 3, 4), n_priors=100, seed=1, lo=0.1, hi=10.0)` | dict (K, T) → max gap difference over random per-arm priors |
| `likelihood_surface(model_name, data, fixed, x_name, x_vals, y_name, y_vals)` | (len(y_vals), len(x_vals)) log-likelihood grid over two parameters, the others held at `fixed` |
| `behavioural_signatures(data, a=1.0, b=1.0)` | dict with `win-stay`, `lose-shift`, `choose best mean` (posterior means under Beta(a, b)) and `choose least tried` |

### `study.plots`
`style()` sets Matplotlib defaults; `confusion(ax, M, labels, ...)`, `recovery_scatter(ax, true, rec, name, color, log=False)` and
`surface(ax, X, Y, Z, xlabel, ylabel, truth=None, mle=None)` draw the study figures. `COLORS` maps each model to a fixed colour
from the IBM colour-blind-safe palette; `SEQ` (ColorBrewer RdPu) and `DIV` (PiYG) are the colour maps.

### `study/run_all.py`
Runs the experiments E0–E10 and writes `tinybamdp/results/summary.json`, `tinybamdp/results/*.pkl` and
`tinybamdp/figures/*.png|pdf`. Commands are run from the repository root.

```bash
python tinybamdp/study/run_all.py                 # everything (long; run on a multi-core machine)
python tinybamdp/study/run_all.py --quick         # small smoke test
python tinybamdp/study/run_all.py --jobs 4        # worker processes (default 2)
python tinybamdp/study/run_all.py --only E3,E9    # selected experiments; results merge into summary.json
```

---

## Data: simulated and your own

`tinybamdp` ships no empirical dataset and downloads none. All data in the repository are simulated:

* `sample_task` draws each game's arm reward rates from an environment prior, Beta(2, 2) by default (`a`, `b` or `env_prior`
  change it);
* `simulate` lets a model choose arms and draws each reward as a Bernoulli outcome with the chosen arm's rate;
* `tinybamdp/results/` holds the outputs of the simulation study: `summary.json` (all reported numbers), pickled recovery results
  (`*.pkl`, lists of `(generating model, true params, {model: FitResult})`; loading them needs `tinybamdp` importable) and the
  console log `full_log.txt`.

To analyse real data, put each subject's choices and rewards into a `BanditData` (one row per game, equal game lengths; use a
list of `BanditData` for games of different lengths):

```python
import numpy as np
from tinybamdp import BanditData, fitting

choices = np.array([[0, 0, 1, 0], [1, 1, 1, 0]])   # G = 2 games, T = 4 trials, arms numbered from 0
rewards = np.array([[1, 0, 1, 1], [0, 1, 1, 0]])   # 0/1 outcomes
subject = BanditData(choices, rewards, K=2)         # set K to the number of arms in your task
fits = {m: fitting.fit(m, subject, n_starts=5) for m in ("RW", "WSLS", "KG", "OPT")}
print(fitting.classify(fits, list(fits), tie_eps=0.1))
```

The models assume a stationary Bernoulli bandit whose arms share one prior; tasks with other reward distributions need a new
model class (see [`Model`](#model-base-class-for-your-own-models)).

---

## References

Duff, M. O. (2002). *Optimal learning: Computational procedures for Bayes-adaptive Markov decision processes* [Doctoral
dissertation, University of Massachusetts Amherst].

Frazier, P. I., Powell, W. B., & Dayanik, S. (2008). A knowledge-gradient policy for sequential information collection. *SIAM
Journal on Control and Optimization, 47*(5), 2410–2439. https://doi.org/10.1137/070693424

Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). Planning and acting in partially observable stochastic domains.
*Artificial Intelligence, 101*(1–2), 99–134. https://doi.org/10.1016/S0004-3702(98)00023-X

Pineau, J., Gordon, G., & Thrun, S. (2003). Point-based value iteration: An anytime algorithm for POMDPs. In *Proceedings of the
Eighteenth International Joint Conference on Artificial Intelligence* (pp. 1025–1031). IJCAI.
https://ijcai.org/proceedings/2003/146

Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and
nonreinforcement. In A. H. Black & W. F. Prokasy (Eds.), *Classical conditioning II: Current research and theory* (pp. 64–99).
Appleton-Century-Crofts.

Robbins, H. (1952). Some aspects of the sequential design of experiments. *Bulletin of the American Mathematical Society, 58*(5),
527–535. https://doi.org/10.1090/S0002-9904-1952-09620-8

Wilson, R. C., & Collins, A. G. E. (2019). Ten simple rules for the computational modeling of behavioral data. *eLife, 8*,
Article e49547. https://doi.org/10.7554/eLife.49547

Yu, A. J., & Cohen, J. D. (2009). Sequential effects: Superstition or rational behavior? In *Advances in Neural Information
Processing Systems* (Vol. 21, pp. 1873–1880). Curran Associates.

Zhang, S., & Yu, A. J. (2013). Forgetful Bayes and myopic planning: Human learning and decision-making in a bandit setting. In
*Advances in Neural Information Processing Systems* (Vol. 26, pp. 2607–2615). Curran Associates.
