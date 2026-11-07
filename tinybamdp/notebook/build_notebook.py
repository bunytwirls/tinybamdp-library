"""Builds the two companion notebooks from the repository:

* notebook/tinybamdp_tutorial.ipynb            how to use the library (beliefs, planning, POMDPs, models, fitting)
* notebook/kg_vs_optimal_identifiability.ipynb the identifiability study (validation, design guideline, findings)

Neither notebook embeds the source code: both load the repository (local checkout, uploaded
tinybamdp.zip, or a git clone) and import ``tinybamdp`` and ``study`` from it.
"""
import json
import pathlib

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
TUT, STU = [], []
cells = TUT                                   # md()/code() append to whichever list is active
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

summary_path = ROOT / "results" / "summary.json"
S = json.loads(summary_path.read_text()) if summary_path.exists() else None


SETUP_MD = """
## 0 · Setup
Run this cell first. It finds the repository and makes `tinybamdp` (the library) and `study` importable, without copying any code:

1. **Local Jupyter**: open the notebook from the repository's `notebook/` folder; nothing else to do.
2. **Colab** (default): the cell clones [bunytwirls/tinybamdp-library](https://github.com/bunytwirls/tinybamdp-library) automatically.
3. **Offline / no GitHub access**: set `REPO_URL = ""` and upload `tinybamdp.zip` (the repository) when asked.
"""
SETUP_CODE = r"""
REPO_URL = "https://github.com/bunytwirls/tinybamdp-library.git"   # set to "" to use tinybamdp.zip instead
import os, sys, pathlib, subprocess, zipfile
try:                   # Colab: enable interactive widgets (no-op elsewhere)
    from google.colab import output
    output.enable_custom_widget_manager()
except Exception:
    pass
def _find_repo():
    for cand in (pathlib.Path.cwd().parent, pathlib.Path.cwd(), pathlib.Path("tinybamdp_repo")):
        if (cand / "tinybamdp" / "__init__.py").exists() and (cand / "study").exists():
            return cand.resolve()
    target = pathlib.Path("tinybamdp_repo")
    if REPO_URL:
        r = subprocess.run(["git", "clone", "-q", "--depth", "1", REPO_URL, str(target)], capture_output=True, text=True)
        if r.returncode == 0:
            return target.resolve()
        print("could not clone", REPO_URL, "- falling back to tinybamdp.zip:", r.stderr.strip()[:200])
    zp = pathlib.Path("tinybamdp.zip")
    if not zp.exists():
        try:
            from google.colab import files
            zp = pathlib.Path(next(iter(files.upload())))
        except ImportError:
            raise FileNotFoundError("Open this notebook from the repository, put tinybamdp.zip next to it, or set REPO_URL.")
    zipfile.ZipFile(zp).extractall(target)
    return target.resolve()
REPO = _find_repo()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
print("repository:", REPO)
"""

# ======================================================================= intro (tutorial)
md(r"""
# tinybamdp tutorial — exact Bayes-adaptive models of exploration

**How to use `tinybamdp`, a small from-scratch library for tabular decision-making under uncertainty.**

When people choose between uncertain options they must trade off *exploiting* what they know against *exploring* to learn more.
Normative accounts treat this as planning in **belief space**: a Bayes-adaptive MDP (BAMDP) when the dynamics are unknown,
and a POMDP when the state is hidden. This notebook builds both up step by step and shows how to wrap them as cognitive
models of choice, simulate agents and fit them to data.

The research study built on the library (when can myopic and Bayes-optimal exploration be told apart?) is in the companion
notebook **`kg_vs_optimal_identifiability.ipynb`**.

| § | Section | What you can do |
|---|---|---|
| 0 | Setup | load the library from the repository |
| – | Quick start | simulate one subject and compare models in five lines |
| 1 | Tests | run the unit tests |
| 2 | Beliefs | update a Beta belief with buttons |
| 3 | Exact BAMDP planning | inspect Bayes-optimal vs KG values in any hyperstate |
| 4 | Value of information | Bayes-adaptive vs certainty-equivalent planning on a generic BAMDP |
| 5 | POMDP: the tiger problem | belief-space value functions and simulated episodes |
| 6 | Cognitive models | simulate five agents (any number of arms) and compare behavioural signatures |
| 7 | Model fitting | fit all models to one simulated subject |

*Every interactive panel has buttons; long outputs are in scrollable boxes. Runs on a standard Colab CPU runtime.*
""")
md(SETUP_MD)
code(SETUP_CODE)

HELPERS = r"""
import importlib, time, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as W
from IPython.display import display, HTML, clear_output
import tinybamdp
for m in list(sys.modules):
    if m.startswith("tinybamdp"): importlib.reload(sys.modules[m])
from tinybamdp import (bandit_dp, bamdp, beliefs, fitting, models, pomdp, task,
                       get_model, simulate, sample_task, log_likelihood)
import study
for m in list(sys.modules):
    if m.startswith("study"): importlib.reload(sys.modules[m])
from study import analysis, plots
plots.style()
fitting.MP_CONTEXT = "spawn"     # forking a running Jupyter kernel can deadlock; spawn is safe
C = plots.COLORS
AUTORUN = os.environ.get("TB_AUTORUN") == "1"     # used for headless testing only

_CSS = '''<style>
.tb-scroll{max-height:300px;overflow:auto;border:1px solid #d9d8d3;border-radius:6px;background:#fff}
.tb-scroll table{border-collapse:collapse;font-size:12.5px;color:#0b0b0b;width:100%}
.tb-scroll th{position:sticky;top:0;background:#fdeef5;color:#0b0b0b;padding:5px 9px;text-align:right}
.tb-scroll td{padding:4px 9px;border-bottom:1px solid #eeede9;text-align:right}
.tb-scroll tr:hover td{background:#fdf5f9}
.tb-card{background:#fdf0f6;color:#0b0b0b;border-left:4px solid #DC267F;padding:8px 12px;border-radius:4px;margin:6px 0;font-size:13px}
.tb-warn{border-left-color:#FE6100;background:#fff4ec}
</style>'''
def table(df, height=300, index=True, fmt="{:.3f}"):
    html = df.to_html(index=index, float_format=lambda x: fmt.format(x), border=0)
    return HTML(_CSS + f'<div class="tb-scroll" style="max-height:{height}px">{html}</div>')
def card(text, warn=False):
    return HTML(_CSS + f'<div class="tb-card {"tb-warn" if warn else ""}">{text}</div>')
def outbox(h=None):
    lay = W.Layout(border="1px solid #e4e3df", padding="4px", overflow="auto")
    if h: lay.max_height = f"{h}px"
    return W.Output(layout=lay)
def button(desc, style="primary", icon=""):
    return W.Button(description=desc, button_style=style, icon=icon, layout=W.Layout(width="auto"))
SL = dict(continuous_update=False, style={"description_width": "110px"}, layout=W.Layout(width="330px"))
print("tinybamdp", tinybamdp.__version__)
"""
md("### Shared imports and UI helpers")
code(HELPERS)

# ================================================================== quick start
md(r"""
## Quick start (about a minute)
Simulate one subject who plans Bayes-optimally, fit the myopic (KG) and optimal (OPT) models, and ask which one the data favour.
The rest of the notebook explains each piece; the companion notebook `kg_vs_optimal_identifiability.ipynb` asks how
reliably this comparison works across task designs.
""")
code(r"""
rng = np.random.default_rng(0)
subject = simulate(get_model("OPT"), {"m": 0.5, "n": 4.0, "tau": 10.0},   # prior mean, prior strength, choice precision
                   sample_task(20, 4, rng), 15, rng)                       # 20 games x 15 trials on 4 arms
fits = {m: fitting.fit(m, subject, n_starts=3, rng=rng) for m in ("RW", "KG", "OPT")}
winner, margin = fitting.classify(fits, list(fits), tie_eps=0.1)
display(table(pd.DataFrame({m: {"NLL": f.nll, "BIC": f.bic, **f.params} for m, f in fits.items()}).T, height=160))
display(card(f"Best model by BIC: <b>{winner}</b> · margin to runner-up: <b>{margin:.1f}</b> BIC "
             f"({'strong' if margin > 6 else 'positive' if margin > 2 else 'weak'} evidence)"))
""")

# ======================================================================= tests
md("## 1 · Unit tests\nThe tests (core and study) check closed-form posteriors, a hand-computed two-trial bandit value, that the vectorised bandit solver equals the generic BAMDP solver, that the optimal policy dominates KG and greedy, likelihood normalisation, MLE recovery, and known tiger-problem values.")
code(r"""
run_tests = button("Run unit tests", "success", "check")
tests_out = outbox(320)
def _run_tests(_=None):
    import subprocess
    with tests_out:
        clear_output()
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(REPO / "tests")], capture_output=True, text=True, cwd=REPO)
        print(r.stdout[-3000:] or r.stderr[-3000:])
run_tests.on_click(_run_tests)
display(run_tests, tests_out)
_run_tests()
""")

# ===================================================================== beliefs
md(r"""
## 2 · Beliefs: Beta–Bernoulli conjugacy
An arm with unknown reward rate $\theta$ and prior $\mathrm{Beta}(a,b)$ has posterior $\mathrm{Beta}(a+s,\,b+f)$ after $s$ wins and
$f$ losses, with predictive $P(\text{win})=\frac{a+s}{a+b+s+f}$. The pair $(s,f)$ is therefore a sufficient statistic — the
**hyperstate** that BAMDP planning operates on. Below, the prior is parameterised by its mean $m=a/(a+b)$ and strength $n=a+b$,
the same parameterisation the fitted models use.
""")
code(r"""
from scipy.stats import beta as _beta
m_s = W.FloatSlider(value=0.5, min=0.05, max=0.95, step=0.05, description="prior mean m", **SL)
n_s = W.FloatLogSlider(value=4, base=10, min=-0.7, max=1.5, step=0.05, description="prior strength n", **SL)
bw, bl, br = button("Observe win", "success", "plus"), button("Observe loss", "danger", "minus"), button("Reset", "", "undo")
obs = {"s": 0, "f": 0, "hist": []}
b_out = outbox()
def draw_belief(*_):
    a0, b0 = m_s.value * n_s.value, (1 - m_s.value) * n_s.value
    a, b = a0 + obs["s"], b0 + obs["f"]
    with b_out:
        clear_output(wait=True)
        x = np.linspace(1e-3, 1 - 1e-3, 400)
        fig, ax = plt.subplots(figsize=(6, 2.6))
        ax.plot(x, _beta.pdf(x, a0, b0), color=plots.MUTED, ls="--", lw=1.5, label="prior")
        ax.plot(x, _beta.pdf(x, a, b), color=C["RW"], label="posterior")
        ax.fill_between(x, _beta.pdf(x, a, b), color=C["RW"], alpha=0.12)
        ax.set_xlabel("reward rate θ"), ax.set_ylabel("density"), ax.legend()
        plt.show()
        bb = beliefs.BetaBelief(a, b)
        display(card(f"<b>Beta({a:.2f}, {b:.2f})</b> · wins {obs['s']} · losses {obs['f']} · "
                     f"mean {bb.mean:.3f} · sd {np.sqrt(bb.var):.3f} · history: {''.join(obs['hist'][-40:]) or '–'}"))
def _obs(o):
    obs["s" if o else "f"] += 1; obs["hist"].append("W" if o else "L"); draw_belief()
bw.on_click(lambda _: _obs(1)); bl.on_click(lambda _: _obs(0))
def _reset(_): obs.update(s=0, f=0, hist=[]); draw_belief()
br.on_click(_reset)
for w in (m_s, n_s): w.observe(draw_belief, "value")
display(W.HBox([m_s, n_s]), W.HBox([bw, bl, br]), b_out); draw_belief()
""")

# ======================================================================= BAMDP
md(r"""
## 3 · Exact Bayes-adaptive planning for a bandit
For a $K$-armed bandit played for $T$ trials, the BAMDP state is the count vector $(s_1,f_1,\dots,s_K,f_K)$. Its Bellman equation is

$$Q_n(h,k)=p_k(h)\,[1+V_{n+1}(h+e^{s}_k)]+(1-p_k(h))\,V_{n+1}(h+e^{f}_k),\qquad V_n(h)=\max_k Q_n(h,k),$$

with $p_k(h)$ the posterior predictive. Reachable hyperstates with $n$ observations are the integer $2K$-vectors summing to $n$,
so the problem is a finite layered graph: $\binom{T+2K}{2K}$ states in total (10,626 for $K=2,T=20$). `bandit_dp` builds the
graph once per $(K,T)$ and then solves it with one vectorised pass per layer. Because all arms share one prior, states that differ
only by reordering the arms have the same value, so the graph stores one sorted (*canonical*) state per class: about 24× fewer
states with 4 arms. Solving takes about a millisecond, fast enough to run inside
every likelihood evaluation during fitting.

**Knowledge Gradient** (KG) approximates this by assuming that after one more observation no further learning happens:
$D_k=\mu_k+(T-t-1)\,\mathrm{KG}_k$, where $\mathrm{KG}_k$ is the expected one-step improvement in the best posterior mean.
Pick a hyperstate below and compare the two.
""")
code(r"""
T_s = W.IntSlider(value=10, min=2, max=30, description="horizon T", **SL)
pm_s = W.FloatSlider(value=0.5, min=0.1, max=0.9, step=0.05, description="prior mean m", **SL)
pn_s = W.FloatLogSlider(value=4, base=10, min=-0.5, max=1.3, step=0.05, description="prior strength n", **SL)
cnt = [W.IntSlider(value=v, min=0, max=10, description=d, **SL) for v, d in
       zip([2, 1, 0, 0], ["arm 1 wins", "arm 1 losses", "arm 2 wins", "arm 2 losses"])]
p_out = outbox()
def draw_planner(*_):
    T = T_s.value
    s1, f1, s2, f2 = (c.value for c in cnt)
    n_obs = s1 + f1 + s2 + f2
    with p_out:
        clear_output(wait=True)
        if n_obs >= T:
            display(card(f"This hyperstate has {n_obs} observations, so the game (T={T}) is over. Reduce the counts.", warn=True)); return
        a, b = pm_s.value * pn_s.value, (1 - pm_s.value) * pn_s.value
        t0 = time.perf_counter(); Q, V = bandit_dp.solve(2, T, a, b); dt = 1e3 * (time.perf_counter() - t0)
        g = bandit_dp.bandit_graph(2, T)
        h = np.array([[s1, f1, s2, f2]])
        q = g.lookup(n_obs, h, Q[n_obs])[0]
        S, F = h[:, 0::2].astype(float), h[:, 1::2].astype(float)
        kg = models.KnowledgeGradient.kg_values(a, b, S, F, T - n_obs - 1)[0]
        mu = (a + S[0]) / (a + b + S[0] + F[0])
        df = pd.DataFrame({"posterior mean": mu, "KG value D_k": kg, "Bayes-optimal Q": q,
                           "info bonus (Q − greedy value)": q - (mu + (T - n_obs - 1) * mu.max())},
                          index=["arm 1", "arm 2"])
        # the two value scales differ (KG: one-trial value, OPT: whole remaining game),
        # so plot each policy's preference for arm 1 over arm 2 and the implied choice probability
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.2))
        names, cols = ["KG", "OPT"], ["KG value D_k", "Bayes-optimal Q"]
        adv = [df[c].iloc[0] - df[c].iloc[1] for c in cols]
        axes[0].barh(names, adv, color=[C[n] for n in names], height=0.5)
        axes[0].axvline(0, color=plots.MUTED, lw=1); axes[0].set_title("preference: value(arm 1) − value(arm 2)")
        for tau, ls in [(5, "-"), (20, "--")]:
            axes[1].barh([f"{n}  τ={tau}" for n in names], [1 / (1 + np.exp(-tau * a)) for a in adv],
                         color=[C[n] for n in names], height=0.5, alpha=1 if tau == 5 else 0.55)
        axes[1].axvline(0.5, color=plots.MUTED, lw=1); axes[1].set_xlim(0, 1); axes[1].set_title("P(choose arm 1) under softmax")
        for ax in axes: ax.invert_yaxis()
        plt.tight_layout(); plt.show()
        display(card(f"KG picks <b>{df['KG value D_k'].idxmax()}</b> · Bayes-optimal picks <b>{df['Bayes-optimal Q'].idxmax()}</b>",
                     warn=df['KG value D_k'].idxmax() != df['Bayes-optimal Q'].idxmax()))
        display(table(df, index=True))
        display(card(f"Trial {n_obs + 1} of {T} · {T - n_obs - 1} trials remain after this one · "
                     f"graph: <b>{g.n_states:,}</b> canonical hyperstates ({g.n_full_states:,} before arm symmetry) · solve time <b>{dt:.2f} ms</b> · "
                     f"Bayes-optimal value of the whole game from scratch: <b>{V[0][0]:.3f}</b>"))
for w in [T_s, pm_s, pn_s, *cnt]: w.observe(draw_planner, "value")
display(W.HBox([W.VBox([T_s, pm_s, pn_s]), W.VBox(cnt)]), p_out); draw_planner()
""")

# ====================================================================== VOI
md(r"""
## 4 · Value of information on a generic BAMDP
`TabularBAMDP` handles any tabular MDP whose transitions are unknown, with a Dirichlet belief per $(s,a)$ pair (or per *tied group*
of pairs). Here: a **known safe arm** that pays with probability `p_safe` against an **unknown arm** with a uniform prior (mean 0.5).
A *certainty-equivalent* (CE) planner plans in the posterior-mean MDP. **Whenever `p_safe` > 0.5** (the unknown arm's prior mean)
it always picks the safe arm, so it never learns; at `p_safe` < 0.5 it would pick the unknown arm for exploitation, not exploration.
The Bayes-adaptive planner tries the unknown arm because what it learns can pay off over the remaining trials. That is the
exploration bonus that the BAMDP produces by itself.
""")
code(r"""
def safe_vs_unknown(p_safe=0.55, conc=1e6):
    # states: 0 = last trial lost, 1 = last trial won; actions: 0 = safe, 1 = unknown
    R = np.zeros((2, 2, 2)); R[:, :, 1] = 1
    prior = np.zeros((2, 2, 2))
    prior[:, 0] = [conc * (1 - p_safe), conc * p_safe]      # ≈ known
    prior[:, 1] = [1.0, 1.0]                                 # uniform belief
    tie = np.array([[0, 1], [0, 1]])
    return bamdp.TabularBAMDP(R, prior, tie)
H_s = W.IntSlider(value=10, min=1, max=16, description="horizon H", **SL)
ps_s = W.FloatSlider(value=0.55, min=0.3, max=0.8, step=0.05, description="p_safe", **SL)
N_s = W.IntSlider(value=400, min=100, max=2000, step=100, description="rollouts", **SL)
go = button("Run rollouts", "primary", "play")
v_out = outbox()
def run_voi(_=None):
    with v_out:
        clear_output(wait=True)
        H, ps = H_s.value, ps_s.value; M = safe_vs_unknown(ps); rng = np.random.default_rng(0)
        v_bayes = M.value(0, M.empty_counts(), H)
        ret = {"Bayes-adaptive": [], "certainty-equiv.": []}
        for _ in range(N_s.value):
            th = rng.uniform(); P = np.zeros((2, 2, 2)); P[:, 0] = [1 - ps, ps]; P[:, 1] = [1 - th, th]
            seed = int(rng.integers(1 << 30))
            ret["Bayes-adaptive"].append(M.rollout(P, 0, H, np.random.default_rng(seed), "bayes"))
            ret["certainty-equiv."].append(M.rollout(P, 0, H, np.random.default_rng(seed), "ce"))
        df = pd.DataFrame({k: [np.mean(v), np.std(v) / np.sqrt(len(v))] for k, v in ret.items()},
                          index=["mean return", "s.e."]).T
        fig, ax = plt.subplots(figsize=(5, 2.2))
        ax.barh(df.index, df["mean return"], xerr=1.96 * df["s.e."], color=[C["OPT"], C["WSLS"]], height=0.5)
        ax.set_xlabel(f"return over {H} trials (95% CI)"); plt.tight_layout(); plt.show()
        display(table(df))
        display(card(f"Exact Bayes-adaptive value <b>{v_bayes:.3f}</b> vs CE value <b>{max(ps, 0.5) * H:.3f}</b> "
                     f"(always safe) · hyperstates explored: {len(M._memo):,}"))
go.on_click(run_voi)
display(W.HBox([H_s, ps_s]), W.HBox([N_s, go]), v_out); run_voi()
""")

# ===================================================================== POMDP
md(r"""
## 5 · POMDP: the tiger problem
A tiger is behind the left or right door. Listening costs 1 and reports the correct side with probability `acc`; opening the
tiger's door costs 100, the other door pays 10, and opening resets the problem. The belief $b=P(\text{tiger-left})$ is updated by
the discrete Bayes filter, and the finite-horizon value function is piecewise-linear and convex in $b$ (a set of alpha-vectors),
computed here by point-based value iteration on a dense belief grid. With 2 states this matches the exact solution.
""")
code(r"""
th_s = W.IntSlider(value=4, min=1, max=12, description="horizon", **SL)
acc_s = W.FloatSlider(value=0.85, min=0.55, max=0.99, step=0.01, description="listen accuracy", **SL)
ep = button("Simulate an episode", "primary", "play")
t_out, e_out = outbox(), outbox(260)
ACOL = {0: C["RW"], 1: C["WSLS"], 2: C["KG"]}
_tiger = {}
def draw_tiger(*_):
    P = pomdp.tiger(acc_s.value); grid = pomdp.belief_grid_2(401)
    layers = P.solve(th_s.value, grid); _tiger.update(P=P, layers=layers)
    with t_out:
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(6.5, 2.8))
        b = grid[:, 0]
        for h, ls in [(th_s.value, "-"), (1, ":")]:
            vals = np.array([P.evaluate(layers[h - 1], g) for g in grid])
            if ls == "-":
                for a in range(3):
                    msk = vals[:, 1] == a
                    ax.scatter(b[msk], vals[msk, 0], s=6, color=ACOL[a], label=P.action_names[a])
            else:
                ax.plot(b, vals[:, 0], ls=ls, lw=1, color=plots.MUTED, label="horizon 1")
        ax.set_xlabel("belief P(tiger-left)"), ax.set_ylabel("value")
        ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), markerscale=3)
        ax.set_title(f"V_{th_s.value}(b): colour = optimal action"); plt.show()
        display(card(f"{len(layers[-1][0])} alpha-vectors at horizon {th_s.value}"))
def run_episode(_=None):
    P, layers = _tiger["P"], _tiger["layers"]
    rng = np.random.default_rng()
    s0 = int(rng.integers(2))
    tr = P.simulate(layers, np.array([.5, .5]), s0, rng)
    df = pd.DataFrame([{"step": i + 1, "P(tiger-left)": b[0], "action": P.action_names[a],
                        "observation": P.obs_names[o] if a == 0 else "(reset)", "reward": r}
                       for i, (b, a, o, r) in enumerate(tr)]).set_index("step")
    with e_out:
        clear_output(wait=True)
        display(card(f"Tiger was behind the <b>{P.state_names[s0]}</b> door · total reward <b>{df.reward.sum():.0f}</b>"))
        display(table(df, height=220, fmt="{:.2f}"))
for w in (th_s, acc_s): w.observe(draw_tiger, "value")
ep.on_click(run_episode)
display(W.HBox([th_s, acc_s]), t_out, ep, e_out); draw_tiger(); run_episode()
""")

# ==================================================================== models
md(r"""
## 6 · Five cognitive models of bandit choice
All models share one agent interface (`init`, `probs`, `update`) that is used both to **simulate** and to compute the
**likelihood**, so the simulation and fitting code paths cannot drift apart. Each subject plays $G$ games of $T$ trials on $K$
arms whose rates are redrawn from an environment prior (default $\mathrm{Beta}(2,2)$) every game.

| Model | Learning | Choice | Free parameters |
|---|---|---|---|
| **RW** | delta rule $Q\leftarrow Q+\eta(r-Q)$ | softmax | $\eta,\ \tau$ |
| **WSLS** | none | win-stay / lose-shift with lapse $\epsilon$ | $\epsilon$ |
| **KG** | Bayesian, prior $(m,n)$ | softmax on Knowledge-Gradient values | $m,\ n,\ \tau$ |
| **KG-DBM** | forgetful Bayes: rates persist with prob. $\gamma$, else redrawn from the prior (Dynamic Belief Model) | softmax on KG values | $m,\ n,\ \gamma,\ \tau$ |
| **OPT** | Bayesian, prior $(m,n)$ | softmax on exact BAMDP Q-values | $m,\ n,\ \tau$ |

Deterministic policies have zero-probability choices, which makes the log-likelihood $-\infty$, so every policy has a noise
term ($\tau$ or $\epsilon$).
""")
code(r"""
mdl = W.ToggleButtons(options=list(models.MODELS), value="OPT", description="model")
sT = W.IntSlider(value=10, min=2, max=30, description="trials / game T", **SL)
sG = W.IntSlider(value=24, min=4, max=200, description="games G", **SL)
sK = W.IntSlider(value=2, min=2, max=4, description="arms K", **SL)
ea_s = W.FloatSlider(value=2.0, min=0.5, max=5, step=0.5, description="env prior a", **SL)
eb_s = W.FloatSlider(value=2.0, min=0.5, max=5, step=0.5, description="env prior b", **SL)
seed_w = W.IntText(value=1, description="seed", layout=W.Layout(width="160px"))
pbox = W.VBox()
def _param_widgets(*_):
    ws = []
    for p in get_model(mdl.value).params:
        lo, hi = p.sample_range(); mid = (lo + hi) / 2 if p.kind == "unit" else np.sqrt(lo * hi)
        if p.kind == "unit":
            ws.append(W.FloatSlider(value=round(mid, 2), min=p.lo, max=p.hi, step=0.01, description=p.name, **SL))
        else:
            ws.append(W.FloatLogSlider(value=mid, base=10, min=np.log10(p.lo), max=np.log10(p.hi), step=0.02, description=p.name, **SL))
    pbox.children = ws
mdl.observe(_param_widgets, "value"); _param_widgets()
sim_b, cmp_b = button("Simulate subject", "primary", "play"), button("Compare all models", "info", "bar-chart")
s_out, d_out = outbox(), outbox(260)
CURRENT = {}
def _learning_curve(d):
    better = (d.thetas.argmax(1)[:, None] == d.choices)
    return better.mean(0)
def run_sim(_=None):
    m = get_model(mdl.value); prm = {w.description: w.value for w in pbox.children}
    rng = np.random.default_rng(seed_w.value)
    if mdl.value == "OPT" and sK.value == 4 and sT.value > 15:
        with s_out:
            clear_output(); display(card("OPT with 4 arms is limited to T ≤ 15 here (the exact solver grows as C(T+8, 8)).", warn=True))
        return
    d = simulate(m, prm, sample_task(sG.value, sK.value, rng, ea_s.value, eb_s.value), sT.value, rng)
    CURRENT.update(data=d, model=mdl.value, params=prm)
    with s_out:
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.plot(np.arange(1, d.T + 1), _learning_curve(d), "o-", color=C[mdl.value])
        ax.axhline(0.5, color=plots.MUTED, ls=":", lw=1)
        ax.set_ylim(0, 1); ax.set_xlabel("trial"), ax.set_ylabel("P(choose better arm)")
        ax.set_title(f"{mdl.value} · " + ", ".join(f"{k}={v:.2f}" for k, v in prm.items())); plt.show()
        sig = analysis.behavioural_signatures(d)
        display(table(pd.DataFrame(sig, index=[mdl.value]), height=90))
    with d_out:
        clear_output(wait=True)
        rows = [{"game": g + 1, **{f"θ arm{k + 1}": d.thetas[g, k] for k in range(d.K)},
                 "choices": "".join(str(c + 1) for c in d.choices[g]),
                 "rewards": "".join(str(r) for r in d.rewards[g]), "total": d.rewards[g].sum()} for g in range(d.G)]
        display(card("Raw data (one row per game; choices are arm numbers, rewards 0/1). This subject is used in § 7."))
        display(table(pd.DataFrame(rows).set_index("game"), height=220, fmt="{:.2f}"))
def compare(_=None):
    K = sK.value if not (sK.value == 4 and sT.value > 15) else 2
    rng = np.random.default_rng(seed_w.value); th = sample_task(200, K, rng, ea_s.value, eb_s.value); rows = {}
    for name in models.MODELS:
        m = get_model(name)
        prm = {p.name: (np.mean(p.sample_range()) if p.kind == "unit" else np.sqrt(np.prod(p.sample_range())))
               for p in m.params}
        rows[name] = analysis.behavioural_signatures(simulate(m, prm, th, sT.value, rng))
    df = pd.DataFrame(rows).T
    with s_out:
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(8, 2.8)); w = 0.16; x = np.arange(df.shape[1]); k = len(df.index)
        for i, name in enumerate(df.index):
            ax.bar(x + (i - (k - 1) / 2) * w, df.loc[name], w * 0.9, color=C[name], label=name)
        ax.set_xticks(x, df.columns); ax.set_ylim(0, 1.05); ax.legend(ncol=5, fontsize=8)
        ax.set_title(f"Behavioural signatures at mid-range parameters (K={K}, T={sT.value}, 200 games)"); plt.show()
        display(table(df, height=180))
sim_b.on_click(run_sim); cmp_b.on_click(compare)
display(mdl, W.HBox([W.VBox([sT, sG, sK, ea_s, eb_s, seed_w]), pbox]), W.HBox([sim_b, cmp_b]), s_out, d_out); run_sim()
""")

# =================================================================== fitting
md(r"""
## 7 · Fitting all models to one subject
Maximum likelihood with L-BFGS-B in an unconstrained parameter space (logit / log transforms) and several random restarts.
Models are compared with BIC $=2\,\mathrm{NLL}+k\ln N$. The subject is the one simulated in § 6. Fitting takes a few
seconds (longer with 4 arms), so it runs only when you press the button. Models within 0.1 BIC of the best are marked as ties,
and $\Delta$BIC > 2 / > 6 is conventionally read as positive / strong evidence (Kass & Raftery, 1995).
""")
code(r"""
rs_w = W.IntSlider(value=4, min=1, max=10, description="restarts", **SL)
fit_b = button("Fit all models", "primary", "cogs")
f_out = outbox()
def run_fit(_=None):
    d = CURRENT["data"]; rng = np.random.default_rng(0)
    with f_out:
        clear_output(wait=True)
        t0 = time.perf_counter()
        fits = {n: fitting.fit(n, d, rs_w.value, rng) for n in models.MODELS}
        winner, margin = fitting.classify(fits, list(fits), tie_eps=0.1)
        df = pd.DataFrame([{"model": n, "k": get_model(n).n_params, "NLL": f.nll, "BIC": f.bic,
                            "ΔBIC": f.bic - min(x.bic for x in fits.values()), "converged": f.converged,
                            "parameters": ", ".join(f"{k}={v:.2f}" for k, v in f.params.items())}
                           for n, f in fits.items()]).set_index("model").sort_values("BIC")
        strength = "a tie" if winner == fitting.TIE else ("strong" if margin > 6 else "positive" if margin > 2 else "weak")
        display(card(f"Generating model: <b>{CURRENT['model']}</b> ({', '.join(f'{k}={v:.2f}' for k, v in CURRENT['params'].items())}) · "
                     f"best by BIC: <b>{df.index[0] if winner != fitting.TIE else 'tie'}</b> · margin to runner-up {margin:.2f} "
                     f"(evidence: {strength}) · {d.n_trials} choices · {time.perf_counter() - t0:.1f} s",
                     warn=winner != CURRENT["model"]))
        display(table(df, height=200, fmt="{:.2f}"))
fit_b.on_click(run_fit)
display(W.HBox([rs_w, fit_b]), f_out)
if AUTORUN: run_fit()
""")

# ---------------------------------------------------- tutorial references
md(r"""
### Where to go next
* The identifiability study: **`kg_vs_optimal_identifiability.ipynb`**.
* Original papers for the models: Bellman (1957) MDPs; Duff (2002) BAMDPs; Kaelbling, Littman & Cassandra (1998) POMDPs and the
  tiger problem; Pineau, Gordon & Thrun (2003) point-based value iteration; Rescorla & Wagner (1972); Robbins (1952)
  win-stay/lose-shift; Frazier, Powell & Dayanik (2008) and Ryzhov, Powell & Frazier (2012) Knowledge Gradient; Yu & Cohen (2009)
  Dynamic Belief Model; Zhang & Yu (2013) KG as a model of human exploration; Wilson & Collins (2019) modelling workflow.
""")

# ================================================================ study notebook
cells = STU
md(r"""
# When can myopic and Bayes-optimal exploration be told apart?

**An identifiability study with `tinybamdp`.**

> **Research question.** Zhang & Yu (2013) found that human bandit choices are better described by a *myopic* Knowledge-Gradient
> (KG) policy than by the Bayes-optimal policy (OPT). Under what task designs can these two models be distinguished from choice
> data at all, and which of their parameters are identifiable?

The models, planners and fitting tools are introduced in the companion notebook **`tinybamdp_tutorial.ipynb`**. This notebook runs
the validation study interactively (§ 1) and reports the full-size reference results (§ 2), whose figures are embedded and display
without running anything.

| § | Section | What you can do |
|---|---|---|
| 0 | Setup | load the library and the study code from the repository |
| 1 | Validation study | equivalence check, tie-aware recovery with CIs, horizon sweep, likelihood surface, CV, mixed horizons, design heat map |
| 2 | Findings & limitations | reference results, figures, power table, take-aways, references |
""")
md(SETUP_MD)
code(SETUP_CODE)
md("### Shared imports and UI helpers")
code(HELPERS)

# ================================================================ validation
md(r"""
## 1 · Validation study
We follow Wilson & Collins (2019): simulate subjects from each model with parameters drawn from a behaviourally sensible range,
fit every model, then check **parameter recovery** (true vs fitted) and **model recovery**.

Model recovery here is **tie-aware** and reports **evidence strength**:
* if the best and second-best models are within 0.1 BIC, the subject counts as a *tie* rather than being assigned by `argmin`'s
  tie-breaking;
* next to accuracy we show 95% Wilson confidence intervals and the share of subjects with $|\Delta\mathrm{BIC}|>2$ (positive) and $>6$
  (strong evidence);
* the **inversion matrix** $P(\text{generating}\mid\text{best fit})$ answers the question you actually face with real data: *if
  model X wins, how likely is it that X generated the data?*

The research question is answered by a **horizon sweep** at a fixed budget of 240 choices per subject (games of length $T$,
$G = 240/T$ games). The cell below first establishes *where KG and OPT are identical*, which is where no amount of data can
separate them. Two facts explain the pattern: (i) with at most two choices left, KG's one-step lookahead is exact for **any**
posteriors; (ii) on the first trial of a game all arms share the prior, so both policies are indifferent. In a T-trial game
only trials 2 … T−2 can tell the models apart, at most a fraction (T−3)/T of the data; it is zero for T ≤ 3.
""")
code(r"""
# Where are KG and OPT behaviourally identical?  Max |gap_OPT - gap_KG| over all hyperstates and arms,
# maximised over random Beta priors (log-uniform on [0.1, 10]).  0 = identical choice probabilities for every tau.
eq_n = W.IntSlider(value=30, min=10, max=300, step=10, description="random priors", **SL)
eq_b = button("Re-run equivalence check", "primary", "refresh")
eq_out = outbox(260)
def run_equiv(_=None):
    with eq_out:
        clear_output(wait=True)
        t0 = time.perf_counter()
        tab = analysis.equivalence_table((2, 3, 4), (2, 3, 4, 5), n_priors=eq_n.value)
        het = analysis.equivalence_table_hetero((2, 3), (2, 3, 4), n_priors=max(10, eq_n.value // 3))
        df = pd.DataFrame([{"arms K": K, "T": T, "max diff": v} for (K, T), v in tab.items()]
                          ).pivot(index="arms K", columns="T", values="max diff")
        dh = pd.DataFrame([{"arms K": K, "choices left": T, "max diff": v} for (K, T), v in het.items()]
                          ).pivot(index="arms K", columns="choices left", values="max diff")
        display(card(f"<b>Shared prior</b> (all arms start equal): max difference in value gaps over all hyperstates and "
                     f"{eq_n.value} priors. Values ~1e-16 are floating-point zero."))
        display(table(df, height=170, fmt="{:.1e}"))
        display(card("<b>Per-arm priors</b> (equivalently, any posteriors) with this many choices left: exact only with ≤ 2 "
                     f"left. So the shared-prior T = 3 equality is a consequence of symmetry, not a deeper result · "
                     f"{time.perf_counter() - t0:.1f} s", warn=True))
        display(table(dh, height=140, fmt="{:.1e}"))
eq_b.on_click(run_equiv)
display(W.HBox([eq_n, eq_b]), eq_out); run_equiv()
""")

code(r"""
TIE_EPS = 0.1
ALL = list(models.MODELS)
Nw = W.IntSlider(value=20, min=5, max=150, step=5, description="subjects/model", **SL)
Jw = W.IntSlider(value=min(2, os.cpu_count() or 1), min=1, max=os.cpu_count() or 2, description="CPU workers", **SL)
Tw = W.IntSlider(value=10, min=3, max=30, description="T (baseline)", **SL)
Kw = W.IntSlider(value=2, min=2, max=4, description="arms K", **SL)
Bw = W.IntSlider(value=240, min=60, max=600, step=30, description="choices/subject", **SL)
Sw = W.IntSlider(value=3, min=1, max=8, description="restarts", **SL)
EAw = W.FloatSlider(value=2.0, min=0.5, max=5, step=0.5, description="env prior a", **SL)
EBw = W.FloatSlider(value=2.0, min=0.5, max=5, step=0.5, description="env prior b", **SL)
Mw = W.SelectMultiple(options=ALL, value=tuple(ALL), description="models", layout=W.Layout(width="230px", height="110px"))
Hw = W.SelectMultiple(options=[2, 3, 5, 8, 12, 20, 30], value=(2, 5, 12, 30), description="sweep T",
                      layout=W.Layout(width="200px", height="110px"))
prog = W.IntProgress(min=0, max=1, description="progress", bar_style="info", layout=W.Layout(width="420px"))
def _p(done, total): prog.max = total; prog.value = done
def _env(): return (EAw.value, EBw.value)
def _too_big(K, T, names):
    return "OPT" in names and K == 4 and T > 15
b1 = button("1 · Parameter + model recovery", "primary", "play")
b2 = button("2 · Horizon sweep (KG vs OPT)", "primary", "play")
b3 = button("3 · Likelihood surface", "primary", "play")
b4 = button("4 · Cross-validation vs BIC", "primary", "play")
b5 = button("5 · Mixed-horizon design", "primary", "play")
b6 = button("6 · Design heat map (K × T)", "success", "th")
o1, o2, o3, o4, o5, o6 = outbox(), outbox(), outbox(), outbox(), outbox(), outbox()

def summary_frame(res, names):
    S = fitting.recovery_summary(res, names, tie_eps=TIE_EPS)
    df = pd.DataFrame(S).T[["n", "acc", "acc_lo", "acc_hi", "tie", "decisive>2", "decisive>6", "acc|decisive>2", "median_margin"]]
    return df.rename(columns={"acc_lo": "95% CI lo", "acc_hi": "95% CI hi", "median_margin": "median ΔBIC to runner-up"})

def run_recovery(_=None):
    names = list(Mw.value); K, T = Kw.value, Tw.value
    with o1:
        clear_output(wait=True)
        if _too_big(K, T, names):
            display(card("OPT with 4 arms is limited to T ≤ 15.", warn=True)); return
        G = max(1, Bw.value // T); t0 = time.perf_counter()
        res = fitting.run_recovery(names, names, Nw.value, G, T, K=K, n_starts=Sw.value, seed=1,
                                   n_jobs=Jw.value, progress=_p, env_prior=_env())
        M = fitting.confusion_matrix(res, names, tie_eps=TIE_EPS)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
        plots.confusion(axes[0], M, names, title=f"P(best fit | generating)  K={K}, T={T}, G={G}")
        plots.confusion(axes[1], fitting.inversion_matrix(M), names, title="P(generating | best fit)")
        fig.tight_layout(); plt.show()
        display(card(f"Tie-aware recovery · {Nw.value} subjects/model · fit failures: {100 * fitting.failure_rate(res):.1f}% "
                     f"· {time.perf_counter() - t0:.0f} s"))
        display(table(summary_frame(res, names), height=220))
        n_par = sum(get_model(m).n_params for m in names); cols = 4; rows = int(np.ceil(n_par / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(10, 2.3 * rows)); axes = np.atleast_1d(axes).ravel(); k = 0
        stats = {}
        for mname in names:
            for pname, (tr, rc) in fitting.param_table(res, mname).items():
                plots.recovery_scatter(axes[k], tr, rc, f"{mname}:{pname}", C[mname], log=pname in ("n", "tau"))
                axes[k].set_xlabel(""); axes[k].set_ylabel(""); k += 1
            for pname, v in fitting.recovery_stats(res, mname).items():
                stats[f"{mname}.{pname}"] = v
        for ax in axes[k:]: ax.axis("off")
        fig.tight_layout(); plt.show()
        display(card("Parameter recovery (log scale for n, τ). Bias = median(recovered − true)."))
        display(table(pd.DataFrame(stats).T, height=260))
        corr = {}
        for mname in [m for m in ("KG", "KG-DBM", "OPT") if m in names]:
            tab = fitting.param_table(res, mname)
            e = {p: (np.log(v[1]) - np.log(v[0])) if p in ("n", "tau") else v[1] - v[0] for p, v in tab.items()}
            corr[mname] = {"corr(err n, err τ)": np.corrcoef(e["n"], e["tau"])[0, 1],
                           "corr(err m, err τ)": np.corrcoef(e["m"], e["tau"])[0, 1]}
        if corr:
            display(card("Correlation of estimation errors across subjects. A large positive value means the two parameters trade off."))
            display(table(pd.DataFrame(corr).T, height=140))

def run_sweep(_=None):
    K = Kw.value
    with o2:
        clear_output(wait=True)
        Hs = sorted(Hw.value)
        if K == 4: Hs = [T for T in Hs if T <= 15]
        rows, dis = {}, {}
        for T in Hs:
            r = fitting.run_recovery(["KG", "OPT"], ["KG", "OPT"], Nw.value, max(1, Bw.value // T), T, K=K, n_starts=Sw.value,
                                     seed=100 + T, n_jobs=Jw.value, progress=_p, env_prior=_env())
            S = fitting.recovery_summary(r, ["KG", "OPT"], tie_eps=TIE_EPS)
            rows[T] = {"P(KG|KG)": S["KG"]["acc"], "KG lo": S["KG"]["acc_lo"], "KG hi": S["KG"]["acc_hi"],
                       "P(OPT|OPT)": S["OPT"]["acc"], "OPT lo": S["OPT"]["acc_lo"], "OPT hi": S["OPT"]["acc_hi"],
                       "tie": (S["KG"]["tie"] + S["OPT"]["tie"]) / 2,
                       "|ΔBIC|>2": (S["KG"]["decisive>2"] + S["OPT"]["decisive>2"]) / 2,
                       "|ΔBIC|>6": (S["KG"]["decisive>6"] + S["OPT"]["decisive>6"]) / 2}
            dis[T] = 100 * analysis.policy_disagreement(T, K=K, G=1000, env_prior=_env()).mean()
        df = pd.DataFrame(rows).T; df["% argmax disagreement"] = pd.Series(dis)
        fig, axes = plt.subplots(1, 3, figsize=(11, 3))
        for g, mk in [("KG", "o"), ("OPT", "s")]:
            y = df[f"P({g}|{g})"]
            axes[0].errorbar(Hs, y, yerr=[y - df[f"{g} lo"], df[f"{g} hi"] - y], fmt=mk + "-", capsize=3, color=C[g], label=f"{g}→{g}")
        axes[0].plot(Hs, df["tie"], "^:", color=plots.MUTED, label="tie"); axes[0].set_ylim(0, 1.02); axes[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
        axes[0].set_title(f"Recovery, {Bw.value} choices (95% CI)")
        axes[1].plot(Hs, df["|ΔBIC|>2"], "o-", color=plots.INK, label=">2"); axes[1].plot(Hs, df["|ΔBIC|>6"], "s--", color=plots.MUTED, label=">6")
        axes[1].set_ylim(0, 1.02); axes[1].legend(fontsize=7); axes[1].set_title("Share with decisive evidence")
        axes[2].plot(Hs, df["% argmax disagreement"], "o-", color=plots.INK); axes[2].set_title("KG vs OPT disagreement (%)")
        for ax in axes: ax.set_xscale("log"); ax.set_xticks(Hs, Hs); ax.minorticks_off(); ax.set_xlabel("horizon T")
        fig.tight_layout(); plt.show()
        display(table(df, height=240))

def run_surface(_=None):
    with o3:
        clear_output(wait=True)
        rng = np.random.default_rng(11); true = {"m": .5, "n": 4.0, "tau": 5.0}
        d = simulate(get_model("OPT"), true, sample_task(max(1, Bw.value // Tw.value), 2, rng, *_env()), Tw.value, rng)
        xs, ys = np.geomspace(.5, 20, 30), np.geomspace(.5, 50, 30)
        Z = analysis.likelihood_surface("OPT", d, {"m": .5}, "n", xs, "tau", ys)
        i, j = np.unravel_index(Z.argmax(), Z.shape)
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        cs = plots.surface(ax, xs, ys, Z, "prior strength n", "softmax precision τ", truth=(4, 5), mle=(xs[j], ys[i]))
        fig.colorbar(cs, ax=ax, label="log-lik − max"); ax.set_title("OPT log-likelihood, m fixed (white line: 95% region)")
        plt.tight_layout(); plt.show()

def run_cv(_=None):
    names = list(Mw.value); K, T = Kw.value, Tw.value
    with o4:
        clear_output(wait=True)
        if _too_big(K, T, names):
            display(card("OPT with 4 arms is limited to T ≤ 15.", warn=True)); return
        t0 = time.perf_counter(); G = max(2, Bw.value // T)
        res = fitting.run_recovery(names, names, max(3, Nw.value // 3), G, T, K=K, n_starts=Sw.value, seed=600,
                                   n_jobs=Jw.value, progress=_p, env_prior=_env(), cv=True)
        Mb = fitting.confusion_matrix(res, names, "bic", tie_eps=TIE_EPS)
        Mc = fitting.confusion_matrix(res, names, "cv_nll", tie_eps=TIE_EPS / 2)
        agree = np.mean([fitting.classify(f, names, "bic", TIE_EPS)[0] == fitting.classify(f, names, "cv_nll", TIE_EPS / 2)[0]
                         for _, _, f in res])
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
        plots.confusion(axes[0], Mb, names, title="BIC")
        plots.confusion(axes[1], Mc, names, title="Leave-one-game-out CV", xlabel="best model (held-out NLL)")
        fig.tight_layout(); plt.show()
        display(card(f"{max(3, Nw.value // 3)} subjects/model (CV costs ≈ G extra warm-started fits per model) · "
                     f"BIC and CV pick the same model for <b>{100 * agree:.0f}%</b> of subjects · {time.perf_counter() - t0:.0f} s"))

def run_mixed(_=None):
    with o5:
        clear_output(wait=True)
        B = Bw.value
        designs = {"fixed T=5": [(B // 5, 5)], "fixed T=20": [(B // 20, 20)], "mixed 5 + 20": [(B // 10, 5), (B // 40, 20)]}
        opt = get_model("OPT"); rng = np.random.default_rng(700); rows = {}
        total = len(designs) * Nw.value; done = 0
        for name, des in designs.items():
            e_n, e_t = [], []
            for _ in range(Nw.value):
                tr = opt.sample_params(rng)
                f = fitting.fit("OPT", task.simulate_design(opt, tr, des, 2, rng, _env()), Sw.value, rng)
                e_n.append(np.log(f.params["n"] / tr["n"])); e_t.append(np.log(f.params["tau"] / tr["tau"]))
                done += 1; _p(done, total)
            rows[name] = {"corr(err n, err τ)": np.corrcoef(e_n, e_t)[0, 1],
                          "RMSE log n": np.sqrt(np.mean(np.square(e_n))), "RMSE log τ": np.sqrt(np.mean(np.square(e_t)))}
        df = pd.DataFrame(rows).T
        fig, ax = plt.subplots(figsize=(6, 2.2))
        ax.barh(df.index, df["corr(err n, err τ)"], color=C["OPT"], height=0.5); ax.set_xlim(0, 1); ax.invert_yaxis()
        ax.set_xlabel("corr(err n, err τ)"); ax.set_title("Does mixing horizons break the trade-off?"); plt.tight_layout(); plt.show()
        display(table(df, height=150))

def run_design(_=None):
    with o6:
        clear_output(wait=True)
        Ks, Ts = [2, 3, 4], [4, 8, 15]
        acc = np.zeros((len(Ks), len(Ts))); dec = np.zeros_like(acc); t0 = time.perf_counter()
        for i, K in enumerate(Ks):
            for j, T in enumerate(Ts):
                r = fitting.run_recovery(["KG", "OPT"], ["KG", "OPT"], max(4, Nw.value // 2), max(1, Bw.value // T), T, K=K,
                                         n_starts=Sw.value, seed=900 + 10 * K + T, n_jobs=Jw.value, progress=_p, env_prior=_env())
                S_ = fitting.recovery_summary(r, ["KG", "OPT"], tie_eps=TIE_EPS)
                acc[i, j] = np.mean([v["acc"] for v in S_.values()]); dec[i, j] = np.mean([v["decisive>2"] for v in S_.values()])
        fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
        for ax, M, title in [(axes[0], acc, "KG vs OPT accuracy"), (axes[1], dec, "share with |ΔBIC| > 2")]:
            ax.imshow(M, cmap=plots.SEQ, vmin=0, vmax=1, aspect="auto"); ax.grid(False)
            ax.set_xticks(range(len(Ts)), Ts); ax.set_yticks(range(len(Ks)), Ks); ax.set_xlabel("trials per game T"); ax.set_ylabel("arms K")
            for a_ in range(len(Ks)):
                for b_ in range(len(Ts)):
                    ax.text(b_, a_, f"{M[a_, b_]:.2f}", ha="center", va="center", color="white" if M[a_, b_] > .55 else plots.INK)
            ax.set_title(title)
        fig.tight_layout(); plt.show()
        display(card(f"{max(4, Nw.value // 2)} subjects per model per cell, {Bw.value} choices each · {time.perf_counter() - t0:.0f} s. "
                     "§ 2 shows the full-size version (60 per cell, five horizons)."))

b1.on_click(run_recovery); b2.on_click(run_sweep); b3.on_click(run_surface); b4.on_click(run_cv); b5.on_click(run_mixed)
b6.on_click(run_design)
display(card("Settings apply to all six analyses. Defaults are sized for a few minutes each on Colab; § 2 shows the full-size reference run."),
        W.HBox([W.VBox([Nw, Jw, Sw, Bw]), W.VBox([Tw, Kw, EAw, EBw]), Mw, Hw]), prog,
        W.HBox([b1, b2, b3]), W.HBox([b4, b5, b6]), o1, o2, o3, o4, o5, o6)
if AUTORUN:
    Nw.value = 4; Sw.value = 2; Mw.value = ("RW", "KG", "KG-DBM", "OPT"); run_recovery()
    Hw.value = (2, 12); run_sweep(); run_surface(); Nw.value = 6; run_cv(); Nw.value = 6; run_mixed(); Nw.value = 4; run_design()
""")
if S:
    _sw, _k4, _cv, _mx, _tc, _s2 = S["E3_horizon_sweep"], S["E5_k4"], S["E6_cv"], S["E7_mixed"], S["E4_error_correlations"], S["E2_summary"]
    _inv = S["E2_inversion"]
    _k4s = _k4["summary"]
    _des = S["E9_design"]["cells"]
    _het = S["E8_equivalence_hetero"]
    TAKEAWAYS = rf"""
### 2.8 Take-aways

1. **Design guideline (§ 2.1).** To tell myopic (KG) from Bayes-optimal exploration, use games of about 12 or more trials and at
   least three arms; going from three to four arms adds little ({_des['K3_T12']['decisive>2']:.2f} vs {_des['K4_T12']['decisive>2']:.2f}
   decisive at T = 12), while the horizon matters a lot. At a fixed budget of 240 choices, accuracy is {_des['K2_T4']['acc']:.2f} with 2 arms × 4 trials
   but {_des['K4_T15']['acc']:.2f} with 4 arms × 15 trials, and the share of subjects with positive evidence (|ΔBIC| > 2) rises from
   {_des['K2_T4']['decisive>2']:.2f} to {_des['K4_T15']['decisive>2']:.2f}.
2. **Why short games fail.** With at most two choices left, KG is *exactly* optimal for any posteriors, and on the first trial both
   policies are indifferent. Only trials 2 … T−2 can carry information, a fraction (T−3)/T: none at T ≤ 3 (every subject is a tie),
   40% at T = 5, 85% at T = 20. The shared-prior T = 3 equality is therefore a consequence of symmetry, not a new result:
   with *unequal* posteriors and three choices left, the two policies differ (max gap difference {_het['K2_T3']:.2f} for 2 arms;
   unequal for {S['E8_share_unequal_T3_hetero_K2']:.0%} of random per-arm priors).
3. **Raw accuracy overstates the evidence at mid horizons.** On 2 arms, accuracy rises from {_sw['5']['acc']:.2f} (T = 5) to
   {_sw['20']['acc']:.2f} (T = 20), but only {_sw['5']['decisive>2']:.0%} of subjects at T = 5 have positive evidence
   ({_sw['5']['decisive>6']:.0%} strong). T = 20 and T = 30 are statistically indistinguishable.
4. **Zhang & Yu's regime works** (4 arms, 15 trials, 20 games, N = {_k4['N']} per model): KG is recovered for {_k4s['KG']['acc']:.0%}
   [95% CI {_k4s['KG']['acc_lo']:.2f}–{_k4s['KG']['acc_hi']:.2f}] and OPT for {_k4s['OPT']['acc']:.0%}
   [{_k4s['OPT']['acc_lo']:.2f}–{_k4s['OPT']['acc_hi']:.2f}]. KG and OPT are rarely confused with each other there
   (KG→OPT {_k4['confusion'][2][4]:.0%}, OPT→KG {_k4['confusion'][4][2]:.0%}); OPT's remaining errors go mostly to RW
   ({_k4['confusion'][4][0]:.0%}).
5. **On short 2-arm games, a delta-rule win is weak evidence.** At T = 10, RW wins for {S['E2_confusion'][2][0]:.0%} of KG,
   {S['E2_confusion'][3][0]:.0%} of KG-DBM and {S['E2_confusion'][4][0]:.0%} of OPT subjects; when RW wins, RW generated the data only
   {_inv[0][0]:.0%} of the time (an OPT win is right {_inv[4][4]:.0%} of the time).
6. **Prior strength and decision noise are confounded** (error correlation {_tc['KG']['n~tau']:.2f} KG, {_tc['KG-DBM']['n~tau']:.2f}
   KG-DBM, {_tc['OPT']['n~tau']:.2f} OPT). Mixing horizons within a subject does not fix this ({_mx['mixed 5 + 20']['n~tau']:.2f} mixed vs
   {_mx['fixed T=5']['n~tau']:.2f} / {_mx['fixed T=20']['n~tau']:.2f} fixed); longer games only make both more precise. Report n and τ
   jointly, or fix one of them.
7. **The forgetful KG-DBM is hard to identify on stationary bandits** (recovered {_s2['KG-DBM']['acc']:.0%}; γ r =
   {S['E1_param_recovery']['KG-DBM']['gamma']['pearson_r']:.2f}).
8. **Cross-validation and BIC disagree for {1 - _cv['agreement']:.0%} of subjects** (N = {_cv['N']} per model): leave-one-game-out CV
   penalises complexity less, favouring KG-DBM more and RW less.

**Limitations.** All results come from simulated data. Simulation parameter ranges were set by hand, and the confusion matrices
depend on them. The tie threshold (|ΔBIC| < {S['tie_eps']}) is a practical choice. Fits are per subject (no hierarchical model). OPT
with a forgetful learner is not implemented (its beliefs are no longer counts).

**Next steps.** Fit real bandit data (e.g., Steyvers et al., 2009); test designs with forced-choice "free information" trials
(Wilson et al., 2014) against the n–τ trade-off; use adaptive design optimisation (Myung & Pitt, 2009; Cavagnaro et al., 2010);
add an I-POMDP module.
"""
# ================================================================== findings
import base64
import io

def _attach(name, width=1100):
    """Downscaled PNG from figures/ as a notebook attachment (renders without running anything)."""
    from PIL import Image
    im = Image.open(ROOT / "figures" / f"{name}.png").convert("RGB")
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return {f"{name}.png": {"image/png": base64.b64encode(buf.getvalue()).decode()}}

def md_fig(text, names):
    att = {}
    for n in names:
        att.update(_attach(n))
    body = text.strip("\n") + "\n\n" + "\n\n".join(f"![{n}](attachment:{n}.png)" for n in names)
    cell = nbf.v4.new_markdown_cell(body)
    cell["attachments"] = att
    cells.append(cell)

def fmt_mat(M, rows, cols):
    head = "| gen \\ fit | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols)
    body = "\n".join(f"| **{r}** | " + " | ".join("–" if (v is None or v != v or v < 0) else f"{v:.2f}" for v in row) + " |"
                     for r, row in zip(rows, M))
    return head + "\n" + body

if S:
    MOD = S["models"]
    cols = MOD + ["tie"]
    sw = S["E3_horizon_sweep"]
    e0 = S["E0_prior_recovery"]["recovered"]
    tc = S["E4_error_correlations"]
    k4 = S["E5_k4"]; cv = S["E6_cv"]; mx = S["E7_mixed"]; eq = S["E8_equivalence"]; eqh = S["E8_equivalence_hetero"]
    info = S["E8_informative_fraction"]; pw = S["E10_power"]; des = S["E9_design"]
    s2 = S["E2_summary"]
    sweep_rows = "\n".join(
        f"| {T} | {S['budget'] // int(T)} | {v['per_model']['KG']['acc']:.2f} [{max(0, v['per_model']['KG']['acc_lo']):.2f}, {v['per_model']['KG']['acc_hi']:.2f}] | "
        f"{v['per_model']['OPT']['acc']:.2f} [{max(0, v['per_model']['OPT']['acc_lo']):.2f}, {v['per_model']['OPT']['acc_hi']:.2f}] | "
        f"{v['tie']:.2f} | {v['decisive>2']:.2f} | {v['decisive>6']:.2f} | {100 * info[T]:.0f}% | {100 * S['E3_policy_disagreement'][T]:.1f} |"
        for T, v in sw.items())
    rec = S["E1_param_recovery"]
    rec_line = " · ".join(f"{m}: " + ", ".join(f"{p} {v['pearson_r']:.2f}" for p, v in rec[m].items()) for m in MOD)
    eq_rows = "\n".join(f"| {K} | " + " | ".join(f"{eq[f'K{K}_T{T}']:.0e}" for T in (2, 3, 4, 5)) + " |" for K in (2, 3, 4))
    eqh_rows = "\n".join(f"| {K} | " + " | ".join(f"{eqh[f'K{K}_T{T}']:.0e}" for T in (2, 3, 4)) + " |" for K in (2, 3))
    pw_rows = "\n".join(f"| {k} | " + " | ".join(f"±{v[n]:.2f}" for n in v) + " |" for k, v in pw["halfwidth"].items())
    pw_head = " | ".join(f"N={n}" for n in next(iter(pw["halfwidth"].values())))
    mix_rows = "\n".join(f"| {n} | {v['n~tau']:.2f} | {v['rmse_log_n']:.2f} | {v['rmse_log_tau']:.2f} | {v['r_n']:.2f} |"
                         for n, v in mx.items())
    FINDINGS_NUMBERS = dict(S=S, sw=sw, k4=k4, cv=cv, mx=mx, tc=tc, s2=s2)

    md_fig(f"""
## 2 · Findings and limitations

Reference run (`python study/run_all.py`): {S['N_per_model']} simulated subjects per model, {S['budget']} choices each,
tie threshold |ΔBIC| < {S['tie_eps']}. The figures below are embedded in the notebook, so they display without running anything.

**Sanity check.** An OPT agent with prior Beta(2, 2) (400 games) is fitted back as Beta({e0[0]:.2f}, {e0[1]:.2f}).
Zhang & Yu (2013) report a similar check, recovering Beta(1.93, 2.15).

### 2.1 Headline: which designs can separate myopic from optimal exploration?
KG-vs-OPT recovery at a fixed budget of {S['budget']} choices, for 2–4 arms and 4–15 trials per game ({des['N']} subjects per model
per cell). Use this as a design guideline: aim for cells with a high share of decisive evidence (right panel), not just high
accuracy.
""", ["fig_design"])

    md_fig(f"""
### 2.2 Horizon sweep in detail (KG vs OPT, 2 arms)
""", ["fig_horizon"])

    md(f"""
| T | games | P(KG→KG) [95% CI] | P(OPT→OPT) [95% CI] | tie | \\|ΔBIC\\|>2 | \\|ΔBIC\\|>6 | informative trials (T−3)/T | % argmax disagreement |
|---|---|---|---|---|---|---|---|---|
{sweep_rows}

**Where the models are identical.** Max difference between KG's and OPT's value gaps over all hyperstates and 200 random priors
(about 1e-16 is floating-point zero):

| arms K | T=2 | T=3 | T=4 | T=5 |
|---|---|---|---|---|
{eq_rows}

The same check with a **separate prior per arm** (equivalently, any posteriors) and this many choices left shows that KG is exact
only with ≤ 2 choices left. The shared-prior equality at T = 3 holds only because the first trial is symmetric:

| arms K | 2 left | 3 left | 4 left |
|---|---|---|---|
{eqh_rows}
""")

    md_fig(f"""
### 2.3 Model recovery at T = {S['T0']} (2 arms) and in Zhang & Yu's regime (4 arms, 15 trials, 20 games)

**2 arms, T = {S['T0']}**: left, P(best fit | generating) with a tie column; right, the inversion P(generating | best fit).

{fmt_mat(S['E2_confusion'], MOD, cols)}
""", ["fig_confusion"])

    md_fig(f"""
**4 arms, T = 15, 20 games** (N = {k4['N']} per model; KG vs OPT argmax disagreement {100 * k4['disagreement']:.1f}% of trials):

{fmt_mat(k4['confusion'], MOD, cols)}
""", ["fig_k4"])

    md_fig(f"""
### 2.4 Cross-validation versus BIC (N = {cv['N']} per model, T = {S['T0']})
Leave-one-game-out held-out NLL picks the same model as BIC for **{100 * cv['agreement']:.0f}%** of subjects; mean diagonal
accuracy BIC {cv['bic_acc']:.2f} vs CV {cv['cv_acc']:.2f}.
""", ["fig_cv"])

    md_fig(f"""
### 2.5 The prior-strength / noise trade-off, and whether mixing horizons breaks it
Error correlation corr(err n, err τ) at T = {S['T0']}: KG {tc['KG']['n~tau']:.2f}, KG-DBM {tc['KG-DBM']['n~tau']:.2f},
OPT {tc['OPT']['n~tau']:.2f}. The prior mean m trades off much less (OPT {tc['OPT']['m~tau']:.2f}).

| design (OPT, 240 choices) | corr(err n, err τ) | RMSE log n | RMSE log τ | r(true n, fitted n) |
|---|---|---|---|---|
{mix_rows}
""", ["fig_mixed", "fig_surface"])

    md_fig(f"""
### 2.6 Parameter recovery (Pearson r; log scale for n, τ)
{rec_line}
""", ["fig_param_recovery"])

    md(f"""
### 2.7 How many subjects do you need? (95% Wilson CI half-width for a recovery proportion p)

| observed p | {pw_head} |
|---|{'---|' * len(next(iter(pw['halfwidth'].values())))}
{pw_rows}

Rule of thumb at p = 0.5: about {pw['n_for']['±0.1']} subjects per model for ±0.10 and {pw['n_for']['±0.05']} for ±0.05. The reference
run uses 100 per model (60 per cell in the design heat map), so differences smaller than about 0.1–0.15 should not be interpreted.
""")

    md(TAKEAWAYS if "TAKEAWAYS" in dir() else "")

md(r"""
### References
- Acuna, D. E., & Schrater, P. (2010). Structure learning in human sequential decision-making. *PLoS Computational Biology*, 6(12), e1001003.
- Cavagnaro, D. R., Myung, J. I., Pitt, M. A., & Kujala, J. V. (2010). Adaptive design optimization: A mutual information-based approach to model discrimination in cognitive science. *Neural Computation*, 22(4), 887–905.
- Duff, M. O. (2002). *Optimal learning: Computational procedures for Bayes-adaptive Markov decision processes.* PhD thesis, UMass Amherst.
- Edwards, J., Fearnhead, P., & Glazebrook, K. (2016). On the identification and mitigation of weaknesses in the Knowledge Gradient policy for multi-armed bandits. arXiv:1607.05970.
- Frazier, P. I., Powell, W. B., & Dayanik, S. (2008). A knowledge-gradient policy for sequential information collection. *SIAM J. Control Optim.*, 47(5), 2410–2439.
- Guez, A., Silver, D., & Dayan, P. (2012). Efficient Bayes-adaptive reinforcement learning using sample-based search. *NeurIPS.*
- Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). Planning and acting in partially observable stochastic domains. *Artificial Intelligence*, 101, 99–134.
- Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *JASA*, 90(430), 773–795.
- Myung, J. I., & Pitt, M. A. (2009). Optimal experimental design for model discrimination. *Psychological Review*, 116(3), 499–518.
- Pineau, J., Gordon, G., & Thrun, S. (2003). Point-based value iteration: An anytime algorithm for POMDPs. *IJCAI.*
- Ryzhov, I. O., Powell, W. B., & Frazier, P. I. (2012). The knowledge gradient algorithm for a general class of online learning problems. *Operations Research*, 60(1), 180–195.
- Steyvers, M., Lee, M. D., & Wagenmakers, E.-J. (2009). A Bayesian analysis of human decision-making on bandit problems. *J. Mathematical Psychology*, 53(3), 168–179.
- Wilson, R. C., & Collins, A. G. E. (2019). Ten simple rules for the computational modeling of behavioral data. *eLife*, 8, e49547.
- Wilson, R. C., Geana, A., White, J. M., Ludvig, E. A., & Cohen, J. D. (2014). Humans use directed and random exploration to solve the explore–exploit dilemma. *J. Experimental Psychology: General*, 143(6), 2074–2081.
- Yu, A. J., & Cohen, J. D. (2009). Sequential effects: Superstition or rational behavior? *NeurIPS.*
- Zhang, S., & Yu, A. J. (2013). Forgetful Bayes and myopic planning: Human learning and decision-making in a bandit setting. *NeurIPS.*
- *Future work:* Gmytrasiewicz, P. J., & Doshi, P. (2005). A framework for sequential planning in multi-agent settings. *JAIR*, 24, 49–79 (I-POMDPs, not yet implemented).
""")

def _write(cell_list, name):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cell_list
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                      "language_info": {"name": "python"}, "colab": {"provenance": [], "toc_visible": True}}
    out = ROOT / "notebook" / name
    nbf.write(nb, out)
    print("wrote", out, len(cell_list), "cells")


_write(TUT, "tinybamdp_tutorial.ipynb")
_write(STU, "kg_vs_optimal_identifiability.ipynb")
