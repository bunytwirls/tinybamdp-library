# tinybamdp

**Exact, tabular Bayes-adaptive (BAMDP) and POMDP models of decision-making under uncertainty, written from scratch, with tools to simulate, fit and validate them as cognitive models.**

[![Tutorial: Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bunytwirls/tinybamdp-library/blob/main/notebook/tinybamdp_tutorial.ipynb)
&nbsp;tutorial &nbsp;&nbsp;
[![Study: Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bunytwirls/tinybamdp-library/blob/main/notebook/kg_vs_optimal_identifiability.ipynb)
&nbsp;identifiability study

Author: Maimuna Chowdhury · License: MIT

![Design guideline: KG vs Bayes-optimal recovery over arms and horizon](figures/fig_design.png)

## Research question

Zhang & Yu (2013) reported that human bandit choices are better described by a *myopic* Knowledge-Gradient (KG) policy than by the
Bayes-optimal policy (OPT). **Which task designs can tell these two models apart, and which of their parameters can be identified?**

## Findings (v0.3; simulated data, 240 choices per subject, 100 subjects per model unless stated)

| Result | Evidence |
|---|---|
| **Design guideline.** Separation needs games of about 12 or more trials and at least three arms. Accuracy rises from 0.32 (2 arms × 4 trials) to 0.93 (4 arms × 15 trials), and the share with positive evidence (\|ΔBIC\| > 2) from 0.02 to 0.88. Going from three to four arms adds little; the horizon matters most. | `figures/fig_design.png` (60 subjects per model per cell) |
| **Why short games fail.** With ≤ 2 choices left, KG is *exactly* optimal for any posteriors, and on the first trial both policies are indifferent. Only trials 2 … T−2 can separate them: none for T ≤ 3, 40% at T = 5, 85% at T = 20. With unequal posteriors and 3 choices left, they differ. | `results/summary.json` (E8), unit tests |
| Raw accuracy overstates mid-horizon evidence: 2 arms, T = 5 gives accuracy 0.55 but only 16% positive evidence. | `figures/fig_horizon.png` |
| Zhang & Yu's regime (4 arms, T = 15, 20 games): KG 0.96 [0.90–0.98], OPT 0.68 [0.58–0.76]. OPT's errors go mostly to RW (28%). | `figures/fig_k4.png` |
| On 2-arm, T = 10 games, a delta-rule (RW) win is weak evidence: RW generated the data only 42% of the times it wins. | `figures/fig_confusion.png` |
| Prior strength *n* and softmax precision τ are confounded (error correlation 0.92–0.94). Mixing horizons does not fix it. | `figures/fig_mixed.png` |
| The forgetful KG-DBM is hard to identify on stationary bandits (recovered 34%). CV and BIC disagree for 38% of subjects. | `figures/fig_cv.png` |

Figures use the IBM colour-blind-safe palette for models (gold darkened for contrast) and ColorBrewer RdPu for maps;
the model colours pass adjacent-pair colour-blind separation checks, and every colour is labelled.

Model recovery is tie-aware (|ΔBIC| < 0.1 = tie), with 95% Wilson CIs, the share of decisive classifications, inversion
matrices and a power table (about 100 subjects per model give ±0.10).

## Layout

```
tinybamdp/   core library, numpy + scipy only
  beliefs     Beta, Dirichlet and discrete (Bayes-filter) beliefs
  bandit_dp   exact BAMDP solver for K-armed Bernoulli bandits, with arm-symmetry reduction
  bamdp       generic tabular BAMDP with unknown transitions (Dirichlet), tying, CE baseline
  pomdp       discrete POMDPs: belief filter, point-based value iteration, tiger problem
  models      RW, WSLS, KG, KG-DBM, Bayes-optimal; shared init/probs/update interface; register_model()
  task        multi-game bandit task (any K, environment prior, mixed-horizon designs)
  fitting     MLE with restarts, LOGO cross-validation, tie-aware recovery, CIs, power helpers
study/       the identifiability study (needs matplotlib)
  analysis    policy disagreement, KG/OPT equivalence checks, likelihood surfaces, signatures
  plots       figures
  run_all.py  experiments E0–E10
tests/       test_core.py, test_study.py (52 tests)
notebook/    tinybamdp_tutorial.ipynb (how to use the library) and kg_vs_optimal_identifiability.ipynb
             (the study, reference figures embedded); both load the code from this repository, and
             build_notebook.py regenerates them
```

## Performance (2 CPU cores)

| | v0.2 | v0.3 |
|---|---|---|
| Exact solve, 4 arms × 15 trials | 90 ms (490,314 states) | 1.4 ms (25,261 canonical states) |
| OPT fit, 4 arms × 15 trials × 20 games | ≈ 11 s | ≈ 0.5 s |
| 4-arm recovery study | 28 min for N = 40 | 6 min for N = 100 |
| Full study | ≈ 51 min | ≈ 35 min, with 2.5–5× more subjects in E5/E6 and the new design heat map |

The speed-ups are exact. Arm symmetry divides the state graph by up to K!. KG and OPT decision values are computed once per
(m, n) and reused for every τ. OPT recovery fits are warm-started from the KG fit. Tests check that the fast paths match the
original computations.

## Quick start

```bash
git clone https://github.com/bunytwirls/tinybamdp-library.git
cd tinybamdp-library
pip install -e .[notebook,test]
pytest -q
python study/run_all.py --jobs 2              # full study (--quick for a smoke test)
python study/run_all.py --only E9             # one experiment; results merge into results/summary.json
```

To use only the library in another project: `pip install git+https://github.com/bunytwirls/tinybamdp-library.git`
(this installs `tinybamdp`; the `study` code and notebooks stay in the repository).

In Colab, the notebooks' first cell clones this repository automatically.

```python
import numpy as np
from tinybamdp import get_model, simulate, sample_task, fitting
rng = np.random.default_rng(0)
subject = simulate(get_model("OPT"), {"m": .5, "n": 4, "tau": 10}, sample_task(20, 4, rng), 15, rng)
fits = {m: fitting.fit(m, subject, n_starts=3, rng=rng) for m in ("RW", "KG", "OPT")}
print(fitting.classify(fits, list(fits), tie_eps=0.1))       # (winner or "tie", ΔBIC margin)
```

## Limitations and next steps

The data are simulated only. Simulation parameter ranges are set by hand, fits are per subject, and the tie threshold is a
practical choice. Not yet implemented: OPT with a forgetful learner, forced-choice designs to break the n–τ trade-off, and
I-POMDPs. Next steps: fit real bandit data (Steyvers et al., 2009) and use adaptive design optimisation.

## References

Bellman (1957) · Duff (2002) · Kaelbling, Littman & Cassandra (1998) · Pineau, Gordon & Thrun (2003) · Frazier, Powell & Dayanik (2008) ·
Ryzhov, Powell & Frazier (2012) · Yu & Cohen (2009) · Zhang & Yu (2013) · Steyvers, Lee & Wagenmakers (2009) · Wilson et al. (2014) ·
Acuna & Schrater (2010) · Wilson & Collins (2019) · Myung & Pitt (2009) · Cavagnaro et al. (2010) · Kass & Raftery (1995) ·
Edwards, Fearnhead & Glazebrook (2016). Full list in `kg_vs_optimal_identifiability.ipynb`, § 2.
