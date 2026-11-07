"""Run every experiment, save numeric results (JSON + pickles) and figures.

    python study/run_all.py [--quick] [--jobs 2] [--only E0,E2,...]

E0  sanity: Beta(2,2) prior recovery for the Bayes-optimal agent
E1  parameter recovery (all models, K=2, T=10)
E2  model recovery, tie-aware, with 95% CIs, evidence strength, inversion matrix
E3  horizon sweep KG vs OPT at a fixed 240-choice budget (research question)
E4  prior-strength / softmax trade-off (error correlations, likelihood surface)
E5  replication point in Zhang & Yu's regime: K=4 arms, T=15, 20 games
E6  leave-one-game-out cross-validation vs BIC
E7  mixed-horizon design: does varying T within a subject break the n-tau ridge?
E8  exhaustive KG/OPT equivalence check: shared priors (K in {2,3,4}) and per-arm priors
E9  design guideline: KG-vs-OPT recovery and decisive-evidence share over arms K x horizon T
E10 power table: CI half-width vs number of subjects
"""
import argparse
import json
import pathlib
import pickle
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinybamdp import bandit_dp, fitting, get_model, sample_task, simulate, simulate_design  # noqa: E402
from study import analysis, plots  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true")
p.add_argument("--jobs", type=int, default=2)
p.add_argument("--only", default="E0,E1,E2,E3,E4,E5,E6,E7,E8,E9,E10")
args = p.parse_args()
ONLY = set(args.only.split(","))

Q = args.quick
N = 20 if Q else 100              # subjects per generating model (baseline, sweep)
N_K4 = 8 if Q else 100            # K=4 replication
N_CV = 6 if Q else 100            # cross-validation
N_DES = 10 if Q else 60           # design heat map, per model per cell
N_MIX = 15 if Q else 60           # mixed-horizon design
BUDGET, T0 = 240, 10
HORIZONS = [2, 3, 5, 8, 12, 20, 30]
MODELS = ["RW", "WSLS", "KG", "KG-DBM", "OPT"]
TIE_EPS = 0.1                     # |dBIC| below this = practically identical fits
LOG = ("n", "tau")
OUT, FIG = ROOT / "results", ROOT / "figures"
OUT.mkdir(exist_ok=True), FIG.mkdir(exist_ok=True)
plots.style()
summ_path = OUT / "summary.json"
summary = json.loads(summ_path.read_text()) if (summ_path.exists() and ONLY != set(p.get_default("only").split(","))) else {}
summary.update({"version": "0.2.0", "N_per_model": N, "budget": BUDGET, "T0": T0, "tie_eps": TIE_EPS,
                "models": MODELS})
t_start = time.time()


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png"), fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def dump(obj, name):
    with open(OUT / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def log(*a):
    print(f"[{time.time() - t_start:6.0f}s]", *a, flush=True)


# ---------------------------------------------------------------- E0 sanity
if "E0" in ONLY:
    rng = np.random.default_rng(7)
    d = simulate(get_model("OPT"), {"m": 0.5, "n": 4.0, "tau": 10.0}, sample_task(400, 2, rng), T0, rng)
    f = fitting.fit("OPT", d, 6, rng)
    a, b = f.params["m"] * f.params["n"], (1 - f.params["m"]) * f.params["n"]
    summary["E0_prior_recovery"] = {"true": [2.0, 2.0], "recovered": [a, b], "tau": f.params["tau"]}
    log(f"E0 Beta(2,2) -> Beta({a:.2f}, {b:.2f})")

# ------------------------------------------------- E1 + E2 baseline (5 x 5)
if ONLY & {"E1", "E2", "E4"}:
    res = fitting.run_recovery(MODELS, MODELS, N, BUDGET // T0, T0, n_starts=4, seed=1, n_jobs=args.jobs)
    dump(res, "baseline_recovery")
    M = fitting.confusion_matrix(res, MODELS, tie_eps=TIE_EPS)
    S = fitting.recovery_summary(res, MODELS, tie_eps=TIE_EPS)
    INV = fitting.inversion_matrix(M)
    summary["E2_confusion"] = M.tolist()
    summary["E2_inversion"] = np.nan_to_num(INV, nan=-1).tolist()
    summary["E2_summary"] = S
    summary["E2_fit_failure_rate"] = fitting.failure_rate(res)
    log("E2 confusion (last col = tie)\n", np.round(M, 2))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    plots.confusion(axes[0], M, MODELS, title=f"P(best fit | generating), T={T0}")
    plots.confusion(axes[1], INV, MODELS, title="P(generating | best fit)", xlabel="best-fitting model (BIC)",
                    ylabel="generating model")
    save(fig, "fig_confusion")

    rstats = {m: fitting.recovery_stats(res, m) for m in MODELS}
    summary["E1_param_recovery"] = rstats
    fig, axes = plt.subplots(4, 4, figsize=(10, 9.5))
    axes = axes.ravel()
    k = 0
    for mname in MODELS:
        for pname, (tr, rc) in fitting.param_table(res, mname).items():
            plots.recovery_scatter(axes[k], tr, rc, f"{mname}: {pname}", plots.COLORS[mname], log=pname in LOG)
            k += 1
    for ax in axes[k:]:
        ax.axis("off")
    save(fig, "fig_param_recovery")
    log("E1 r:", {f"{m}.{q}": round(v["pearson_r"], 2) for m, d_ in rstats.items() for q, v in d_.items()})

    tradeoff = {}
    for mname in ["KG", "KG-DBM", "OPT"]:
        tab = fitting.param_table(res, mname)
        err = {q: np.log(v[1]) - np.log(v[0]) if q in LOG else v[1] - v[0] for q, v in tab.items()}
        tradeoff[mname] = {"n~tau": float(np.corrcoef(err["n"], err["tau"])[0, 1]),
                           "m~tau": float(np.corrcoef(err["m"], err["tau"])[0, 1]),
                           "m~n": float(np.corrcoef(err["m"], err["n"])[0, 1])}
    summary["E4_error_correlations"] = tradeoff
    log("E4 error correlations", tradeoff)

# ----------------------------------------------- E3 horizon sweep (KG vs OPT)
if "E3" in ONLY:
    sweep, dis = {}, {}
    for T in HORIZONS:
        r = fitting.run_recovery(["KG", "OPT"], ["KG", "OPT"], N, BUDGET // T, T, n_starts=3,
                                 seed=100 + T, n_jobs=args.jobs)
        dump(r, f"sweep_T{T}")
        S = fitting.recovery_summary(r, ["KG", "OPT"], tie_eps=TIE_EPS)
        win = [fitting.classify(fs, ["KG", "OPT"], tie_eps=TIE_EPS)[0] for _, _, fs in r]
        correct = np.array([w == g for w, (g, _, _) in zip(win, r)])
        taus = np.array([tr["tau"] for _, tr, _ in r])
        hi = taus > np.median(taus)
        k = int(correct.sum())
        lo_, hi_ = fitting.wilson_ci(k, len(correct))
        sweep[T] = {"per_model": S, "acc": float(correct.mean()), "acc_lo": lo_, "acc_hi": hi_,
                    "tie": float(np.mean([w == fitting.TIE for w in win])),
                    "decisive>2": float(np.mean([S[g]["decisive>2"] for g in S])),
                    "decisive>6": float(np.mean([S[g]["decisive>6"] for g in S])),
                    "acc_high_tau": float(correct[hi].mean()), "acc_low_tau": float(correct[~hi].mean())}
        dis[T] = float(analysis.policy_disagreement(T).mean())
        log(f"E3 T={T:2d} acc={sweep[T]['acc']:.2f} [{lo_:.2f},{hi_:.2f}] tie={sweep[T]['tie']:.2f} "
            f"dec>2={sweep[T]['decisive>2']:.2f} dec>6={sweep[T]['decisive>6']:.2f} disagree={dis[T]:.3f}")
    summary["E3_horizon_sweep"] = {str(k): v for k, v in sweep.items()}
    summary["E3_policy_disagreement"] = {str(k): v for k, v in dis.items()}

    H = np.array(HORIZONS)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    ax = axes[0]
    for g, mk in [("KG", "o"), ("OPT", "s")]:
        acc = np.array([sweep[T]["per_model"][g]["acc"] for T in H])
        lo_ = np.array([sweep[T]["per_model"][g]["acc_lo"] for T in H])
        hi_ = np.array([sweep[T]["per_model"][g]["acc_hi"] for T in H])
        ax.errorbar(H, acc, yerr=[np.maximum(acc - lo_, 0), hi_ - acc], fmt=mk + "-", color=plots.COLORS[g], capsize=3,
                    label=f"{g} recovered as {g}")
    ax.plot(H, [sweep[T]["tie"] for T in H], "^:", color=plots.MUTED, label="tie (|ΔBIC| < 0.1)")
    ax.set_ylim(0, 1.02), ax.set_ylabel("proportion of subjects"), ax.set_title(f"Recovery ({BUDGET} choices, 95% CI)")
    ax.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 0.12))
    ax = axes[1]
    ax.plot(H, [sweep[T]["decisive>2"] for T in H], "o-", color=plots.INK, label="|ΔBIC| > 2 (positive)")
    ax.plot(H, [sweep[T]["decisive>6"] for T in H], "s--", color=plots.MUTED, label="|ΔBIC| > 6 (strong)")
    ax.set_ylim(0, 1.02), ax.set_ylabel("share of subjects"), ax.set_title("Strength of evidence"), ax.legend(fontsize=7)
    ax = axes[2]
    ax.plot(H, [100 * dis[T] for T in H], "o-", color=plots.INK)
    ax.set_ylabel("% trials, argmax differs"), ax.set_title("KG vs OPT disagreement")
    for ax in axes:
        ax.set_xscale("log"), ax.set_xticks(H, H), ax.minorticks_off(), ax.set_xlabel("horizon T")
    save(fig, "fig_horizon")

# ------------------------------------------------- E4 likelihood surface
if "E4" in ONLY:
    rng = np.random.default_rng(11)
    true = {"m": 0.5, "n": 4.0, "tau": 5.0}
    d = simulate(get_model("OPT"), true, sample_task(BUDGET // T0, 2, rng), T0, rng)
    xs, ys = np.geomspace(0.5, 20, 40), np.geomspace(0.5, 50, 40)
    Z = analysis.likelihood_surface("OPT", d, {"m": 0.5}, "n", xs, "tau", ys)
    i, j = np.unravel_index(Z.argmax(), Z.shape)
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    cs = plots.surface(ax, xs, ys, Z, "prior strength n = a + b", "softmax precision τ",
                       truth=(true["n"], true["tau"]), mle=(xs[j], ys[i]))
    fig.colorbar(cs, ax=ax, label="log-lik − max")
    ax.set_title("OPT likelihood (m fixed)")
    save(fig, "fig_surface")
    inside = (Z - Z.max()) > -1.92
    summary["E4_surface"] = {"n_range_95": [float(xs[inside.any(0)].min()), float(xs[inside.any(0)].max())],
                             "tau_range_95": [float(ys[inside.any(1)].min()), float(ys[inside.any(1)].max())]}

# ----------------------------------------- E5 Zhang & Yu regime: K=4, T=15
if "E5" in ONLY:
    G5, T5, K5 = 20, 15, 4
    r5 = fitting.run_recovery(MODELS, MODELS, N_K4, G5, T5, K=K5, n_starts=3, seed=500, n_jobs=args.jobs)
    dump(r5, "k4_recovery")
    M5 = fitting.confusion_matrix(r5, MODELS, tie_eps=TIE_EPS)
    summary["E5_k4"] = {"K": K5, "T": T5, "G": G5, "N": N_K4, "confusion": M5.tolist(),
                        "summary": fitting.recovery_summary(r5, MODELS, tie_eps=TIE_EPS),
                        "disagreement": float(analysis.policy_disagreement(T5, K=K5, G=1000).mean()),
                        "fit_failure_rate": fitting.failure_rate(r5)}
    log("E5 K=4 confusion\n", np.round(M5, 2))
    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    plots.confusion(ax, M5, MODELS, title=f"K={K5} arms, T={T5}, {G5} games (N={N_K4})")
    save(fig, "fig_k4")

# ---------------------------------------- E6 cross-validation versus BIC
if "E6" in ONLY:
    r6 = fitting.run_recovery(MODELS, MODELS, N_CV, BUDGET // T0, T0, n_starts=3, seed=600,
                              n_jobs=args.jobs, cv=True)
    dump(r6, "cv_recovery")
    Mb = fitting.confusion_matrix(r6, MODELS, "bic", tie_eps=TIE_EPS)
    Mc = fitting.confusion_matrix(r6, MODELS, "cv_nll", tie_eps=TIE_EPS / 2)
    agree = np.mean([fitting.classify(f, MODELS, "bic", TIE_EPS)[0] == fitting.classify(f, MODELS, "cv_nll", TIE_EPS / 2)[0]
                     for _, _, f in r6])
    summary["E6_cv"] = {"N": N_CV, "bic_confusion": Mb.tolist(), "cv_confusion": Mc.tolist(),
                        "agreement": float(agree),
                        "bic_acc": float(np.mean(np.diag(Mb[:, :len(MODELS)]))),
                        "cv_acc": float(np.mean(np.diag(Mc[:, :len(MODELS)])))}
    log(f"E6 BIC acc {summary['E6_cv']['bic_acc']:.2f}  CV acc {summary['E6_cv']['cv_acc']:.2f}  agreement {agree:.2f}")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    plots.confusion(axes[0], Mb, MODELS, title=f"BIC (N={N_CV}/model)")
    plots.confusion(axes[1], Mc, MODELS, title="Leave-one-game-out CV", xlabel="best model (held-out NLL)")
    save(fig, "fig_cv")

# ------------------------------- E7 mixed-horizon design vs. fixed horizons
if "E7" in ONLY:
    designs = {"fixed T=5": [(48, 5)], "fixed T=20": [(12, 20)], "mixed 5 + 20": [(24, 5), (6, 20)]}
    opt = get_model("OPT")
    E7 = {}
    rng = np.random.default_rng(700)
    for name, des in designs.items():
        trues, recs = [], []
        for _ in range(N_MIX):
            tr = opt.sample_params(rng)
            data = simulate_design(opt, tr, des, 2, rng)
            f = fitting.fit("OPT", data, 4, rng)
            trues.append(tr), recs.append(f.params)
        err = {q: np.array([np.log(r[q]) - np.log(t[q]) if q in LOG else r[q] - t[q] for t, r in zip(trues, recs)])
               for q in ("m", "n", "tau")}
        E7[name] = {"n~tau": float(np.corrcoef(err["n"], err["tau"])[0, 1]),
                    "rmse_log_n": float(np.sqrt(np.mean(err["n"] ** 2))),
                    "rmse_log_tau": float(np.sqrt(np.mean(err["tau"] ** 2))),
                    "r_n": float(np.corrcoef(np.log([t["n"] for t in trues]), np.log([r["n"] for r in recs]))[0, 1])}
        log(f"E7 {name}: {E7[name]}")
    # likelihood surfaces for one subject per design (same true parameters)
    true = {"m": 0.5, "n": 4.0, "tau": 5.0}
    xs, ys = np.geomspace(0.5, 20, 32), np.geomspace(0.5, 50, 32)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0), gridspec_kw={"width_ratios": [1, 1, 1, 0.8]})
    for ax, (name, des) in zip(axes, designs.items()):
        data = simulate_design(opt, true, des, 2, np.random.default_rng(11))
        Z = analysis.likelihood_surface("OPT", data, {"m": 0.5}, "n", xs, "tau", ys)
        i, j = np.unravel_index(Z.argmax(), Z.shape)
        plots.surface(ax, xs, ys, Z, "prior strength n", "softmax precision τ", truth=(4, 5), mle=(xs[j], ys[i]))
        inside = (Z - Z.max()) > -1.92
        E7[name]["area95_cells"] = int(inside.sum())
        ax.set_title(name)
    ax = axes[3]
    names = list(designs)
    ax.barh(names, [E7[n]["n~tau"] for n in names], color=plots.COLORS["OPT"], height=0.5)
    ax.set_xlim(0, 1), ax.set_xlabel("corr(err n, err τ)"), ax.set_title(f"Trade-off (N={N_MIX})"), ax.invert_yaxis()
    save(fig, "fig_mixed")
    summary["E7_mixed"] = E7

# --------------------------------------------- E8 equivalence check
if "E8" in ONLY:
    tab = analysis.equivalence_table((2, 3, 4), (2, 3, 4, 5), n_priors=40 if Q else 200)
    summary["E8_equivalence"] = {f"K{K}_T{T}": v for (K, T), v in tab.items()}
    het = analysis.equivalence_table_hetero((2, 3), (2, 3, 4), n_priors=40 if Q else 200)
    summary["E8_equivalence_hetero"] = {f"K{K}_T{T}": v for (K, T), v in het.items()}
    rng = np.random.default_rng(8)
    pri = np.exp(rng.uniform(np.log(0.1), np.log(10), size=(200, 2, 2)))
    summary["E8_share_unequal_T3_hetero_K2"] = float(np.mean([analysis.kg_opt_gap_difference_hetero(a, b, 3) > 1e-9
                                                               for a, b in pri]))
    # Only trials with >= 3 choices left in an asymmetric state can separate KG from OPT:
    # trial 1 is symmetric and the last two trials are KG-exact, so T - 3 of T trials carry information.
    summary["E8_informative_fraction"] = {str(T): max(0, T - 3) / T for T in HORIZONS}
    log("E8 shared", {k: f"{v:.1e}" for k, v in summary["E8_equivalence"].items()})
    log("E8 per-arm", {k: f"{v:.1e}" for k, v in summary["E8_equivalence_hetero"].items()},
        "share unequal at T=3:", summary["E8_share_unequal_T3_hetero_K2"])

# ------------------------------------------- E9 design guideline (K x T)
if "E9" in ONLY:
    Ks, Ts = [2, 3, 4], [4, 6, 8, 12, 15]
    acc = np.full((len(Ks), len(Ts)), np.nan)
    dec = np.full_like(acc, np.nan)
    cells = {}
    for i, K in enumerate(Ks):
        for j, T in enumerate(Ts):
            r = fitting.run_recovery(["KG", "OPT"], ["KG", "OPT"], N_DES, BUDGET // T, T, K=K, n_starts=3,
                                     seed=900 + 10 * K + T, n_jobs=args.jobs)
            S = fitting.recovery_summary(r, ["KG", "OPT"], tie_eps=TIE_EPS)
            acc[i, j] = np.mean([S[g]["acc"] for g in S])
            dec[i, j] = np.mean([S[g]["decisive>2"] for g in S])
            cells[f"K{K}_T{T}"] = {"acc": acc[i, j], "decisive>2": dec[i, j],
                                   "decisive>6": float(np.mean([S[g]["decisive>6"] for g in S]))}
            log(f"E9 K={K} T={T:2d} acc={acc[i, j]:.2f} dec>2={dec[i, j]:.2f}")
    summary["E9_design"] = {"Ks": Ks, "Ts": Ts, "N": N_DES, "cells": cells}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 2.9))
    for ax, M, title in [(axes[0], acc, "KG vs OPT recovery accuracy"), (axes[1], dec, "share with |ΔBIC| > 2")]:
        im = ax.imshow(M, cmap=plots.SEQ, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(Ts)), Ts), ax.set_yticks(range(len(Ks)), Ks)
        ax.set_xlabel(f"trials per game T  (games = {BUDGET}/T)"), ax.set_ylabel("arms K"), ax.grid(False)
        for a_ in range(len(Ks)):
            for b_ in range(len(Ts)):
                ax.text(b_, a_, f"{M[a_, b_]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M[a_, b_] > 0.55 else plots.INK)
        ax.set_title(title)
    fig.colorbar(im, ax=axes, fraction=0.025)
    fig.savefig(FIG / "fig_design.png", bbox_inches="tight"), fig.savefig(FIG / "fig_design.pdf", bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------ E10 power table
if "E10" in ONLY:
    Ns = [20, 40, 60, 100, 200, 400]
    summary["E10_power"] = {"halfwidth": {f"p={p_}": {str(n): fitting.ci_halfwidth(p_, n) for n in Ns} for p_ in (0.5, 0.7, 0.9)},
                            "n_for": {f"±{h}": fitting.n_for_halfwidth(h) for h in (0.15, 0.10, 0.05)}}
    log("E10", summary["E10_power"]["n_for"])

summary.setdefault("runtime_by_stage", {})[args.only] = time.time() - t_start
summary["runtime_s"] = time.time() - t_start
summary["graph_states"] = {k: {"canonical": g.n_states, "full": g.n_full_states}
                           for k, g in {"K2_T30": bandit_dp.bandit_graph(2, 30), "K4_T15": bandit_dp.bandit_graph(4, 15)}.items()}
summ_path.write_text(json.dumps(summary, indent=2, default=float))
log("done")
