"""Maximum-likelihood fitting, cross-validation, parameter and model recovery.

Follows the validation recipe of Wilson & Collins (2019): simulate, fit with
multiple random restarts, check parameter recovery, then model recovery.

Model recovery is *tie-aware*: when two models give the same fit (|delta| below
``tie_eps``) the subject is counted as "tie" rather than handed to whichever
model ``argmin`` happens to list first.  Evidence strength (the margin between
the best and second-best model) is reported alongside accuracy.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .models import get_model
from .task import BanditData, log_likelihood, n_trials, sample_task, simulate

TIE = "tie"

#: Recovery fits model -> earlier-fitted model whose MLE is used as an extra start.
#: KG and OPT share (m, n, tau) and make near-identical choices, so the KG fit is a good start for OPT.
WARM_START = {"OPT": "KG"}

#: Start method for worker processes in ``run_recovery(n_jobs > 1)``.  None = platform default
#: (fork on Linux, fastest for scripts).  Notebooks should use "spawn": forking a Jupyter kernel
#: that runs helper threads can deadlock.
MP_CONTEXT = None


@dataclass
class FitResult:
    model: str
    params: Dict[str, float]
    nll: float
    bic: float
    aic: float
    n_failed: int = 0           # restarts whose optimiser did not report success
    converged: bool = True      # did the best restart report success?
    cv_nll: Optional[float] = None   # held-out NLL (leave-one-game-out), if computed


def fit(model_name: str, data, n_starts: int = 5, rng: Optional[np.random.Generator] = None,
        maxiter: int = 200, init: Optional[Dict[str, float]] = None) -> FitResult:
    """MLE with ``n_starts`` random restarts (plus an optional warm start ``init``).

    ``data`` is a BanditData or a list of them (mixed-horizon designs).
    """
    rng = rng or np.random.default_rng()
    model = get_model(model_name)
    bounds = model.z_bounds()

    def nll(z):
        return -log_likelihood(model, model.to_params(z), data)

    starts = [np.array([rng.uniform(lo, hi) for lo, hi in bounds]) for _ in range(n_starts)]
    if init is not None:
        lo, hi = np.array(bounds).T
        starts.insert(0, np.clip(model.to_z(init), lo, hi))
    best, failed = None, 0
    for z0 in starts:
        res = minimize(nll, z0, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
        failed += int(not res.success)
        if best is None or res.fun < best.fun:
            best = res
    k, N = model.n_params, n_trials(data)
    return FitResult(model_name, model.to_params(best.x), float(best.fun),
                     float(2 * best.fun + k * np.log(N)), float(2 * best.fun + 2 * k),
                     failed, bool(best.success))


def cross_validate(model_name: str, data: BanditData, full_fit: Optional[FitResult] = None,
                   n_starts: int = 0, rng: Optional[np.random.Generator] = None) -> float:
    """Leave-one-game-out held-out NLL (Zhang & Yu, 2013).

    Each fold is warm-started from the full-data MLE (plus ``n_starts`` random
    restarts), which keeps the cost close to G single-start fits.
    """
    rng = rng or np.random.default_rng()
    full_fit = full_fit or fit(model_name, data, 3, rng)
    model = get_model(model_name)
    total = 0.0
    games = np.arange(data.G)
    for g in games:
        train = data.subset(games != g)
        f = fit(model_name, train, n_starts, rng, init=full_fit.params)
        total -= log_likelihood(model, f.params, data.subset(games == g))
    return float(total)


# ------------------------------------------------------------ classification
def classify(fits: Dict[str, FitResult], models: Sequence[str], criterion: str = "bic",
             tie_eps: float = 1e-6) -> Tuple[str, float]:
    """Return (winner or "tie", margin between best and second-best score)."""
    scores = np.array([getattr(fits[m], criterion) for m in models], float)
    order = np.argsort(scores)
    margin = float(scores[order[1]] - scores[order[0]]) if len(models) > 1 else np.inf
    return (TIE if margin < tie_eps else models[order[0]]), margin


def wilson_ci(k: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion k / n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(c - h), float(c + h))


def ci_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    """Half-width of the 95% Wilson interval for an observed proportion p with n subjects."""
    lo, hi = wilson_ci(p * n, n, z)
    return (hi - lo) / 2


def n_for_halfwidth(h: float, p: float = 0.5, z: float = 1.96) -> int:
    """Subjects per model needed for a CI half-width of about h (normal approximation)."""
    return int(np.ceil(z * z * p * (1 - p) / (h * h)))


def confusion_matrix(results, models: Sequence[str], criterion: str = "bic",
                     tie_eps: Optional[float] = None) -> np.ndarray:
    """P(best-fitting model | generating model); rows = generating.

    With ``tie_eps`` set, an extra last column holds P(tie).
    """
    idx = {m: i for i, m in enumerate(models)}
    M = np.zeros((len(models), len(models) + (tie_eps is not None)))
    for gen, _, fits in results:
        if tie_eps is None:
            scores = [getattr(fits[m], criterion) for m in models]
            M[idx[gen], int(np.argmin(scores))] += 1
        else:
            w, _ = classify(fits, models, criterion, tie_eps)
            M[idx[gen], len(models) if w == TIE else idx[w]] += 1
    return M / M.sum(axis=1, keepdims=True)


def inversion_matrix(M: np.ndarray) -> np.ndarray:
    """P(generating model | best-fitting model), assuming equal numbers per generating model.

    Uses the model columns of a confusion matrix (a trailing tie column is ignored).
    """
    n = M.shape[0]
    C = M[:, :n]
    col = C.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(col > 0, C / col, np.nan)


def recovery_summary(results, models: Sequence[str], criterion: str = "bic", tie_eps: float = 1e-6,
                     thresholds: Sequence[float] = (2.0, 6.0)) -> Dict:
    """Accuracy with 95% CIs, tie rate and evidence strength per generating model.

    ``decisive>X`` is the share of subjects whose best model beats the
    runner-up by more than X criterion units (BIC: 2 = positive, 6 = strong
    evidence, Kass & Raftery 1995); ``acc|decisive>X`` is accuracy among them.
    """
    out = {}
    for gen in models:
        rows = [classify(f, models, criterion, tie_eps) for g, _, f in results if g == gen]
        if not rows:
            continue
        n = len(rows)
        win = np.array([w for w, _ in rows])
        margin = np.array([m for _, m in rows])
        k = int((win == gen).sum())
        lo, hi = wilson_ci(k, n)
        d = {"n": n, "acc": k / n, "acc_lo": lo, "acc_hi": hi, "tie": float((win == TIE).mean()),
             "median_margin": float(np.median(margin))}
        for t in thresholds:
            dec = margin > t
            d[f"decisive>{t:g}"] = float(dec.mean())
            d[f"acc|decisive>{t:g}"] = float((win[dec] == gen).mean()) if dec.any() else np.nan
        out[gen] = d
    return out


# ---------------------------------------------------------------- one unit
def _simulate_and_fit(job):
    rng = np.random.default_rng(job["seed"])
    m = get_model(job["gen"])
    true = m.sample_params(rng)
    data = simulate(m, true, sample_task(job["G"], job["K"], rng, *job["env_prior"]), job["T"], rng)
    fits = {}
    for f in job["fit_models"]:
        src = WARM_START.get(f)
        if src in fits:      # nearly identical model already fitted: start there, one random restart fewer
            fr = fit(f, data, max(1, job["n_starts"] - 1), rng, init=fits[src].params)
        else:
            fr = fit(f, data, job["n_starts"], rng)
        if job["cv"]:
            fr.cv_nll = cross_validate(f, data, fr, 0, rng)
        fits[f] = fr
    return job["gen"], true, fits


def run_recovery(gen_models: Sequence[str], fit_models: Sequence[str], n_subjects: int,
                 G: int, T: int, K: int = 2, n_starts: int = 4, seed: int = 0,
                 n_jobs: int = 1, progress: Optional[Callable[[int, int], None]] = None,
                 env_prior: Tuple[float, float] = (2.0, 2.0), cv: bool = False) -> List:
    """Simulate ``n_subjects`` per generating model and fit every fit model.

    Returns a list of (generating model, true params, {fit model: FitResult}).
    ``cv=True`` also stores leave-one-game-out NLL in ``FitResult.cv_nll``.
    ``progress(done, total)`` is called after each simulated subject.
    """
    ss = np.random.SeedSequence(seed)
    seeds = ss.generate_state(len(gen_models) * n_subjects)
    jobs = [dict(gen=str(g), fit_models=tuple(fit_models), G=G, T=T, K=K, n_starts=n_starts,
                 seed=int(s), env_prior=tuple(env_prior), cv=cv)
            for g, s in zip(np.repeat(gen_models, n_subjects), seeds)]
    out = []
    if n_jobs == 1:
        for j in jobs:
            out.append(_simulate_and_fit(j))
            if progress:
                progress(len(out), len(jobs))
        return out
    import multiprocessing as mp
    ctx = mp.get_context(MP_CONTEXT) if MP_CONTEXT else None
    with ProcessPoolExecutor(n_jobs, mp_context=ctx) as ex:
        for r in ex.map(_simulate_and_fit, jobs, chunksize=2):
            out.append(r)
            if progress:
                progress(len(out), len(jobs))
    return out


def failure_rate(results) -> float:
    """Share of fits whose best restart did not report optimiser success."""
    flags = [not f.converged for _, _, fs in results for f in fs.values()]
    return float(np.mean(flags)) if flags else np.nan


def param_table(results, model: str):
    """Arrays of (true, recovered) per parameter for self-fits of ``model``."""
    out = {}
    for gen, true, fits in results:
        if gen != model:
            continue
        for k, v in true.items():
            out.setdefault(k, [[], []])
            out[k][0].append(v)
            out[k][1].append(fits[model].params[k])
    return {k: (np.array(a), np.array(b)) for k, (a, b) in out.items()}


def recovery_stats(results, model: str, log_params=("n", "tau")) -> Dict[str, Dict[str, float]]:
    """Pearson r, Spearman rho and median bias (log scale for positive params)."""
    from scipy.stats import spearmanr
    out = {}
    for p, (tr, rc) in param_table(results, model).items():
        x, y = (np.log(tr), np.log(rc)) if p in log_params else (tr, rc)
        out[p] = {"pearson_r": float(np.corrcoef(x, y)[0, 1]),
                  "spearman_rho": float(spearmanr(tr, rc)[0]),
                  "median_bias": float(np.median(y - x))}
    return out


__all__ = ["WARM_START", "MP_CONTEXT", "FitResult", "fit", "cross_validate", "classify", "wilson_ci", "ci_halfwidth",
           "n_for_halfwidth", "confusion_matrix",
           "inversion_matrix", "recovery_summary", "run_recovery", "failure_rate", "param_table",
           "recovery_stats", "TIE"]
