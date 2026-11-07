"""Matplotlib figures with a fixed, colour-blind-checked model palette.

Model colours: the IBM Design colour-blind-safe palette (blue, orange, purple,
gold, magenta-pink), with the gold darkened from #FFB000 to #D99A00 so it stays
visible on white.  The order below passes the adjacent-pair colour-blind and
normal-vision separation checks; every figure also labels each colour.
Sequential maps use ColorBrewer RdPu (white -> pink -> purple), diverging maps PiYG.
"""
from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Fixed model -> colour mapping (never cycled).
COLORS = {"RW": "#648FFF", "WSLS": "#FE6100", "KG": "#785EF0", "KG-DBM": "#D99A00", "OPT": "#DC267F"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = matplotlib.colormaps["RdPu"]
DIV = matplotlib.colormaps["PiYG"]
ACCENT = "#648FFF"        # markers drawn on top of sequential maps (e.g. the true parameter value)


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 220, "font.size": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
        "lines.linewidth": 2, "lines.markersize": 6,
    })


def confusion(ax, M, labels, title="Model recovery", annotate=True, xlabel="best-fitting model (BIC)",
              ylabel="generating model", col_labels=None):
    """Heat-map of a (possibly non-square, e.g. with a 'tie' column) recovery matrix."""
    col_labels = col_labels or (list(labels) + ["tie"] if M.shape[1] == len(labels) + 1 else list(labels))
    im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=30 if len(col_labels) > 4 else 0)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isnan(M[i, j]):
                    continue
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if M[i, j] > 0.55 else INK)
    ax.set_title(title)
    return im


def recovery_scatter(ax, true, rec, name, color, log=False):
    ax.scatter(true, rec, s=14, color=color, alpha=0.7, edgecolor="white", linewidth=0.5)
    lo, hi = min(true.min(), rec.min()), max(true.max(), rec.max())
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
    if log:
        from matplotlib.ticker import LogLocator, NullFormatter, FormatStrFormatter
        ax.set_xscale("log"), ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
            axis.set_major_formatter(FormatStrFormatter("%g"))
            axis.set_minor_formatter(NullFormatter())
        r = np.corrcoef(np.log(true), np.log(rec))[0, 1]
    else:
        r = np.corrcoef(true, rec)[0, 1]
    ax.set_title(f"{name}   r = {r:.2f}", fontsize=9)
    ax.set_xlabel("simulated"), ax.set_ylabel("recovered")
    return r


def surface(ax, X, Y, Z, xlabel, ylabel, truth=None, mle=None):
    Zr = Z - Z.max()
    cs = ax.contourf(X, Y, np.maximum(Zr, -30), levels=15, cmap=SEQ)
    ax.contour(X, Y, Zr, levels=[-1.92], colors=["white"], linewidths=1.4, linestyles="solid")  # ~95% LR region (white on the dark peak)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel(xlabel), ax.set_ylabel(ylabel)
    ax.grid(False)
    if truth is not None:
        ax.plot(*truth, marker="*", ms=12, color=ACCENT, mec="white", ls="none", label="true")
    if mle is not None:
        ax.plot(*mle, marker="o", ms=7, color="white", mec=INK, ls="none", label="grid max")
    ax.legend(loc="upper right", fontsize=8)
    return cs


__all__ = ["COLORS", "INK", "MUTED", "GRID", "SEQ", "DIV", "ACCENT", "style", "confusion", "recovery_scatter", "surface"]
