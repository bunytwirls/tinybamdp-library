"""Analyses, figures and experiments for the KG-vs-Bayes-optimal identifiability study.

Kept outside the core ``tinybamdp`` package so the library itself depends only on numpy and scipy.
"""
from . import analysis, plots

__all__ = ["analysis", "plots"]
