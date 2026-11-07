"""tinybamdp: small, exact, tabular models of decision-making under uncertainty.

Modules
-------
beliefs    conjugate (Beta, Dirichlet) and discrete belief states
bandit_dp  vectorised exact Bayes-adaptive planning for Bernoulli bandits
bamdp      generic tabular BAMDP with unknown transitions (Dirichlet priors)
pomdp      discrete POMDPs: Bayes filter + point-based value iteration
models     cognitive choice models sharing one agent interface
task       multi-game bandit task: simulation and likelihood (incl. mixed horizons)
fitting    MLE with restarts, cross-validation, tie-aware parameter/model recovery

Study-specific analyses and figures live in the separate ``study`` package.
The core depends only on numpy and scipy.
"""
from . import bandit_dp, bamdp, beliefs, fitting, models, pomdp, task
from .models import MODELS, get_model, register_model
from .task import BanditData, log_likelihood, sample_task, simulate, simulate_design

__version__ = "0.1.0"
__all__ = ["bandit_dp", "bamdp", "beliefs", "fitting", "models", "pomdp", "task",
           "MODELS", "get_model", "register_model", "BanditData", "log_likelihood", "sample_task",
           "simulate", "simulate_design"]
