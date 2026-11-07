import numpy as np
import pytest

from study import analysis


@pytest.mark.parametrize("K", [2, 3, 4])


def test_kg_opt_equivalent_at_T3_for_any_K(K):
    rng = np.random.default_rng(K)
    for a, b in np.exp(rng.uniform(np.log(0.1), np.log(10), size=(15, 2))):
        assert analysis.kg_opt_gap_difference(K, 3, a, b) < 1e-12


def test_kg_opt_not_equivalent_at_T4_in_general():
    assert analysis.kg_opt_gap_difference(2, 4, 0.7, 3.0) > 1e-3


def test_disagreement_excludes_ties_of_both_policies():
    d = analysis.policy_disagreement(3, G=500)       # KG == OPT at T=3, so no real disagreements
    assert np.all(d == 0)


@pytest.mark.parametrize("K", [2, 3])
def test_kg_exact_with_two_choices_left_for_any_posteriors(K):
    rng = np.random.default_rng(20 + K)
    for _ in range(20):
        a, b = np.exp(rng.uniform(np.log(0.1), np.log(10), size=(2, K)))
        assert analysis.kg_opt_gap_difference_hetero(a, b, 2) < 1e-12


def test_kg_not_exact_with_three_choices_left_for_unequal_posteriors():
    assert analysis.kg_opt_gap_difference_hetero([0.4, 4.9], [0.2, 2.9], 3) > 0.1   # found by random search


def test_hetero_solver_matches_symmetric_solver_for_shared_prior():
    assert np.isclose(analysis.kg_opt_gap_difference_hetero([0.7, 0.7], [3.0, 3.0], 4),
                      analysis.kg_opt_gap_difference(2, 4, 0.7, 3.0))
