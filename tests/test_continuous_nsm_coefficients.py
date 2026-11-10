"""Self-contained equivalence checks against the original scalar grid search.

All inputs are generated here. No external datasets or AnyEval/AnyRank imports
are needed; the reference deliberately retains the pre-refactor arithmetic.
"""
from copy import deepcopy
import pickle

import numpy as np
import pytest
from scipy.stats import dirichlet

from nscore.nonparametric_nsm import (
    ContinuousNsmTest,
    MirroredContinuousNsmTest,
    _coefficient_grid,
)
from statistical_comparison_core import Hypothesis


LAMBDAS = (np.arange(100) + 0.5) / 100
ALTERNATIVES = [Hypothesis.P0LessThanP1, Hypothesis.P0MoreThanP1]


def scalar_objectives(test, pbar, delta):
    values = np.zeros(100)
    for index, coefficient in enumerate(LAMBDAS):
        value = 0.0
        for i in range(test.K - 1):
            for j in range(i + 1, test.K):
                difference = test.c[j] - test.c[i]
                signal = coefficient * np.sign(delta[i, j]) * difference
                if test.alternative is Hypothesis.P0LessThanP1:
                    signal_log = np.log(1.0 + signal)
                else:
                    signal_log = np.log(1.0 - signal)
                value += np.abs(delta[i, j]) * signal_log
                value += pbar[i, j] * np.log(
                    1.0 - coefficient * coefficient * difference * difference
                )
        values[index] = value
    return values


def posterior_terms(test):
    p0 = test._dirichlet_posterior_0.mean().reshape(-1, 1)
    p1 = test._dirichlet_posterior_1.mean().reshape(-1, 1)
    joint = p0 @ p1.T
    pbar = np.zeros((test.K, test.K))
    delta = np.zeros((test.K, test.K))
    for i in range(test.K - 1):
        for j in range(i + 1, test.K):
            pbar[i, j] = np.minimum(joint[i, j], joint[j, i])
            delta[i, j] = joint[i, j] - joint[j, i]
    return pbar, delta


class ScalarContinuousNsmTest(ContinuousNsmTest):
    """Original search, independent of cached tables and vector evaluation."""

    def _compute_optimal_lambda_long(self, verbose=False):
        less = self.alternative is Hypothesis.P0LessThanP1
        if ((less and self._estimated_mu_1 <= self._estimated_mu_0)
                or (not less and self._estimated_mu_0 <= self._estimated_mu_1)):
            self.lambda_parameter = 0.0
        else:
            values = scalar_objectives(self, *posterior_terms(self))
            self.lambda_parameter = LAMBDAS[np.argmax(values)]
        self._store_lambda_params.append(self.lambda_parameter)


class ScalarMirroredNsmTest(MirroredContinuousNsmTest):
    _base_class = ScalarContinuousNsmTest


def set_counts(test, first, second):
    test._alpha_0 = np.array(first, dtype=float)
    test._alpha_1 = np.array(second, dtype=float)
    test._dirichlet_posterior_0 = dirichlet(test._alpha_0)
    test._dirichlet_posterior_1 = dirichlet(test._alpha_1)
    test._estimate_parameters(False)


GRIDS = [
    pytest.param(np.array([0.5]), id="single-bin"),
    pytest.param(np.arange(2), id="binary-int"),
    pytest.param(np.array([0.0, 1.0]), id="binary-float"),
    pytest.param(np.arange(11) / 10, id="eleven-bin"),
    pytest.param(np.array([0.0, 0.07, 0.3, 0.71, 1.0]), id="nonuniform"),
    pytest.param(np.array([0.0, 0.0, 0.4, 1.0]), id="duplicate-bins"),
    pytest.param(np.linspace(0, 1, 9)[::2], id="noncontiguous"),
    pytest.param(np.array([0.0, 0.1, 0.7, 1.0], dtype=np.float32), id="float32"),
    pytest.param(np.array([0.0, 0.1, 0.7, 1.0], dtype=np.longdouble), id="longdouble"),
]


@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_every_objective_and_selected_coefficient_matches_scalar(grid, alternative):
    actual = ContinuousNsmTest(alternative, 0.05, grid)
    reference = ScalarContinuousNsmTest(alternative, 0.05, grid)
    rng = np.random.default_rng(20261109)
    for n in (0, 1, 10, 1000, 1000000):
        first = 2 / len(grid) + rng.multinomial(n, rng.dirichlet(np.ones(len(grid))))
        second = 2 / len(grid) + rng.multinomial(n, rng.dirichlet(np.ones(len(grid))))
        for test in (actual, reference):
            set_counts(test, first, second)
        terms = posterior_terms(reference)
        np.testing.assert_array_equal(actual._f_of_lambdas(*terms), scalar_objectives(reference, *terms))
        actual._compute_optimal_lambda_long(False)
        reference._compute_optimal_lambda_long(False)
        assert actual.lambda_parameter == reference.lambda_parameter
    assert actual._store_lambda_params == reference._store_lambda_params


def test_documented_101_bin_grid_matches_scalar():
    test = ContinuousNsmTest(ALTERNATIVES[0], 0.05, np.arange(101) / 100)
    set_counts(test, np.arange(1, 102), np.arange(101, 0, -1))
    terms = posterior_terms(test)
    np.testing.assert_array_equal(test._f_of_lambdas(*terms), scalar_objectives(test, *terms))


def assert_child_state_equal(actual, reference):
    for name in ("lambda_parameter", "_martingale", "_p_value", "_p_value_less",
                 "_p_value_more", "_t", "_estimated_mu_0", "_estimated_mu_1"):
        assert getattr(actual, name) == getattr(reference, name), name
    assert actual._store_lambda_params == reference._store_lambda_params
    np.testing.assert_array_equal(actual._alpha_0, reference._alpha_0)
    np.testing.assert_array_equal(actual._alpha_1, reference._alpha_1)


def observations():
    # Equal scores, strong evidence in both directions, then interior/boundary values.
    rng = np.random.default_rng(42)
    return [(0.5, 0.5)] * 2 + [(0.0, 1.0)] * 12 + [(1.0, 0.0)] * 12 + list(rng.random((10, 2)))


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_full_sequential_state_and_reset_match_scalar(alternative):
    actual = ContinuousNsmTest(alternative, 0.05, np.arange(11) / 10)
    reference = ScalarContinuousNsmTest(alternative, 0.05, np.arange(11) / 10)
    assert_child_state_equal(actual, reference)
    for first, second in observations():
        assert actual.step(first, second) == reference.step(first, second)
        assert_child_state_equal(actual, reference)
    actual.reset()
    reference.reset()
    assert_child_state_equal(actual, reference)
    for first, second in observations()[:5]:
        assert actual.step(first, second) == reference.step(first, second)
        assert_child_state_equal(actual, reference)


@pytest.mark.parametrize("mode", ["comparison", "ranking", "ranking_no_ties"])
@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_mirrored_modes_match_scalar_at_every_step(mode, alternative):
    kwargs = dict(alternative=alternative, alpha=0.05, c=np.arange(11) / 10, inference_mode=mode)
    actual = MirroredContinuousNsmTest(**kwargs)
    reference = ScalarMirroredNsmTest(**kwargs)
    for first, second in observations():
        assert actual.step(first, second) == reference.step(first, second)
        for name in ("_p_value", "_p_value_less", "_p_value_more", "_p_type"):
            assert getattr(actual, name) == getattr(reference, name)
        for name in ("_test_for_alternative", "_test_for_null"):
            assert_child_state_equal(getattr(actual, name), getattr(reference, name))
    actual.reset()
    reference.reset()
    for name in ("_test_for_alternative", "_test_for_null"):
        assert_child_state_equal(getattr(actual, name), getattr(reference, name))


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_zero_boundary_and_flat_objective_tie(alternative):
    first, second = ([1000000, 1], [1, 1000000])
    if alternative is Hypothesis.P0MoreThanP1:
        first, second = second, first
    test = ContinuousNsmTest(alternative, 0.05, np.array([0.0, 1.0]))
    set_counts(test, first, second)
    test._compute_optimal_lambda_long(False)
    assert test.lambda_parameter == 0.995
    set_counts(test, second, first)
    test._compute_optimal_lambda_long(False)
    assert test.lambda_parameter == 0.0
    set_counts(test, first, first)
    test._compute_optimal_lambda_long(False)
    assert test.lambda_parameter == 0.0

    # Positive mean difference, but log arguments round to 1 for all candidates.
    test = ContinuousNsmTest(alternative, 0.05, np.array([0.0, 1e-20]))
    set_counts(test, first, second)
    terms = posterior_terms(test)
    np.testing.assert_array_equal(test._f_of_lambdas(*terms), np.zeros(100))
    np.testing.assert_array_equal(scalar_objectives(test, *terms), np.zeros(100))
    test._compute_optimal_lambda_long(False)
    assert test.lambda_parameter == 0.005  # Original first-maximum tie break.


def test_grid_edits_and_dtype_changes_do_not_reuse_stale_terms():
    grid = np.array([0.0, 0.1, 0.7, 1.0], dtype=np.float32)
    test = ContinuousNsmTest(ALTERNATIVES[0], 0.05, grid)
    set_counts(test, [2, 8, 3, 1], [1, 3, 8, 2])
    terms = posterior_terms(test)
    before = test._f_of_lambdas(*terms)
    grid[1] = 0.4
    after = test._f_of_lambdas(*terms)
    assert not np.array_equal(before, after)
    np.testing.assert_array_equal(after, scalar_objectives(test, *terms))
    test.c = grid.astype(np.float64)
    np.testing.assert_array_equal(test._f_of_lambdas(*terms), scalar_objectives(test, *terms))


def test_tables_are_shared_read_only_and_bounded():
    _coefficient_grid.cache_clear()
    grid = np.array([0.0, 1.0])
    tables = _coefficient_grid(grid.dtype.str, grid.tobytes())
    copy = grid.copy()
    assert _coefficient_grid(copy.dtype.str, copy.tobytes()) is tables
    assert all(not table.flags.writeable for table in tables)
    for endpoint in np.linspace(0.1, 0.9, 10):
        different_grid = np.array([0.0, endpoint])
        _coefficient_grid(different_grid.dtype.str, different_grid.tobytes())
    assert _coefficient_grid.cache_info().currsize <= 8


@pytest.mark.parametrize("clone", [deepcopy, lambda test: pickle.loads(pickle.dumps(test))])
def test_copied_tests_continue_with_independent_histories(clone):
    original = ContinuousNsmTest(ALTERNATIVES[0], 0.05, np.arange(11) / 10)
    reference = ScalarContinuousNsmTest(ALTERNATIVES[0], 0.05, np.arange(11) / 10)
    for first, second in observations()[:5]:
        original.step(first, second)
        reference.step(first, second)
    actual = clone(original)
    saved_history = list(original._store_lambda_params)
    for first, second in observations()[5:10]:
        assert actual.step(first, second) == reference.step(first, second)
        assert_child_state_equal(actual, reference)
    assert original._store_lambda_params == saved_history


def test_verbose_path_preserves_objectives_and_diagnostics(capsys):
    actual = ContinuousNsmTest(ALTERNATIVES[0], 0.05, np.arange(11) / 10)
    reference = ScalarContinuousNsmTest(ALTERNATIVES[0], 0.05, np.arange(11) / 10)
    assert actual.step(0.0, 1.0, verbose=True) == reference.step(0.0, 1.0)
    assert_child_state_equal(actual, reference)
    output = capsys.readouterr().out
    for label in ("Current Phat:", "Current Pbar:", "Current delta_P:", "Estimated optimal lambda:"):
        assert label in output
