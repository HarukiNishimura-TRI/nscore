"""Tests for deprecation shims in nscore.tools.plotting."""

import warnings

import numpy as np
import pytest

import statistical_comparison_helpers as sch


class TestCLDShim:
    def test_emits_deprecation_warning(self):
        from nscore.tools.plotting import compact_letter_display

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compact_letter_display(
                [("A", "B")], ["A", "B", "C"]
            )
            depr_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert depr_warnings
            assert any("deprecated" in str(x.message).lower() for x in depr_warnings)
    def test_output_matches_statistical_comparison_helpers(self):
        from nscore.tools.plotting import compact_letter_display

        pairs = [("A", "B"), ("B", "C")]
        models = ["A", "B", "C", "D"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            shim_result = compact_letter_display(pairs, models)

        expected = sch.compact_letter_display(pairs, models)
        assert shim_result == expected

    def test_returns_list_of_str(self):
        from nscore.tools.plotting import compact_letter_display

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = compact_letter_display([], ["A", "B"])

        assert isinstance(result, list)
        assert all(isinstance(x, str) for x in result)


class TestBetaPosteriorShim:
    def test_emits_deprecation_warning(self):
        from nscore.tools.plotting import draw_samples_from_beta_posterior

        rng = np.random.default_rng(42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            draw_samples_from_beta_posterior(np.array([1, 0, 1]), rng, num_samples=100)
            depr_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert depr_warnings
            assert any("deprecated" in str(x.message).lower() for x in depr_warnings)

    def test_output_shape(self):
        from nscore.tools.plotting import draw_samples_from_beta_posterior

        rng = np.random.default_rng(42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = draw_samples_from_beta_posterior(np.array([1, 0, 1, 1]), rng, num_samples=500)
        assert result.shape == (500,)


class TestPlotShim:
    def test_emits_deprecation_warning(self):
        from nscore.tools.plotting import plot_model_comparison
        import matplotlib
        matplotlib.use("Agg")

        rng = np.random.default_rng(42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = plot_model_comparison(
                ["A", "B"],
                [np.array([1, 0, 1]), np.array([0, 1, 0])],
                ["a", "b"],
                rng,
            )
            depr_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert depr_warnings
            assert any("deprecated" in str(x.message).lower() for x in depr_warnings)

        import matplotlib.pyplot as plt
        plt.close(fig)
