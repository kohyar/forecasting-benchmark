"""Statistical testing over the results file.

Overlapping test windows make fold errors correlated, so the Diebold-Mariano
test needs the Harvey-Leybourne-Newbold small-sample correction. Both post-hoc
procedures are run because the Nemenyi mean-rank procedure has a known
instability (Benavoli et al. 2016); disagreement between them is a result.
"""
import numpy as np
import pandas as pd
import pytest

from tsbench.stats.tests import (
    bootstrap_skill_ci,
    diebold_mariano,
    friedman_nemenyi,
    holm_adjust,
    wilcoxon_holm,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --- Diebold-Mariano ------------------------------------------------------

def test_identical_forecasts_give_no_evidence_of_difference():
    e = np.array([1.0, -2.0, 3.0, -1.0, 0.5, 2.0, -3.0, 1.5])

    result = diebold_mariano(e, e.copy(), horizon=1)

    assert result["statistic"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)


def test_a_clearly_better_forecast_is_detected(rng):
    good = rng.normal(0, 1, 200)
    bad = rng.normal(0, 3, 200)

    result = diebold_mariano(good, bad, horizon=1)

    assert result["statistic"] < 0, "negative favours the first model"
    assert result["p_value"] < 0.01


def test_the_small_sample_correction_is_applied_and_shrinks_the_statistic(rng):
    a, b = rng.normal(0, 1, 40), rng.normal(0, 1.6, 40)

    corrected = diebold_mariano(a, b, horizon=13)
    raw = diebold_mariano(a, b, horizon=13, harvey_correction=False)

    assert abs(corrected["statistic"]) < abs(raw["statistic"])
    assert corrected["harvey_correction"] is True
    assert corrected["p_value"] > raw["p_value"]


def test_the_correction_factor_matches_the_published_formula(rng):
    a, b = rng.normal(0, 1, 50), rng.normal(0, 2, 50)
    h, n = 13, 50

    corrected = diebold_mariano(a, b, horizon=h)
    raw = diebold_mariano(a, b, horizon=h, harvey_correction=False)
    expected = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)

    assert corrected["statistic"] / raw["statistic"] == pytest.approx(expected)


def test_multi_step_horizons_widen_the_variance(rng):
    """Autocovariances up to h-1 enter the variance for h-step forecasts."""
    a, b = rng.normal(0, 1, 100), rng.normal(0, 1.3, 100)

    one_step = diebold_mariano(a, b, horizon=1, harvey_correction=False)
    thirteen = diebold_mariano(a, b, horizon=13, harvey_correction=False)

    assert one_step["n_autocovariances"] == 0
    assert thirteen["n_autocovariances"] == 12


def test_absolute_error_loss_is_selectable(rng):
    a, b = rng.normal(0, 1, 60), rng.normal(0, 2, 60)

    squared = diebold_mariano(a, b, horizon=1, power=2)
    absolute = diebold_mariano(a, b, horizon=1, power=1)

    assert squared["power"] == 2
    assert absolute["power"] == 1
    assert absolute["statistic"] < 0


def test_a_degenerate_comparison_is_reported_not_crashed():
    result = diebold_mariano(np.zeros(10), np.zeros(10), horizon=1)

    assert np.isnan(result["statistic"]) or result["statistic"] == 0
    assert not np.isinf(result["p_value"])


# --- Friedman and the post-hocs -------------------------------------------

@pytest.fixture
def scores():
    """20 datasets x 3 models; model A is consistently best."""
    rng = np.random.default_rng(3)
    n = 20
    return pd.DataFrame({
        "A": rng.normal(1.0, 0.1, n),
        "B": rng.normal(1.4, 0.1, n),
        "C": rng.normal(1.8, 0.1, n),
    })


def test_friedman_detects_a_consistent_winner(scores):
    result = friedman_nemenyi(scores)

    assert result["p_value"] < 0.001
    assert result["mean_ranks"]["A"] < result["mean_ranks"]["C"]


def test_indistinguishable_models_are_not_separated():
    rng = np.random.default_rng(5)
    tied = pd.DataFrame({k: rng.normal(1.0, 0.1, 30) for k in "ABC"})

    result = friedman_nemenyi(tied)

    assert result["p_value"] > 0.05


def test_the_critical_difference_follows_the_nemenyi_formula(scores):
    from scipy.stats import studentized_range

    result = friedman_nemenyi(scores, alpha=0.05)
    k, n = 3, 20
    q = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)

    assert result["critical_difference"] == pytest.approx(q * np.sqrt(k * (k + 1) / (6 * n)))


def test_nemenyi_reports_which_pairs_separate(scores):
    result = friedman_nemenyi(scores)
    pairs = result["pairs"].set_index(["model_a", "model_b"])

    assert pairs.loc[("A", "C"), "significant"]
    assert "rank_difference" in pairs.columns


def test_wilcoxon_with_holm_agrees_on_a_clear_winner(scores):
    result = wilcoxon_holm(scores)
    pair = result.set_index(["model_a", "model_b"]).loc[("A", "C")]

    assert pair["p_value_adjusted"] < 0.05
    assert pair["significant"]


def test_holm_adjustment_is_monotone_and_conservative():
    raw = [0.001, 0.01, 0.02, 0.04]

    adjusted = holm_adjust(raw)

    assert all(a >= r for a, r in zip(adjusted, raw))
    assert all(b >= a for a, b in zip(adjusted, adjusted[1:]))


def test_holm_matches_a_hand_computed_example():
    # sorted p * (m - i), then cumulative max
    adjusted = holm_adjust([0.01, 0.04])

    assert adjusted[0] == pytest.approx(0.02)
    assert adjusted[1] == pytest.approx(0.04)


def test_the_two_post_hocs_are_compared_and_disagreement_is_visible(scores):
    from tsbench.stats.tests import posthoc_comparison

    result = posthoc_comparison(scores)

    assert {"nemenyi_significant", "wilcoxon_significant", "agree"} <= set(result.columns)
    assert result["agree"].dtype == bool


# --- bootstrap ------------------------------------------------------------

def test_bootstrap_interval_brackets_the_point_estimate(rng):
    skill = rng.normal(0.3, 0.1, 200)

    result = bootstrap_skill_ci(skill, n_resamples=500, seed=1)

    assert result["lower"] < result["estimate"] < result["upper"]
    assert result["estimate"] == pytest.approx(skill.mean(), abs=1e-9)


def test_more_data_narrows_the_interval(rng):
    small = bootstrap_skill_ci(rng.normal(0.3, 0.1, 30), n_resamples=500, seed=1)
    large = bootstrap_skill_ci(rng.normal(0.3, 0.1, 3000), n_resamples=500, seed=1)

    assert (large["upper"] - large["lower"]) < (small["upper"] - small["lower"])


def test_bootstrap_is_reproducible_from_its_seed(rng):
    data = rng.normal(0.3, 0.1, 100)

    first = bootstrap_skill_ci(data, n_resamples=200, seed=7)
    second = bootstrap_skill_ci(data, n_resamples=200, seed=7)

    assert first == second


def test_one_differential_per_series_uses_no_lag_terms(rng):
    """When each observation is a series, not a time step, the horizon must
    not smuggle autocovariance terms into the variance: the series have no
    order to be correlated along, and the correction factor reduces to the
    one-step case whatever the horizon."""
    a, b = rng.normal(0, 1, 200), rng.normal(0, 1.5, 200)

    per_series = diebold_mariano(a, b, horizon=13, power=1, dependence_lags=0)
    one_step = diebold_mariano(a, b, horizon=1, power=1)

    assert per_series["n_autocovariances"] == 0
    assert per_series["statistic"] == pytest.approx(one_step["statistic"])
    assert per_series["p_value"] == pytest.approx(one_step["p_value"])
    raw = diebold_mariano(a, b, horizon=13, power=1, dependence_lags=0,
                          harvey_correction=False)
    assert per_series["statistic"] / raw["statistic"] == pytest.approx(np.sqrt((200 - 1) / 200))


def test_clustered_variance_reduces_to_independent_when_every_series_is_its_own_cluster(rng):
    a, b = rng.normal(0, 1, 120), rng.normal(0, 1.4, 120)
    independent = diebold_mariano(a, b, horizon=1, power=1, harvey_correction=False)
    clustered = diebold_mariano(a, b, horizon=13, power=1, clusters=np.arange(120))

    assert clustered["n_clusters"] == 120
    assert clustered["n_autocovariances"] == 0
    assert clustered["statistic"] == pytest.approx(independent["statistic"])
    assert clustered["p_value"] == pytest.approx(independent["p_value"])


def test_shared_shocks_within_a_cluster_widen_the_variance(rng):
    """Twenty products in thirty markets each: a product-level shock moves
    every market's differential together, and treating the 600 series as
    independent would overstate the evidence."""
    products = np.repeat(np.arange(20), 30)
    shock = rng.normal(0, 1, 20)[products]
    d = 0.1 + shock + rng.normal(0, 0.3, 600)
    a, b = d, np.zeros(600)

    independent = diebold_mariano(a, b, horizon=1, power=1, dependence_lags=0)
    clustered = diebold_mariano(a, b, horizon=1, power=1, clusters=products)

    assert clustered["n_clusters"] == 20
    assert abs(clustered["statistic"]) < abs(independent["statistic"])
    assert clustered["p_value"] > independent["p_value"]
