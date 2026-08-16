"""Metrics, and the seasonal denominator every scaled metric depends on.

If the denominator is wrong, every MASE and RMSSE in the paper is wrong, so it
is computed once per series from training data and passed in - never
recomputed per model.
"""
import numpy as np
import pandas as pd
import pytest

from tsbench.config import Config
from tsbench.data.loader import normalize_panel
from tsbench.eval.metrics import (
    denominators_for_protocol,
    evaluate,
    seasonal_denominators,
)
from tsbench.eval.splitter import RollingOriginSplitter

WEEK = pd.Timedelta(days=7)


def frame(uid, values, start="2022-01-09", col="y"):
    ds = pd.date_range(start, periods=len(values), freq="7D")
    return pd.DataFrame({"unique_id": uid, "ds": ds, col: np.asarray(values, dtype=float)})


@pytest.fixture
def train():
    return frame("A", [2, 4, 6, 8, 4, 8, 12, 16])


@pytest.fixture
def actual():
    return frame("A", [10, 12, 14, 16], start="2022-02-27")


@pytest.fixture
def forecast():
    return frame("A", [11, 11, 15, 15], start="2022-02-27", col="yhat")


@pytest.fixture
def den(train):
    return seasonal_denominators(train, season_length=4)


# --- the denominator ------------------------------------------------------

def test_denominator_is_the_in_sample_seasonal_naive_error(den):
    """Differences at lag 4 are [2, 4, 6, 8]."""
    row = den.set_index("unique_id").loc["A"]

    assert row["mase_denom"] == pytest.approx(5.0)
    assert row["rmsse_denom"] == pytest.approx(np.sqrt(30.0))
    assert row["n_pairs"] == 4


def test_denominator_depends_on_the_season_length(train):
    """The guard against a library default of 1, 7 or 12 on weekly data."""
    wrong = seasonal_denominators(train, season_length=1)

    assert wrong.loc[0, "mase_denom"] != pytest.approx(5.0)


def test_gaps_are_skipped_and_the_pair_count_reports_it():
    t = frame("A", [2, 4, 6, 8, np.nan, 8, 12, 16])
    den = seasonal_denominators(t, season_length=4).set_index("unique_id").loc["A"]

    assert den["n_pairs"] == 3, "the pair spanning the missing week is dropped"
    assert den["mase_denom"] == pytest.approx((4 + 6 + 8) / 3)


def test_series_shorter_than_one_season_yields_no_denominator():
    t = frame("A", [1, 2, 3])
    den = seasonal_denominators(t, season_length=4).set_index("unique_id").loc["A"]

    assert den["n_pairs"] == 0
    assert np.isnan(den["mase_denom"])


def test_flat_series_is_flagged_rather_than_dividing_by_zero():
    t = frame("A", [5.0] * 10)
    den = seasonal_denominators(t, season_length=4).set_index("unique_id").loc["A"]

    assert den["mase_denom"] == 0
    assert bool(den["zero_denominator"])


def test_protocol_denominator_uses_only_pre_origin_data(raw_frame, config_dict):
    """Corrupting everything after the earliest origin must not move it."""
    cfg = Config.from_dict(config_dict)
    panel = normalize_panel(raw_frame, cfg)
    splitter = RollingOriginSplitter(cfg)
    earliest = splitter.origins(panel)[0]

    poisoned = panel.copy()
    poisoned.loc[poisoned["ds"] > earliest, "y"] = 1e9

    pd.testing.assert_frame_equal(
        denominators_for_protocol(panel, cfg, splitter),
        denominators_for_protocol(poisoned, cfg, splitter))


def test_protocol_denominator_is_identical_across_folds(raw_frame, config_dict):
    cfg = Config.from_dict(config_dict)
    panel = normalize_panel(raw_frame, cfg)
    splitter = RollingOriginSplitter(cfg)

    den = denominators_for_protocol(panel, cfg, splitter)

    assert den["unique_id"].is_unique, "one denominator per series, not per fold"


# --- point metrics --------------------------------------------------------

def kv(rows):
    return dict(zip(rows["metric"], rows["value"]))


def test_point_metrics_are_computed_as_defined(actual, forecast, den):
    m = kv(evaluate(actual, forecast, den, model="naive", fold=0, horizon=4))

    assert m["MAE"] == pytest.approx(1.0)
    assert m["RMSE"] == pytest.approx(1.0)
    assert m["MASE"] == pytest.approx(1 / 5)
    assert m["RMSSE"] == pytest.approx(1 / np.sqrt(30))


def test_mase_uses_the_denominator_it_is_given(actual, forecast, den):
    """Halving the denominator doubles MASE - proof it is not recomputed."""
    halved = den.copy()
    halved[["mase_denom", "rmsse_denom"]] /= 2

    base = kv(evaluate(actual, forecast, den, model="m", fold=0, horizon=4))
    scaled = kv(evaluate(actual, forecast, halved, model="m", fold=0, horizon=4))

    assert scaled["MASE"] == pytest.approx(2 * base["MASE"])
    assert scaled["RMSSE"] == pytest.approx(2 * base["RMSSE"])


def test_missing_actuals_are_excluded_and_counted(den):
    a = frame("A", [10, np.nan, 14, 16], start="2022-02-27")
    f = frame("A", [11, 11, 15, 15], start="2022-02-27", col="yhat")

    rows = evaluate(a, f, den, model="m", fold=0, horizon=4)

    assert rows["n_points"].unique().tolist() == [3]
    assert kv(rows)["MAE"] == pytest.approx(1.0)


def test_smape_reports_undefined_points_instead_of_dropping_them_silently(den):
    a = frame("A", [0.0, 10.0], start="2022-02-27")
    f = frame("A", [0.0, 10.0], start="2022-02-27", col="yhat")

    rows = evaluate(a, f, den, model="m", fold=0, horizon=2)
    smape = rows[rows["metric"] == "sMAPE"].iloc[0]

    assert smape["n_undefined"] == 1
    assert smape["n_points"] == 1
    assert smape["value"] == pytest.approx(0.0)


def test_mape_is_undefined_at_zero_actuals(den):
    a = frame("A", [0.0, 10.0], start="2022-02-27")
    f = frame("A", [1.0, 11.0], start="2022-02-27", col="yhat")

    mape = evaluate(a, f, den, model="m", fold=0, horizon=2)
    mape = mape[mape["metric"] == "MAPE"].iloc[0]

    assert mape["n_undefined"] == 1
    assert mape["value"] == pytest.approx(10.0)


def test_zero_denominator_series_yields_nan_not_inf():
    t = frame("A", [5.0] * 10)
    den = seasonal_denominators(t, season_length=4)
    a = frame("A", [5, 5], start="2022-03-27")
    f = frame("A", [6, 6], start="2022-03-27", col="yhat")

    m = kv(evaluate(a, f, den, model="m", fold=0, horizon=2))

    assert np.isnan(m["MASE"])
    assert not np.isinf(m["MASE"])


# --- probabilistic metrics ------------------------------------------------

def quantile_forecast(uid, med, spread, n, start="2022-02-27"):
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    f = frame(uid, [med] * n, start=start, col="yhat")
    for q in levels:
        f[f"yhat_q{int(q * 100)}"] = med + spread * (q - 0.5) * 2
    return f


def test_pinball_loss_matches_the_definition(den):
    a = frame("A", [10.0], start="2022-02-27")
    f = frame("A", [8.0], start="2022-02-27", col="yhat")
    f["yhat_q90"] = 8.0

    rows = evaluate(a, f, den, model="m", fold=0, horizon=1, quantile_levels=[0.9])

    assert kv(rows)["pinball"] == pytest.approx(0.9 * 2.0)


def test_crps_of_a_degenerate_forecast_equals_mae(den):
    """All quantiles collapsed onto the point forecast."""
    a = frame("A", [10.0, 12.0], start="2022-02-27")
    f = quantile_forecast("A", 9.0, spread=0.0, n=2)

    m = kv(evaluate(a, f, den, model="m", fold=0, horizon=2))

    assert m["CRPS"] == pytest.approx(m["MAE"])


def test_crps_rewards_a_sharper_interval_when_both_are_centred(den):
    a = frame("A", [10.0, 10.0], start="2022-02-27")
    tight = quantile_forecast("A", 10.0, spread=1.0, n=2)
    wide = quantile_forecast("A", 10.0, spread=5.0, n=2)

    tight_crps = kv(evaluate(a, tight, den, model="m", fold=0, horizon=2))["CRPS"]
    wide_crps = kv(evaluate(a, wide, den, model="m", fold=0, horizon=2))["CRPS"]

    assert tight_crps < wide_crps


def test_empirical_coverage_at_the_nominal_level(den):
    a = frame("A", [10.0, 10.0, 10.0, 100.0], start="2022-02-27")
    f = quantile_forecast("A", 10.0, spread=2.0, n=4)

    m = kv(evaluate(a, f, den, model="m", fold=0, horizon=4, coverage_levels=[0.8]))

    assert m["coverage_80"] == pytest.approx(0.75)


def test_point_only_models_emit_no_probabilistic_rows(actual, forecast, den):
    rows = evaluate(actual, forecast, den, model="naive", fold=0, horizon=4)

    assert not {"CRPS", "pinball", "coverage_80"} & set(rows["metric"])


# --- output shape ---------------------------------------------------------

def test_output_is_long_format_with_the_traceability_columns(actual, forecast, den):
    rows = evaluate(actual, forecast, den, model="naive", fold=0, horizon=4)

    assert set(rows.columns) == {"model", "fold", "horizon", "unique_id", "metric",
                                 "value", "n_points", "n_undefined"}
    assert (rows["model"] == "naive").all()
    assert (rows["fold"] == 0).all()
    assert (rows["horizon"] == 4).all()
    assert rows.duplicated(["model", "fold", "horizon", "unique_id", "metric"]).sum() == 0


def test_multiple_series_are_scored_independently(den):
    den2 = pd.concat([den, den.assign(unique_id="B")], ignore_index=True)
    a = pd.concat([frame("A", [10, 10], start="2022-02-27"),
                   frame("B", [10, 10], start="2022-02-27")], ignore_index=True)
    f = pd.concat([frame("A", [11, 11], start="2022-02-27", col="yhat"),
                   frame("B", [13, 13], start="2022-02-27", col="yhat")], ignore_index=True)

    rows = evaluate(a, f, den2, model="m", fold=0, horizon=2)
    mae = rows[rows["metric"] == "MAE"].set_index("unique_id")["value"]

    assert mae["A"] == pytest.approx(1.0)
    assert mae["B"] == pytest.approx(3.0)


def test_a_series_missing_from_the_forecast_is_an_error(den):
    a = pd.concat([frame("A", [10, 10], start="2022-02-27"),
                   frame("B", [10, 10], start="2022-02-27")], ignore_index=True)
    f = frame("A", [11, 11], start="2022-02-27", col="yhat")

    with pytest.raises(ValueError, match="missing forecast"):
        evaluate(a, f, den, model="m", fold=0, horizon=2)
