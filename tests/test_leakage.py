"""No observation at or after a forecast origin may reach any model.

This is the non-negotiable. Foundation models are included: a zero-shot model
still receives a context window, and that context must stop at the origin.
"""
import numpy as np
import pandas as pd
import pytest

from tsbench.config import Config
from tsbench.data.loader import normalize_panel
from tsbench.data.schema import covariate_groups
from tsbench.eval.splitter import RollingOriginSplitter

WEEK = pd.Timedelta(days=7)


@pytest.fixture
def cfg(config_dict):
    return Config.from_dict(config_dict)


@pytest.fixture
def panel(raw_frame, cfg):
    return normalize_panel(raw_frame, cfg)


@pytest.fixture
def folds(cfg, panel):
    return list(RollingOriginSplitter(cfg).split(panel))


def test_no_training_timestamp_is_at_or_after_the_origin(folds):
    for fold in folds:
        assert (fold.train["ds"] <= fold.origin).all()


def test_no_test_timestamp_ever_appears_in_training(folds, cfg):
    for fold in folds:
        for h in cfg.protocol.horizons:
            overlap = set(fold.test(h)["ds"]) & set(fold.train["ds"])
            assert not overlap, f"fold {fold.fold_id} h={h} leaks {sorted(overlap)}"


def test_training_covariates_stop_at_the_origin(folds):
    """Not just the target - every contemporaneous covariate column too."""
    groups = covariate_groups("Dollars")
    for fold in folds:
        after = fold.train[fold.train["ds"] > fold.origin]
        for col in groups.past_observed + ["y"]:
            assert after[col].isna().all() if len(after) else True


def test_future_frame_carries_no_realised_measures(folds):
    """What a model may see for the forecast window: known_future only."""
    groups = covariate_groups("Dollars")
    for fold in folds:
        future = fold.future_covariates(3)
        for col in groups.past_observed + ["y"]:
            assert col not in future.columns


def test_yago_covariates_are_rebuilt_from_training_not_read_from_the_future(raw_frame, cfg):
    """The export ships _Yago as a column. Reading it at a future timestamp
    would be reading a row that does not exist yet at the origin, so the
    harness reconstructs it from the training target instead."""
    poisoned = raw_frame.copy()
    poisoned["Dollars_Yago"] = 999_999.0

    panel = normalize_panel(poisoned, cfg)
    fold = next(iter(RollingOriginSplitter(cfg).split(panel)))
    future = fold.future_covariates(3)

    assert not (future["Dollars_Yago"] == 999_999.0).any()

    a = future[future["unique_id"] == "SUB_A@@BOSTON, MA - MULO"].sort_values("ds")
    train_a = fold.train[fold.train["unique_id"] == "SUB_A@@BOSTON, MA - MULO"].set_index("ds")["y"]
    expected = [train_a.get(ds - 52 * WEEK, np.nan) for ds in a["ds"]]

    np.testing.assert_allclose(a["Dollars_Yago"].to_numpy(), expected)


def test_planned_price_is_the_last_price_seen_at_the_origin(folds):
    """ARP is realised price, so future ARP is unknowable. What a planner does
    have is today's price carried forward - built from training data only."""
    for fold in folds:
        future = fold.future_covariates(3)
        last_seen = (fold.train.dropna(subset=["ARP"])
                     .sort_values("ds").groupby("unique_id")["ARP"].last())

        for uid, g in future.groupby("unique_id"):
            assert g["ARP_planned"].nunique() == 1, "one price, held across the horizon"
            assert g["ARP_planned"].iloc[0] == pytest.approx(last_seen[uid])


def test_planned_price_ignores_future_prices(raw_frame, cfg):
    poisoned = raw_frame.copy()
    after = pd.to_datetime(poisoned["Time_Period_End_Date"]) > pd.Timestamp("2022-12-01")
    poisoned.loc[after, "ARP"] = 12_345.0

    panel = normalize_panel(poisoned, cfg)
    fold = next(iter(RollingOriginSplitter(cfg).split(panel)))

    clean = normalize_panel(raw_frame, cfg)
    clean_fold = next(iter(RollingOriginSplitter(cfg).split(clean)))

    assert fold.origin > pd.Timestamp("2022-12-01"), "the poison starts inside training"
    assert (fold.future_covariates(3)["ARP_planned"] == 12_345.0).all(), \
        "prices before the origin are legitimately used"
    assert not (clean_fold.future_covariates(3)["ARP_planned"] == 12_345.0).any()


def test_realised_price_is_never_exposed_for_the_forecast_window(folds):
    for fold in folds:
        assert "ARP" not in fold.future_covariates(3).columns


def test_future_frame_only_spans_the_requested_horizon(folds):
    for fold in folds:
        future = fold.future_covariates(1)

        assert future["ds"].min() == fold.origin + WEEK
        assert future["ds"].max() == fold.origin + WEEK


def test_actuals_are_never_handed_out_with_the_future_frame(folds):
    for fold in folds:
        assert "y" not in fold.future_covariates(3).columns


def test_a_model_seeing_only_train_cannot_reconstruct_the_test_target(folds):
    """Regression guard: the frames handed to an adapter share no rows."""
    for fold in folds:
        merged = fold.train.merge(fold.test(3), on=["unique_id", "ds"], how="inner")

        assert merged.empty
