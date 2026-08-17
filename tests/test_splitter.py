"""Rolling-origin, expanding-window protocol."""
import pandas as pd
import pytest

from tsbench.config import Config
from tsbench.data.loader import normalize_panel
from tsbench.eval.splitter import ProtocolError, RollingOriginSplitter

WEEK = pd.Timedelta(days=7)


@pytest.fixture
def cfg(config_dict):
    return Config.from_dict(config_dict)


@pytest.fixture
def panel(raw_frame, cfg):
    return normalize_panel(raw_frame, cfg)


@pytest.fixture
def splitter(cfg):
    return RollingOriginSplitter(cfg)


def test_produces_the_configured_number_of_origins(splitter, panel):
    assert len(splitter.origins(panel)) == 2


def test_origins_are_spaced_by_step_not_by_horizon(splitter, panel):
    origins = splitter.origins(panel)

    assert all(b - a == 2 * WEEK for a, b in zip(origins, origins[1:]))


def test_last_origin_leaves_room_for_the_longest_horizon(splitter, panel):
    assert splitter.origins(panel)[-1] == panel["ds"].max() - 3 * WEEK


def test_origins_sit_on_the_weekly_grid(splitter, panel):
    grid = set(panel["ds"])

    assert all(o in grid for o in splitter.origins(panel))


def test_train_never_reaches_past_its_origin(splitter, panel):
    for fold in splitter.split(panel):
        assert fold.train["ds"].max() <= fold.origin


def test_window_expands_as_origins_advance(splitter, panel):
    folds = list(splitter.split(panel))

    assert folds[0].train["ds"].min() == folds[1].train["ds"].min()
    assert len(folds[0].train) < len(folds[1].train)


def test_test_window_starts_the_week_after_the_origin(splitter, panel):
    fold = next(iter(splitter.split(panel)))
    test = fold.test(3)

    assert test["ds"].min() == fold.origin + WEEK
    assert test["ds"].max() == fold.origin + 3 * WEEK


def test_short_horizon_is_a_prefix_of_the_long_one(splitter, panel):
    fold = next(iter(splitter.split(panel)))

    assert set(fold.test(1)["ds"]) < set(fold.test(3)["ds"])


def test_every_series_gets_exactly_h_test_rows(splitter, panel):
    for fold in splitter.split(panel):
        counts = fold.test(3).groupby("unique_id").size()
        assert counts.eq(3).all(), "missing weeks stay as NaN rows, they are not dropped"


def test_metadata_records_the_fold_spacing(splitter, panel):
    """The DM test needs the spacing to justify the HLN correction."""
    fold = next(iter(splitter.split(panel)))

    assert fold.metadata["step_weeks"] == 2
    assert fold.metadata["folds"] == 2
    assert fold.metadata["horizons"] == [1, 3]
    assert fold.metadata["overlapping_test_windows"] is True


def test_history_too_short_for_the_protocol_is_refused(config_dict, raw_frame):
    config_dict["protocol"]["folds"] = 40
    cfg = Config.from_dict(config_dict)
    panel = normalize_panel(raw_frame, cfg)

    with pytest.raises(ProtocolError, match="history"):
        RollingOriginSplitter(cfg).origins(panel)


def test_series_below_the_minimum_training_window_are_reported(config_dict, raw_frame):
    config_dict["protocol"]["min_train_weeks"] = 200
    cfg = Config.from_dict(config_dict)
    panel = normalize_panel(raw_frame, cfg)
    splitter = RollingOriginSplitter(cfg)

    report = splitter.eligibility(panel)

    assert (~report["meets_min_train"]).all()
    assert report["train_weeks_at_earliest_origin"].min() < 200


def test_assertion_fails_loudly_and_names_the_minimum(config_dict, raw_frame):
    config_dict["protocol"]["min_train_weeks"] = 200
    cfg = Config.from_dict(config_dict)
    panel = normalize_panel(raw_frame, cfg)

    with pytest.raises(ProtocolError, match="minimum found"):
        RollingOriginSplitter(cfg).assert_protocol_feasible(panel)


def test_feasible_protocol_passes_the_assertion(splitter, panel):
    splitter.assert_protocol_feasible(panel)


def test_eligibility_counts_observed_weeks_not_grid_rows(cfg, raw_frame):
    """A week present in the export but carrying a null measure is unknown,
    so it is not a usable training week."""
    import numpy as np

    df = raw_frame.copy()
    mask = ((df["internal_id"] == "SUB_A")
            & df["Time_Period_End_Date"].isin(["2022-05-15", "2022-05-22"]))
    df.loc[mask, "Dollars"] = np.nan
    panel = normalize_panel(df, cfg)

    report = RollingOriginSplitter(cfg).eligibility(panel).set_index("unique_id")
    nulled = report.loc["SUB_A@@BOSTON, MA - MULO", "train_weeks_at_earliest_origin"]
    clean = report.loc["SUB_C@@CHICAGO, IL - MULO", "train_weeks_at_earliest_origin"]

    assert nulled == clean - 2


def test_series_ending_before_the_evaluation_span_are_flagged(raw_frame, cfg):
    from tests.conftest import raw_rows, weeks

    early = raw_rows("SUB_E", "MIAMI, FL - MULO", weeks("2022-01-09", 30), range(1, 31))
    panel = normalize_panel(pd.concat([raw_frame, early], ignore_index=True), cfg)
    report = RollingOriginSplitter(cfg).eligibility(panel).set_index("unique_id")

    assert not report.loc["SUB_E@@MIAMI, FL - MULO", "covers_evaluation_span"]
    assert report.loc["SUB_A@@BOSTON, MA - MULO", "covers_evaluation_span"]
