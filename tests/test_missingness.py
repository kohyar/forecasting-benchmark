"""Two kinds of missing week, handled differently.

A week the export omits is a week with no sales - an observed zero, and a real
target to score against. A week that is present but carries a null measure is
genuinely unknown: it is not scored, and only the training frame interpolates
it.
"""
import numpy as np
import pandas as pd
import pytest

from tsbench.config import Config
from tsbench.data.loader import normalize_panel
from tsbench.data.prepare import impute_gaps
from tests.conftest import raw_rows, weeks


@pytest.fixture
def cfg(config_dict):
    return Config.from_dict(config_dict)


@pytest.fixture
def mixed(raw_frame):
    """Series B already has two omitted weeks; give it a present-but-null week."""
    df = raw_frame.copy()
    mask = (df["internal_id"] == "SUB_B") & (df["Time_Period_End_Date"] == "2022-05-15")
    assert mask.any()
    df.loc[mask, "Dollars"] = np.nan
    return df


def test_omitted_weeks_become_observed_zeros(mixed, cfg):
    panel = normalize_panel(mixed, cfg)
    b = panel[panel["unique_id"] == "SUB_B@@ALBANY, NY - MULO"].sort_values("ds")

    omitted = b[~b["row_present"]]

    assert len(omitted) == 2
    assert (omitted["y"] == 0).all()


def test_present_but_null_weeks_stay_unknown(mixed, cfg):
    panel = normalize_panel(mixed, cfg)
    b = panel[panel["unique_id"] == "SUB_B@@ALBANY, NY - MULO"]

    reported_null = b[b["row_present"] & b["y"].isna()]

    assert len(reported_null) == 1


def test_the_two_kinds_are_distinguishable(mixed, cfg):
    panel = normalize_panel(mixed, cfg)

    assert "row_present" in panel.columns
    assert panel["row_present"].dtype == bool


def test_the_policy_can_be_switched_off_for_a_sensitivity_check(mixed, config_dict):
    config_dict["data"]["absent_rows"] = "missing"
    panel = normalize_panel(mixed, Config.from_dict(config_dict))
    b = panel[panel["unique_id"] == "SUB_B@@ALBANY, NY - MULO"]

    assert b[~b["row_present"]]["y"].isna().all()


def test_an_unknown_policy_is_refused(mixed, config_dict):
    config_dict["data"]["absent_rows"] = "guess"

    with pytest.raises(ValueError, match="absent_rows"):
        normalize_panel(mixed, Config.from_dict(config_dict))


def test_only_the_training_frame_interpolates_the_unknown_weeks(mixed, cfg):
    panel = normalize_panel(mixed, cfg)
    train, report = impute_gaps(panel)

    assert report["n_imputed"] == 1
    assert train["y"].notna().all()
    assert panel["y"].isna().sum() == 1, "the panel keeps the truth for scoring"


def test_structural_zeros_are_never_interpolated_away(mixed, cfg):
    panel = normalize_panel(mixed, cfg)
    train, _ = impute_gaps(panel)
    b = train[train["unique_id"] == "SUB_B@@ALBANY, NY - MULO"]

    assert (b[~b["row_present"]]["y"] == 0).all()


def test_zero_sales_weeks_count_toward_intermittency(mixed, cfg):
    from tsbench.data.sampling import series_profile

    panel = normalize_panel(mixed, cfg)
    prof = series_profile(panel).set_index("unique_id")
    b = prof.loc["SUB_B@@ALBANY, NY - MULO"]

    assert b["n_zero"] == 5, "three reported zeros plus two zero-sales weeks"
    assert b["zero_share"] == pytest.approx(5 / 59)
