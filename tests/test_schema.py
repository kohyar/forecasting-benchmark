"""Schema validation and the weekly-frequency assertion.

Getting the frequency wrong silently is the failure mode this guards against:
every seasonal model and the MASE denominator assume 52 weekly periods.
"""
import pandas as pd
import pytest

from tsbench.data.schema import (
    FrequencyError,
    SchemaError,
    assert_weekly,
    covariate_groups,
    validate_raw_columns,
)
from tests.conftest import raw_rows, weeks


def test_accepts_the_real_export_columns(raw_frame):
    validate_raw_columns(raw_frame)


def test_missing_column_names_the_column(raw_frame):
    broken = raw_frame.drop(columns=["Dollars"])

    with pytest.raises(SchemaError, match="Dollars"):
        validate_raw_columns(broken)


def test_weekly_dates_pass_and_report_the_anchor():
    ds = weeks("2022-01-09", 20)

    assert assert_weekly(pd.Series(ds)) == "W-SUN"


def test_monthly_dates_fail_loudly():
    ds = pd.Series(pd.date_range("2022-01-31", periods=12, freq="ME"))

    with pytest.raises(FrequencyError, match="not weekly"):
        assert_weekly(ds)


def test_mixed_weekday_dates_fail_loudly():
    ds = pd.Series(list(weeks("2022-01-09", 5)) + [pd.Timestamp("2022-02-15")])

    with pytest.raises(FrequencyError, match="weekday"):
        assert_weekly(ds)


def test_gaps_that_are_whole_weeks_are_allowed():
    """Series with missing weeks are still weekly - the grid is intact."""
    ds = pd.Series(weeks("2022-01-09", 20).delete([5, 6]))

    assert assert_weekly(ds) == "W-SUN"


def test_target_is_never_offered_as_its_own_covariate():
    groups = covariate_groups(target="Dollars")

    assert "Dollars" not in groups.past_observed
    assert "Dollars" not in groups.known_future
    assert "Dollars" not in groups.static


def test_yago_columns_are_known_future_and_realised_price_is_not():
    """_Yago is the 52-week calendar lag, so it is known at any origin for h<=52.
    ARP is realised average price - only observable after the fact."""
    groups = covariate_groups(target="Dollars")

    assert "Dollars_Yago" in groups.known_future
    assert "Units_Yago" in groups.known_future
    assert "ARP" in groups.past_observed
    assert "ARP" not in groups.known_future
    assert "Units" in groups.past_observed


def test_statics_are_the_non_time_varying_attributes():
    groups = covariate_groups(target="Dollars")

    assert set(groups.static) >= {"Department", "Category", "Subcategory", "region", "channel"}


def test_swapping_the_target_swaps_the_covariate_roles():
    groups = covariate_groups(target="Units")

    assert "Units" not in groups.past_observed
    assert "Dollars" in groups.past_observed
    assert "Units_Yago" in groups.known_future
