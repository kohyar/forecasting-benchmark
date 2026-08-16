"""Normalising the SPINS export into a long panel."""
import numpy as np
import pandas as pd
import pytest

from tsbench.config import Config
from tsbench.data.loader import normalize_panel
from tests.conftest import raw_rows, weeks


@pytest.fixture
def cfg(config_dict):
    return Config.from_dict(config_dict)


@pytest.fixture
def panel(raw_frame, cfg):
    return normalize_panel(raw_frame, cfg)


def test_emits_canonical_columns(panel):
    assert {"unique_id", "ds", "y"} <= set(panel.columns)
    assert panel["ds"].dtype.kind == "M"
    assert panel["y"].dtype.kind == "f"


def test_scope_drops_census_region_and_other_channels(panel):
    assert panel["geography_level"].unique().tolist() == ["MARKET"]
    assert panel["channel"].unique().tolist() == ["CONVENTIONAL|MULTI OUTLET"]
    assert panel["unique_id"].nunique() == 3


def test_unique_id_pairs_product_with_geography(panel):
    assert "SUB_A@@BOSTON, MA - MULO" in set(panel["unique_id"])


def test_target_column_from_config_becomes_y(raw_frame, config_dict):
    config_dict["data"]["target"] = "Units"
    panel = normalize_panel(raw_frame, Config.from_dict(config_dict))
    a = panel[panel["unique_id"] == "SUB_A@@BOSTON, MA - MULO"].sort_values("ds")
    raw_a = raw_frame[(raw_frame["internal_id"] == "SUB_A")
                      & (raw_frame["Geography"] == "BOSTON, MA - MULO")].sort_values("Time_Period_End_Date")

    np.testing.assert_allclose(a["y"].to_numpy(), raw_a["Units"].to_numpy())


def test_internal_gaps_become_explicit_missing_rows(panel):
    b = panel[panel["unique_id"] == "SUB_B@@ALBANY, NY - MULO"].sort_values("ds")

    assert len(b) == 60, "series is reindexed onto the complete weekly grid"
    assert b["y"].isna().sum() == 2
    assert b["ds"].diff().dropna().eq(pd.Timedelta(days=7)).all()


def test_grid_does_not_extend_past_a_series_own_history(raw_frame, cfg):
    """A series that starts late or ends early is not padded to the panel span."""
    short = raw_rows("SUB_D", "DENVER, CO - MULO", weeks("2022-03-06", 10), np.arange(1.0, 11.0))
    panel = normalize_panel(pd.concat([raw_frame, short], ignore_index=True), cfg)
    d = panel[panel["unique_id"] == "SUB_D@@DENVER, CO - MULO"]

    assert len(d) == 10
    assert d["ds"].min() == pd.Timestamp("2022-03-06")
    assert d["ds"].max() == pd.Timestamp("2022-05-08")


def test_null_strings_become_nan(raw_frame, cfg):
    raw_frame["Dollars"] = raw_frame["Dollars"].astype(object)
    raw_frame.loc[3, "Dollars"] = "null"
    panel = normalize_panel(raw_frame, cfg)

    assert panel["y"].isna().sum() >= 1


def test_zero_targets_are_preserved_not_dropped(panel):
    b = panel[panel["unique_id"] == "SUB_B@@ALBANY, NY - MULO"]

    assert (b["y"] == 0).sum() == 3


def test_panel_is_sorted_by_series_then_time(panel):
    assert panel.equals(panel.sort_values(["unique_id", "ds"]).reset_index(drop=True))


def test_covariates_are_carried_through(panel):
    for col in ("ARP", "Units", "Dollars_Yago", "Units_Yago"):
        assert col in panel.columns


def test_statics_are_constant_within_a_series(panel):
    for col in ("Department", "Category", "Subcategory", "region", "channel"):
        assert panel.groupby("unique_id")[col].nunique().max() == 1


def test_rejects_a_non_weekly_source(raw_frame, cfg):
    from tsbench.data.schema import FrequencyError

    monthly = raw_frame.copy()
    monthly["Time_Period_End_Date"] = pd.date_range(
        "2022-01-31", periods=len(monthly), freq="ME").strftime("%Y-%m-%d")

    with pytest.raises(FrequencyError):
        normalize_panel(monthly, cfg)


def test_duplicate_series_timestamps_are_rejected(raw_frame, cfg):
    from tsbench.data.schema import SchemaError

    dupes = pd.concat([raw_frame, raw_frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(SchemaError, match="duplicate"):
        normalize_panel(dupes, cfg)
