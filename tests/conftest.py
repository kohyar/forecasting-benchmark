import numpy as np
import pandas as pd
import pytest

WEEK = pd.Timedelta(days=7)


def weeks(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="7D")


def raw_rows(internal_id, geography, dates, dollars, units=None, arp=None,
             geo_level="MARKET", channel="CONVENTIONAL|MULTI OUTLET",
             department="PRODUCE", category="GRAPES", subcategory="OTHER GRAPES",
             region="BOSTON, MA "):
    """Build rows shaped like the SPINS export, including the _Yago columns."""
    n = len(dates)
    dollars = np.asarray(dollars, dtype=float)
    units = np.asarray(units if units is not None else dollars / 5.0, dtype=float)
    arp = np.asarray(arp if arp is not None else dollars / np.where(units == 0, np.nan, units), dtype=float)
    df = pd.DataFrame({
        "internal_id": internal_id,
        "Geography": geography,
        "Time_Period_End_Date": [d.strftime("%Y-%m-%d") for d in dates],
        "Time_Period": "WEEK",
        "Channel/Outlet": channel,
        "Geography_Level": geo_level,
        "Region": region,
        "State": "null",
        "County": "null",
        "Product_Universe": "TPL",
        "Product_Level": "SUBCATEGORY",
        "Department": department,
        "Category": category,
        "Subcategory": subcategory,
        "UPC": "null",
        "Brand": "null",
        "Description": subcategory,
        "Dollars": dollars,
        "Units": units,
        "ARP": arp,
        "ARP_EQ_Units": 0.2,
        "ingested_at": "2026-06-29T14:16:55.263Z",
        "label": "null",
    })
    # _Yago columns are the exact 52-week calendar lag in the real export
    for col in ("Dollars", "Units", "ARP", "ARP_EQ_Units"):
        lag = df[col].shift(52)
        df[f"{col}_Yago"] = lag
    return df[[
        "internal_id", "Geography", "Time_Period_End_Date", "Time_Period", "Channel/Outlet",
        "Geography_Level", "Region", "State", "County", "Product_Universe", "Product_Level",
        "Department", "Category", "Subcategory", "UPC", "Brand", "Description",
        "Dollars", "Dollars_Yago", "Units", "Units_Yago", "ARP", "ARP_Yago",
        "ARP_EQ_Units", "ARP_EQ_Units_Yago", "ingested_at", "label",
    ]]


@pytest.fixture
def config_dict():
    """Small config for unit tests. season_length is deliberately short so
    fixtures stay readable; the shipped config is asserted separately."""
    return {
        "run": {"name": "unit-test", "seed": 42, "n_repeats": 3, "output_dir": "results"},
        "data": {
            "path": "unused.csv",
            "target": "Dollars",
            "season_length": 4,
            "scope": {
                "geography_level": ["MARKET"],
                "channels": ["CONVENTIONAL|MULTI OUTLET"],
            },
        },
        "sampling": {
            "n_series": 4,
            "volume_deciles": 2,
            "zero_share_bins": [0.0, 0.05, 1.01],
            "min_observed_weeks": 8,
            "sample_path": "data/sample_series.csv",
        },
        "protocol": {"folds": 2, "step": 2, "horizons": [1, 3], "min_train_weeks": 8},
        "tuning": {"budget_trials": 20},
        "metrics": {
            "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "coverage_levels": [0.8],
            "mase_denominator_window": "earliest_origin",
        },
        "models": {"enabled": ["naive", "seasonal_naive"]},
    }


@pytest.fixture
def raw_frame():
    """Three in-scope series plus two out-of-scope rows.

    - series A: 60 contiguous weeks, high volume
    - series B: 60 weeks with an internal 2-week gap, low volume, some zeros
    - series C: 60 weeks, mid volume
    - one CENSUS REGION row and one FOOD-channel row that scope must drop
    """
    ds = weeks("2022-01-09", 60)
    rng = np.random.default_rng(0)

    a = raw_rows("SUB_A", "BOSTON, MA - MULO", ds, 1000 + 50 * np.sin(np.arange(60)) + rng.normal(0, 5, 60))

    ds_b = ds.delete([20, 21])
    vals_b = np.full(len(ds_b), 3.0)
    vals_b[[5, 9, 14]] = 0.0
    b = raw_rows("SUB_B", "ALBANY, NY - MULO", ds_b, vals_b)

    c = raw_rows("SUB_C", "CHICAGO, IL - MULO", ds, 200 + np.arange(60) * 1.5)

    census = raw_rows("SUB_A", "MIDWEST - US CENSUS REGION - MULO", ds[:5], [9] * 5,
                      geo_level="CENSUS REGION")
    food = raw_rows("SUB_A", "BOSTON, MA - FOOD", ds[:5], [7] * 5,
                    channel="CONVENTIONAL|FOOD")

    return pd.concat([a, b, c, census, food], ignore_index=True)
