"""Column contract and frequency assertion for the SPINS weekly export."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


class SchemaError(ValueError):
    pass


class FrequencyError(ValueError):
    pass


RAW_REQUIRED = [
    "internal_id", "Geography", "Time_Period_End_Date", "Time_Period", "Channel/Outlet",
    "Geography_Level", "Region", "Department", "Category", "Subcategory",
    "Dollars", "Dollars_Yago", "Units", "Units_Yago", "ARP", "ARP_Yago",
]

# Contemporaneous measures: observable only after the week closes.
TIME_VARYING = ["Dollars", "Units", "ARP", "ARP_EQ_Units"]

# Verified against the export: *_Yago is the exact 52-week calendar lag
# (99.9995% match), so it is known at any origin for horizons <= 52.
YAGO = [f"{c}_Yago" for c in TIME_VARYING]
YAGO_LAG_WEEKS = 52

# Realised average price. Future values are unknowable and, with Dollars as the
# target, are a factor of it - so what a model may see for the forecast window
# is the last price observed at the origin, carried forward.
PRICE = "ARP"
PLANNED_PRICE = f"{PRICE}_planned"

STATIC = ["Department", "Category", "Subcategory", "region", "channel", "geography_level"]

NUMERIC = TIME_VARYING + YAGO

_ANCHOR = {0: "MON", 1: "TUE", 2: "WED", 3: "THU", 4: "FRI", 5: "SAT", 6: "SUN"}


@dataclass(frozen=True)
class CovariateGroups:
    static: list
    past_observed: list
    known_future: list


def validate_raw_columns(df: pd.DataFrame) -> None:
    missing = [c for c in RAW_REQUIRED if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required column(s): {', '.join(missing)}")


def assert_weekly(ds) -> str:
    """Return the weekly anchor (e.g. W-SUN) or raise. Whole-week gaps are fine;
    anything else means the seasonal period of 52 does not hold."""
    u = pd.DatetimeIndex(pd.to_datetime(pd.Series(ds)).dropna().unique()).sort_values()
    if len(u) < 2:
        raise FrequencyError("inferred frequency is not weekly: fewer than two distinct dates")

    problems = []
    weekdays = sorted(set(u.dayofweek))
    if len(weekdays) > 1:
        names = ", ".join(_ANCHOR[w] for w in weekdays)
        problems.append(f"dates fall on {len(weekdays)} distinct weekdays ({names})")

    gaps = np.diff(u.values).astype("timedelta64[D]").astype(int)
    off_grid = sorted({int(g) for g in gaps if g % 7 != 0})
    if off_grid:
        problems.append(f"gaps of {off_grid[:5]} days are not whole weeks")

    if problems:
        raise FrequencyError("inferred frequency is not weekly: " + "; ".join(problems))
    return f"W-{_ANCHOR[weekdays[0]]}"


def covariate_groups(target: str) -> CovariateGroups:
    """A variable is known_future only if genuinely available in advance.

    ARP is realised average price (Dollars/Units), not a posted price plan, so
    it is past_observed. The _Yago lags are known in advance for h <= 52.
    """
    if target not in TIME_VARYING:
        raise SchemaError(f"unsupported target {target!r}; expected one of {TIME_VARYING}")
    return CovariateGroups(
        static=list(STATIC),
        past_observed=[c for c in TIME_VARYING if c != target],
        known_future=list(YAGO) + [PLANNED_PRICE],
    )
