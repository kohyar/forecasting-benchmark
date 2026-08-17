"""Training-frame preparation.

Weeks the export omitted are already zeros by the time the panel is built -
those are observed values, not gaps. What is left here is the other kind: a
week present in the export with a null measure, genuinely unknown. Those are
bridged by linear interpolation, once, for every model, because if adapters
each handled them their own way the benchmark would be comparing imputations.
Leading gaps are left alone - there is nothing to interpolate between and
filling them invents history. Trailing gaps are carried forward instead: the
training frame has to reach the forecast origin, because a model anchored on
an earlier week forecasts the wrong weeks entirely. Actuals used for scoring
are never imputed.
"""
import pandas as pd

METHOD = "linear"


def impute_gaps(df: pd.DataFrame, target: str = "y") -> tuple:
    out = df.sort_values(["unique_id", "ds"]).copy()
    before = out[target].isna()

    grouped = out.groupby("unique_id", sort=False)[target]
    out[target] = grouped.transform(
        lambda s: s.interpolate(method=METHOD, limit_area="inside"))
    interpolated = before & out[target].notna()

    # Carry the last known value to the end of the window, never backwards.
    out[target] = out.groupby("unique_id", sort=False)[target].ffill()
    carried = before & ~interpolated & out[target].notna()

    report = {
        "method": METHOD,
        "n_rows": len(out),
        "n_imputed": int(interpolated.sum()),
        "n_series_imputed": int(out.loc[interpolated, "unique_id"].nunique()),
        "n_carried_forward": int(carried.sum()),
        "n_series_carried_forward": int(out.loc[carried, "unique_id"].nunique()),
        "n_still_missing": int(out[target].isna().sum()),
    }
    return out.reset_index(drop=True), report
