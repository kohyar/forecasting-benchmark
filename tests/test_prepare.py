"""Gap handling.

Missing weeks are a protocol decision, not a per-adapter one: every model must
see the same training frame or the comparison is between imputations rather
than between models. Actuals used for scoring are never imputed.
"""
import numpy as np
import pandas as pd
import pytest

from tsbench.data.prepare import impute_gaps


def frame(uid, values, start="2022-01-09"):
    return pd.DataFrame({
        "unique_id": uid,
        "ds": pd.date_range(start, periods=len(values), freq="7D"),
        "y": np.asarray(values, dtype=float),
    })


def test_interior_gaps_are_interpolated():
    df = frame("A", [1, 2, np.nan, 4, 5])

    out, report = impute_gaps(df)

    assert out["y"].tolist() == [1, 2, 3, 4, 5]
    assert report["n_imputed"] == 1
    assert report["n_series_imputed"] == 1


def test_a_run_of_missing_weeks_is_bridged():
    df = frame("A", [10, np.nan, np.nan, np.nan, 50])

    out, _ = impute_gaps(df)

    assert out["y"].tolist() == [10, 20, 30, 40, 50]


def test_leading_gaps_are_left_alone():
    """Nothing to interpolate between - filling them would invent history."""
    df = frame("A", [np.nan, 2, 3])

    out, report = impute_gaps(df)

    assert np.isnan(out["y"].iloc[0])
    assert report["n_imputed"] == 0


def test_a_trailing_gap_is_carried_forward():
    """The training frame must reach the origin: a model anchored on an
    earlier week would forecast the wrong weeks entirely."""
    df = frame("A", [1, 2, 3, np.nan, np.nan])

    out, report = impute_gaps(df)

    assert out["y"].tolist() == [1, 2, 3, 3, 3]
    assert report["n_carried_forward"] == 2
    assert report["n_imputed"] == 0, "carrying forward is not interpolation"


def test_every_series_ends_on_a_value(raw_frame):
    """The invariant the runner depends on."""
    import pandas as pd

    df = pd.concat([frame("A", [1, 2, np.nan]), frame("B", [5, np.nan, np.nan])],
                   ignore_index=True)

    out, _ = impute_gaps(df)
    last = out.groupby("unique_id")["y"].last()

    assert last.notna().all()


def test_series_are_imputed_independently():
    df = pd.concat([frame("A", [1, np.nan, 3]), frame("B", [100, 200, 300])],
                   ignore_index=True)

    out, report = impute_gaps(df)

    assert out.set_index(["unique_id", "ds"])["y"].loc["A"].tolist() == [1, 2, 3]
    assert report["n_series_imputed"] == 1


def test_the_input_frame_is_not_mutated():
    df = frame("A", [1, np.nan, 3])
    before = df["y"].isna().sum()

    impute_gaps(df)

    assert df["y"].isna().sum() == before


def test_zeros_are_not_treated_as_missing():
    df = frame("A", [1, 0, 3])

    out, report = impute_gaps(df)

    assert out["y"].tolist() == [1, 0, 3]
    assert report["n_imputed"] == 0


def test_report_names_the_method_for_the_methods_section():
    df = frame("A", [1, np.nan, 3])

    _, report = impute_gaps(df)

    assert report["method"] == "linear"
    assert report["n_rows"] == 3
