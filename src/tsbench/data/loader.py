"""Load the SPINS export into a long panel: unique_id, ds, y, covariates, statics."""
import numpy as np
import pandas as pd

from tsbench.config import Config
from tsbench.data.schema import (
    NUMERIC,
    SchemaError,
    assert_weekly,
    validate_raw_columns,
)

ID_SEP = "@@"
WEEK = pd.Timedelta(days=7)

_RENAME = {"Region": "region", "Channel/Outlet": "channel", "Geography_Level": "geography_level"}


def normalize_panel(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    validate_raw_columns(raw)
    df = raw.copy()

    df = df.rename(columns=_RENAME)
    scope = cfg.data.scope
    df = df[df["geography_level"].isin(scope.geography_level) & df["channel"].isin(scope.channels)]
    if df.empty:
        raise SchemaError(
            f"no rows in scope: geography_level={scope.geography_level}, channels={scope.channels}")

    df["ds"] = pd.to_datetime(df["Time_Period_End_Date"])
    assert_weekly(df["ds"])

    df["unique_id"] = df["internal_id"].astype(str) + ID_SEP + df["Geography"].astype(str)

    dupes = df.duplicated(subset=["unique_id", "ds"]).sum()
    if dupes:
        raise SchemaError(f"{dupes} duplicate (unique_id, ds) row(s) in the source")

    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    target = cfg.data.target
    df["y"] = df[target]

    df["row_present"] = True

    statics = [c for c in ("Department", "Category", "Subcategory", "region", "channel",
                           "geography_level", "internal_id", "Geography") if c in df.columns]
    covariates = [c for c in NUMERIC if c in df.columns]
    keep = ["unique_id", "ds", "y", "row_present"] + covariates + statics

    panel = _reindex_to_weekly_grid(df[keep].sort_values(["unique_id", "ds"]), statics)
    panel["row_present"] = panel["row_present"] == True  # noqa: E712 - NaN -> False
    panel = _apply_absent_row_policy(panel, cfg)
    return panel.reset_index(drop=True)


def _apply_absent_row_policy(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """A week the export omits is a week with no sales, so its target is zero.

    Evidence: sales in the week before a gap run at ~3.5% of the series median,
    gaps cluster in the winter and early-spring weeks, and the export carries
    almost no explicit zeros - it drops the row instead.
    """
    policy = cfg.data.absent_rows
    if policy == "missing":
        return panel
    if policy != "zero":
        raise ValueError(
            f"absent_rows must be 'zero' or 'missing', got {policy!r}")

    panel.loc[~panel["row_present"], "y"] = 0.0
    return panel


def _reindex_to_weekly_grid(df: pd.DataFrame, statics: list) -> pd.DataFrame:
    """Expose internal gaps as explicit NaN rows, without padding a series
    beyond its own first and last observation."""
    span = df.groupby("unique_id", sort=True)["ds"].agg(["min", "max"])
    n_weeks = ((span["max"] - span["min"]).dt.days // 7 + 1).to_numpy()

    uid = np.repeat(span.index.to_numpy(), n_weeks)
    offsets = np.concatenate([np.arange(k) for k in n_weeks]) if len(n_weeks) else np.array([])
    start = np.repeat(span["min"].to_numpy(), n_weeks)
    grid = pd.DataFrame({"unique_id": uid, "ds": start + offsets * np.timedelta64(7, "D")})

    out = grid.merge(df, on=["unique_id", "ds"], how="left")
    if statics:
        g = out.groupby("unique_id", sort=False)[statics]
        out[statics] = g.ffill().bfill()
    return out.sort_values(["unique_id", "ds"])


def load_panel(cfg: Config, chunksize: int = 500_000) -> pd.DataFrame:
    """Stream the CSV, keeping only in-scope rows."""
    scope = cfg.data.scope
    parts = []
    for chunk in pd.read_csv(cfg.data.path, chunksize=chunksize, na_values=["null"],
                             low_memory=False):
        chunk = chunk.rename(columns=_RENAME)
        in_scope = (chunk["geography_level"].isin(scope.geography_level)
                    & chunk["channel"].isin(scope.channels))
        if in_scope.any():
            parts.append(chunk[in_scope].rename(columns={v: k for k, v in _RENAME.items()}))
    if not parts:
        raise SchemaError(f"no rows in scope in {cfg.data.path}")
    return normalize_panel(pd.concat(parts, ignore_index=True), cfg)
