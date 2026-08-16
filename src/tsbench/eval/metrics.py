"""Forecast accuracy metrics.

Scaled metrics divide by the in-sample seasonal naive error. That denominator
is computed once per series, from training data only, and handed to every
model and fold - recomputing it per model is a quiet source of inconsistent
numbers. Metrics that are undefined near zero report how many points they
dropped rather than dropping them silently.
"""
import numpy as np
import pandas as pd

from tsbench.config import Config

WEEK = pd.Timedelta(days=7)

POINT_METRICS = ["MASE", "RMSSE", "MAE", "RMSE", "sMAPE", "MAPE"]


def seasonal_denominators(train: pd.DataFrame, season_length: int) -> pd.DataFrame:
    """Per-series MASE and RMSSE denominators from in-sample seasonal differences.

    Pairs where either endpoint is missing are skipped; n_pairs records how
    many survived so thin denominators are visible in the results.
    """
    if season_length < 1:
        raise ValueError(f"season_length must be >= 1, got {season_length}")

    out = []
    for uid, g in train.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=True):
        y = g["y"].to_numpy(dtype=float)
        diffs = np.abs(y[season_length:] - y[:-season_length]) if len(y) > season_length \
            else np.array([])
        diffs = diffs[~np.isnan(diffs)]
        n = len(diffs)
        mase = float(diffs.mean()) if n else np.nan
        rmsse = float(np.sqrt((diffs ** 2).mean())) if n else np.nan
        out.append({
            "unique_id": uid,
            "mase_denom": mase,
            "rmsse_denom": rmsse,
            "n_pairs": n,
            "zero_denominator": bool(n and mase == 0),
            "season_length": season_length,
        })
    return pd.DataFrame(out)


def denominators_for_protocol(panel: pd.DataFrame, cfg: Config, splitter) -> pd.DataFrame:
    """One denominator per series for the whole run.

    Taken from data strictly before the earliest origin, so it is in-sample for
    every fold and identical across folds and models.
    """
    window = cfg.metrics.mase_denominator_window
    if window != "earliest_origin":
        raise ValueError(f"unsupported mase_denominator_window: {window!r}")

    earliest = splitter.origins(panel)[0]
    train = panel[panel["ds"] <= earliest]
    return seasonal_denominators(train, cfg.data.season_length)


def evaluate(actual: pd.DataFrame, forecast: pd.DataFrame, denominators: pd.DataFrame,
             model: str, fold: int, horizon: int,
             quantile_levels=None, coverage_levels=(0.8,)) -> pd.DataFrame:
    """Long-format metric rows: one per (model, fold, horizon, series, metric)."""
    missing = set(actual["unique_id"]) - set(forecast["unique_id"])
    if missing:
        raise ValueError(f"missing forecast for {len(missing)} series: {sorted(missing)[:5]}")

    df = actual.merge(forecast, on=["unique_id", "ds"], how="left", validate="one_to_one")
    df = df.merge(denominators, on="unique_id", how="left", validate="many_to_one")

    q_cols = _quantile_columns(forecast, quantile_levels)
    rows = []
    for uid, g in df.groupby("unique_id", sort=True):
        rows.extend(_score_series(g, uid, q_cols, coverage_levels))

    out = pd.DataFrame(rows)
    out.insert(0, "horizon", horizon)
    out.insert(0, "fold", fold)
    out.insert(0, "model", model)
    return out[["model", "fold", "horizon", "unique_id", "metric", "value",
                "n_points", "n_undefined"]]


def _quantile_columns(forecast: pd.DataFrame, levels) -> dict:
    found = {}
    for col in forecast.columns:
        if col.startswith("yhat_q"):
            level = int(col[len("yhat_q"):]) / 100
            if levels is None or level in levels:
                found[level] = col
    return dict(sorted(found.items()))


def _score_series(g: pd.DataFrame, uid: str, q_cols: dict, coverage_levels) -> list:
    scored = g[g["y"].notna() & g["yhat"].notna()]
    y = scored["y"].to_numpy(dtype=float)
    yhat = scored["yhat"].to_numpy(dtype=float)
    n = len(y)
    n_dropped = len(g) - n

    def row(metric, value, points=n, undefined=n_dropped):
        return {"unique_id": uid, "metric": metric, "value": value,
                "n_points": points, "n_undefined": undefined}

    if n == 0:
        return [row(m, np.nan, 0) for m in POINT_METRICS]

    err = y - yhat
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    mase_den = scored["mase_denom"].iloc[0]
    rmsse_den = scored["rmsse_denom"].iloc[0]

    out = [
        row("MASE", _scale(mae, mase_den)),
        row("RMSSE", _scale(rmse, rmsse_den)),
        row("MAE", mae),
        row("RMSE", rmse),
    ]

    denom = np.abs(y) + np.abs(yhat)
    ok = denom > 0
    out.append(row("sMAPE",
                   float((200.0 * np.abs(err[ok]) / denom[ok]).mean()) if ok.any() else np.nan,
                   points=int(ok.sum()), undefined=n_dropped + int((~ok).sum())))

    nz = y != 0
    out.append(row("MAPE",
                   float((100.0 * np.abs(err[nz] / y[nz])).mean()) if nz.any() else np.nan,
                   points=int(nz.sum()), undefined=n_dropped + int((~nz).sum())))

    if q_cols:
        out.extend(_score_quantiles(scored, y, q_cols, coverage_levels, row))
    return out


def _score_quantiles(scored, y, q_cols, coverage_levels, row) -> list:
    losses = []
    for level, col in q_cols.items():
        pred = scored[col].to_numpy(dtype=float)
        diff = y - pred
        losses.append(np.maximum(level * diff, (level - 1) * diff))
    pinball = float(np.mean([loss.mean() for loss in losses]))

    out = [row("pinball", pinball)]
    # CRPS = 2 * integral of pinball over the quantile levels; on a symmetric
    # grid this reduces to twice the mean pinball, and collapses to MAE when
    # the predictive distribution is degenerate.
    levels = np.array(list(q_cols))
    if np.isclose(levels.mean(), 0.5):
        out.append(row("CRPS", 2.0 * pinball))

    for c in coverage_levels or []:
        lo, hi = round((1 - c) / 2, 4), round(1 - (1 - c) / 2, 4)
        if lo in q_cols and hi in q_cols:
            inside = ((y >= scored[q_cols[lo]].to_numpy())
                      & (y <= scored[q_cols[hi]].to_numpy()))
            out.append(row(f"coverage_{int(c * 100)}", float(inside.mean())))
    return out


def _scale(value: float, denominator: float) -> float:
    if denominator is None or np.isnan(denominator) or denominator == 0:
        return np.nan
    return float(value / denominator)
