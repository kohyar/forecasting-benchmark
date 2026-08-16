"""Stratified series sampling, frozen to disk so every run uses one sample."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tsbench.config import Config

_META_PREFIX = "#tsbench "
_DATE_COLS = ["first_ds", "last_ds"]


def series_profile(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-series characteristics used for stratification and eligibility."""
    g = panel.groupby("unique_id", sort=True)
    prof = pd.DataFrame({
        "n_obs": g["y"].count(),
        "n_grid": g["y"].size(),
        "total_volume": g["y"].sum(min_count=1),
        "n_zero": g["y"].apply(lambda s: int((s == 0).sum())),
        "first_ds": g["ds"].min(),
        "last_ds": g["ds"].max(),
    })
    prof["n_missing"] = prof["n_grid"] - prof["n_obs"]
    prof["zero_share"] = prof["n_zero"] / prof["n_obs"].where(prof["n_obs"] > 0)
    return prof.reset_index()


def label_strata(profile: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Eligible series labelled by volume decile x zero-share bin."""
    s = cfg.sampling
    elig = profile[profile["n_obs"] >= s.min_observed_weeks].copy()
    if elig.empty:
        raise ValueError(f"no eligible series: none has >= {s.min_observed_weeks} observed weeks")

    ranks = elig["total_volume"].rank(method="first")
    elig["volume_decile"] = pd.qcut(ranks, q=s.volume_deciles, labels=False).astype(int)
    elig["zero_bin"] = pd.cut(elig["zero_share"], bins=s.zero_share_bins, labels=False,
                              include_lowest=True, right=False).astype(int)
    elig["stratum"] = (elig["volume_decile"].astype(str) + "|" + elig["zero_bin"].astype(str))
    return elig.sort_values("unique_id").reset_index(drop=True)


def stratified_sample(profile: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Proportional allocation with largest-remainder rounding, then a seeded
    draw inside each stratum. Deterministic given (profile, config, seed)."""
    n = cfg.sampling.n_series
    elig = label_strata(profile, cfg)
    if len(elig) < n:
        raise ValueError(f"requested {n} series but only {len(elig)} are eligible")

    sizes = elig["stratum"].value_counts().sort_index()
    quota = _largest_remainder(sizes / sizes.sum(), n, cap=sizes)

    rng = np.random.default_rng(cfg.run.seed)
    picks = []
    for stratum in sizes.index:
        k = int(quota[stratum])
        if k == 0:
            continue
        pool = elig[elig["stratum"] == stratum]
        idx = rng.choice(pool.index.to_numpy(), size=k, replace=False)
        picks.append(pool.loc[np.sort(idx)])

    return pd.concat(picks).sort_values("unique_id").reset_index(drop=True)


def _largest_remainder(shares: pd.Series, n: int, cap: pd.Series) -> pd.Series:
    exact = shares * n
    quota = np.floor(exact).astype(int).clip(upper=cap)
    remainder = (exact - np.floor(exact)).sort_values(ascending=False)

    for stratum in remainder.index:
        if quota.sum() >= n:
            break
        if quota[stratum] < cap[stratum]:
            quota[stratum] += 1
    # strata that hit their cap push the shortfall onto whoever still has room
    for stratum in cap.sort_values(ascending=False).index:
        while quota.sum() < n and quota[stratum] < cap[stratum]:
            quota[stratum] += 1
    return quota


def save_sample(sample: pd.DataFrame, path, cfg: Config) -> None:
    """CSV with a provenance header - the anchor every later run reads."""
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": cfg.run.seed,
        "config_hash": cfg.hash,
        "n_series": len(sample),
        "target": cfg.data.target,
        "scope": {"geography_level": cfg.data.scope.geography_level,
                  "channels": cfg.data.scope.channels},
        "volume_deciles": cfg.sampling.volume_deciles,
        "zero_share_bins": cfg.sampling.zero_share_bins,
        "min_observed_weeks": cfg.sampling.min_observed_weeks,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(_META_PREFIX + json.dumps(meta) + "\n")
        sample.to_csv(fh, index=False)


def load_sample(path, expect: Config | None = None):
    path = Path(path)
    with open(path) as fh:
        first = fh.readline()
        if not first.startswith(_META_PREFIX):
            raise ValueError(f"{path} has no tsbench provenance header")
        meta = json.loads(first[len(_META_PREFIX):])
        sample = pd.read_csv(fh, parse_dates=_DATE_COLS)

    if expect is not None and meta["config_hash"] != expect.hash:
        raise ValueError(
            f"sample config_hash {meta['config_hash']} does not match the current "
            f"config {expect.hash}: regenerate the sample or restore the config")
    return sample, meta
