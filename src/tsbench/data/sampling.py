"""Stratified series sampling, frozen to disk so every run uses one sample."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tsbench.config import Config

_META_PREFIX = "#tsbench "
_DATE_COLS = ["first_ds", "last_ds"]


def _canonical_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Pin datetime resolution so the frozen sample round-trips exactly;
    pandas infers seconds or nanoseconds depending on how a frame was built."""
    for col in _DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
    return df


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
    return _canonical_dates(elig.sort_values("unique_id").reset_index(drop=True))


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


def nested_ladder(pool: pd.DataFrame, anchor: pd.DataFrame, sizes, seed: int) -> dict:
    """Nested stratified samples for a cardinality curve: S_k is a subset of
    S_k+1 for every adjacent pair, and the rung at len(anchor) is the anchor.

    Nesting and exact stratum proportionality cannot both hold - a rung
    inherits everything below it, so a stratum the anchor over-represents
    stays over-represented. Nesting wins, because a curve whose points do not
    share series measures the draw as much as the scaling.
    """
    sizes = sorted(int(s) for s in sizes)
    anchor_n = len(anchor)
    if anchor_n not in sizes:
        raise ValueError(f"the anchor's size {anchor_n} must be one of the rungs {sizes}")
    missing = set(anchor["unique_id"]) - set(pool["unique_id"])
    if missing:
        raise ValueError(f"{len(missing)} anchor series are outside the pool; "
                         "the ladder cannot be anchored on it")
    if sizes[-1] > len(pool):
        raise ValueError(f"largest rung {sizes[-1]} exceeds the pool ({len(pool)})")

    pool = pool.sort_values("unique_id").reset_index(drop=True)
    strata = pool["stratum"].value_counts().sort_index()
    rungs = {anchor_n: anchor.sort_values("unique_id").reset_index(drop=True)}

    # Down from the anchor: thin the rung above, so each stays a subset of it.
    for n in [s for s in sizes if s < anchor_n][::-1]:
        parent = rungs[min(s for s in rungs if s > n)]
        held = parent["stratum"].value_counts().reindex(strata.index, fill_value=0)
        quota = _largest_remainder(strata / strata.sum(), n, cap=held)
        rng = np.random.default_rng([seed, n])
        keep = []
        for stratum in strata.index:
            k = int(quota[stratum])
            if k == 0:
                continue
            members = parent.index[parent["stratum"] == stratum].to_numpy()
            keep.append(parent.loc[np.sort(rng.choice(members, size=k, replace=False))])
        rungs[n] = pd.concat(keep).sort_values("unique_id").reset_index(drop=True)

    # Up from the anchor: keep the rung below whole, draw the shortfall from
    # whatever that rung has not already taken.
    for n in [s for s in sizes if s > anchor_n]:
        parent = rungs[max(s for s in rungs if s < n)]
        taken = set(parent["unique_id"])
        held = parent["stratum"].value_counts().reindex(strata.index, fill_value=0)
        free = pool[~pool["unique_id"].isin(taken)]
        available = free["stratum"].value_counts().reindex(strata.index, fill_value=0)

        ideal = _largest_remainder(strata / strata.sum(), n, cap=strata)
        need = (ideal - held).clip(lower=0).clip(upper=available)
        need = _rebalance(need, available, n - len(parent))

        rng = np.random.default_rng([seed, n])
        add = []
        for stratum in strata.index:
            k = int(need[stratum])
            if k == 0:
                continue
            members = free.index[free["stratum"] == stratum].to_numpy()
            add.append(free.loc[np.sort(rng.choice(members, size=k, replace=False))])
        grown = pd.concat([parent, *add]) if add else parent
        rungs[n] = grown.sort_values("unique_id").reset_index(drop=True)

    return {n: rungs[n] for n in sizes}


def _rebalance(need: pd.Series, available: pd.Series, target: int) -> pd.Series:
    """Make the per-stratum draw sum to exactly `target`, respecting supply.

    The ideal quota is what proportionality asks for; inherited series may have
    already overshot it in some strata and undershot it in others, so the
    shortfall is settled against whoever still has room.
    """
    need = need.copy()
    while need.sum() > target:                     # trim the largest asks first
        order = need[need > 0].sort_values(ascending=False).index
        if not len(order):
            break
        need[order[0]] -= 1
    room = (available - need).clip(lower=0)
    while need.sum() < target:                     # spend the rest where supply is
        order = room[room > 0].sort_values(ascending=False).index
        if not len(order):
            raise ValueError(f"cannot reach {target}: the pool is exhausted")
        need[order[0]] += 1
        room[order[0]] -= 1
    return need


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
        "sample_key": cfg.sample_key,
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
        sample = _canonical_dates(pd.read_csv(fh, parse_dates=_DATE_COLS))

    if expect is not None:
        # Older files carry only the full config hash; treat that as the key.
        found = meta.get("sample_key", meta.get("config_hash"))
        if found != expect.sample_key:
            raise ValueError(
                f"sample config_hash/sample_key {found} does not match the current "
                f"config {expect.sample_key}: regenerate the sample or restore the "
                f"data, sampling or protocol settings")
    return sample, meta
