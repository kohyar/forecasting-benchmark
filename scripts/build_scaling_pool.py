"""Profile the eligible universe and freeze it as the pool the scaling ladder draws from.

    python scripts/build_scaling_pool.py [--config configs/default.yaml]

Read-only with respect to the frozen benchmark sample: this never writes
sample_path. It reports how many series the protocol can actually run on and
writes the stratum-labelled pool, which is what the nested cardinality ladder
is built from.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from tsbench.config import Config
from tsbench.data.loader import load_panel
from tsbench.data.sampling import label_strata, load_sample
from tsbench.eval.splitter import RollingOriginSplitter
from tsbench.pipeline import eligible_profile

_META_PREFIX = "#tsbench "


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="data/eligible_pool.csv")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    print(f"config {args.config}  sample_key={cfg.sample_key}  seed={cfg.run.seed}")

    panel = load_panel(cfg)
    print(f"panel: {len(panel):,} rows  {panel['unique_id'].nunique():,} series  "
          f"{panel['ds'].min().date()} .. {panel['ds'].max().date()}")

    splitter = RollingOriginSplitter(cfg)
    origins = splitter.origins(panel)
    print(f"origins: " + ", ".join(str(o.date()) for o in origins))

    profile, elig = eligible_profile(panel, cfg)
    total = elig["series_total"]
    print(f"\neligibility ({total:,} series in the panel):")
    print(f"  below min_train_weeks={cfg.protocol.min_train_weeks}: {elig['dropped_below_min_train']:,}")
    print(f"  do not cover the evaluation span:                {elig['dropped_no_evaluation_coverage']:,}")
    print(f"  zero seasonal denominator (unscoreable):         {elig['dropped_zero_denominator']:,}")
    print(f"  ELIGIBLE:                                        {elig['eligible']:,}"
          f"  ({elig['eligible'] / total:.1%} of the panel)")

    pool = label_strata(profile, cfg)
    print(f"\npool: {len(pool):,} series across {pool['stratum'].nunique()} strata")

    # The ladder can only climb as far as the pool allows, and only in steps the
    # strata can fill proportionally; the smallest stratum sets the floor.
    sizes = pool["stratum"].value_counts().sort_index()
    print(f"  stratum sizes: min {sizes.min()}, median {int(sizes.median())}, max {sizes.max()}")
    print(f"  ceiling for a proportional ladder: {len(pool):,} series")

    per_decile = pool["volume_decile"].value_counts().sort_index()
    print("  per volume decile: " + ", ".join(f"{d}:{n}" for d, n in per_decile.items()))
    print(f"  intermittent (zero share >= 0.05): "
          f"{int((pool['zero_share'] >= 0.05).sum()):,}")

    # How much of the pool the frozen 1,000-series sample already covers - the
    # ladder is anchored on it, so its strata have to be a subset of the pool's.
    frozen_path = Path(cfg.sampling.sample_path)
    anchor = None
    if frozen_path.exists():
        frozen, _ = load_sample(frozen_path)
        anchor = set(frozen["unique_id"])
        missing = anchor - set(pool["unique_id"])
        print(f"\nfrozen sample: {len(anchor):,} series, "
              f"{len(missing):,} of them outside the current eligible pool")
        if missing:
            print("  WARNING: the anchor is not a subset of the pool; the ladder "
                  "cannot be built on it without re-running the N=1,000 point")

    meta = {
        "sample_key": cfg.sample_key,
        "seed": cfg.run.seed,
        "panel_series": int(panel["unique_id"].nunique()),
        "eligible": int(elig["eligible"]),
        "strata": int(pool["stratum"].nunique()),
        "min_stratum": int(sizes.min()),
        "min_observed_weeks": cfg.sampling.min_observed_weeks,
        "min_train_weeks": cfg.protocol.min_train_weeks,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write(_META_PREFIX + json.dumps(meta) + "\n")
        pool.to_csv(fh, index=False)
    print(f"\npool -> {out}")

    report = Path(cfg.run.output_dir) / "eligible_pool_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        **meta,
        "panel_rows": len(panel),
        "panel_start": str(panel["ds"].min().date()),
        "panel_end": str(panel["ds"].max().date()),
        "origins": [str(o.date()) for o in origins],
        **elig,
        "pool_per_decile": {int(k): int(v) for k, v in per_decile.items()},
        "pool_intermittent": int((pool["zero_share"] >= 0.05).sum()),
        "anchor_in_pool": None if anchor is None else len(anchor & set(pool["unique_id"])),
    }, indent=2))
    print(f"report -> {report}")


if __name__ == "__main__":
    main()
