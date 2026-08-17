"""Build and freeze the stratified series sample.

    python scripts/build_sample.py [--config configs/default.yaml]

Writes the sample (with a provenance header) and a report of what was dropped
and why - both are inputs to the paper's data section.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from tsbench.config import Config
from tsbench.data.loader import load_panel
from tsbench.data.sampling import save_sample, stratified_sample
from tsbench.eval.metrics import denominators_for_protocol
from tsbench.eval.splitter import RollingOriginSplitter
from tsbench.pipeline import eligible_profile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    print(f"config {args.config}  hash={cfg.hash}  seed={cfg.run.seed}")

    panel = load_panel(cfg)
    print(f"panel: {len(panel):,} rows  {panel['unique_id'].nunique():,} series  "
          f"{panel['ds'].min().date()} .. {panel['ds'].max().date()}")

    splitter = RollingOriginSplitter(cfg)
    origins = splitter.origins(panel)
    print(f"origins ({cfg.protocol.folds} folds, step {cfg.protocol.step}w): "
          + ", ".join(str(o.date()) for o in origins))

    profile, elig = eligible_profile(panel, cfg)
    n_total = elig["series_total"]
    short = elig["dropped_below_min_train"]
    uncovered = elig["dropped_no_evaluation_coverage"]
    flat = elig["dropped_zero_denominator"]
    print(f"\neligibility ({n_total:,} series):")
    print(f"  below min_train_weeks={cfg.protocol.min_train_weeks}: {short:,}")
    print(f"  do not cover the evaluation span:                {uncovered:,}")
    print(f"  zero seasonal denominator (unscoreable):         {flat:,}")
    print(f"  eligible:                                        {elig['eligible']:,}")
    print(f"  minimum training weeks found (eligible): {elig['min_train_weeks_found_eligible']}")
    print(f"  minimum training weeks found (all):      {elig['min_train_weeks_found_all']}")

    sample = stratified_sample(profile, cfg)
    save_sample(sample, cfg.sampling.sample_path, cfg)
    print(f"\nsample: {len(sample):,} series -> {cfg.sampling.sample_path}")

    per_decile = sample["volume_decile"].value_counts().sort_index()
    print("  per volume decile: " + ", ".join(f"{d}:{n}" for d, n in per_decile.items()))
    print(f"  zero-share > 0:    {int((sample['zero_share'] > 0).sum())} series "
          f"(population {int((profile['zero_share'] > 0).sum())})")
    print(f"  volume range:      {sample['total_volume'].min():,.0f} .. "
          f"{sample['total_volume'].max():,.0f}")

    den = denominators_for_protocol(panel, cfg, splitter)
    den = den[den["unique_id"].isin(sample["unique_id"])]
    print(f"  MASE denominators: {den['n_pairs'].min()} .. {den['n_pairs'].max()} pairs, "
          f"{int(den['zero_denominator'].sum())} zero")

    out = Path(cfg.run.output_dir) / "sample_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config_hash": cfg.hash,
        "seed": cfg.run.seed,
        "panel_rows": len(panel),
        "panel_series": int(panel["unique_id"].nunique()),
        "panel_start": str(panel["ds"].min().date()),
        "panel_end": str(panel["ds"].max().date()),
        "origins": [str(o.date()) for o in origins],
        **elig,
        "sample_size": len(sample),
        "sample_per_decile": {int(k): int(v) for k, v in per_decile.items()},
        "sample_intermittent": int((sample["zero_share"] > 0).sum()),
    }, indent=2))
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
