"""Statistical testing and publication tables, over a finished results file.

    python scripts/run_stats.py --run results/spins-weekly-v1

Deliberately separate from the runner: the tests are re-runnable against
stored results without recomputing a single forecast.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from tsbench.config import Config
from tsbench.stats.aggregate import (
    accuracy_table,
    blocks_for_testing,
    cost_table,
    critical_difference_diagram,
    skill_scores,
)
from tsbench.stats.tests import bootstrap_skill_ci, diebold_mariano, posthoc_comparison


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a results/<run-name> directory")
    ap.add_argument("--metric", default="MASE")
    ap.add_argument("--baseline", default="seasonal_naive")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    run_dir = Path(args.run)
    metrics = pd.read_parquet(run_dir / "metrics.parquet")
    timings = pd.read_parquet(run_dir / "timings.parquet")
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    cfg = Config.from_dict(meta["config"])

    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"run {meta['run_name']}  config {meta['config_hash']}  "
          f"seed {meta['seed']}  repeats {meta['n_repeats']}")
    print(f"protocol: {meta['protocol']['folds']} folds, step "
          f"{meta['protocol']['step_weeks']}w, horizons {meta['protocol']['horizons']}, "
          f"overlapping={meta['protocol']['overlapping_test_windows']}\n")

    accuracy = accuracy_table(metrics, metric=args.metric)
    accuracy.to_csv(out_dir / f"accuracy_{args.metric}.csv")
    print(f"median {args.metric} by horizon:")
    print(accuracy.xs("median", level=1, axis=1).round(4).to_string(), "\n")

    cost = cost_table(timings, cfg=cfg)
    cost.to_csv(out_dir / "cost.csv")
    print("cost (fit and predict kept apart):")
    columns = [c for c in ("family", "fit_seconds_total", "predict_seconds_total",
                           "peak_memory_mb", "tuning_trials", "usd_per_1k_series")
               if c in cost.columns]
    print(cost[columns].round(3).to_string(), "\n")

    if args.baseline in set(metrics["model"]):
        skill = skill_scores(metrics, baseline=args.baseline, metric=args.metric)
        skill.to_parquet(out_dir / "skill.parquet", index=False)
        print(f"skill vs {args.baseline} (bootstrapped 95% CI):")
        for model, group in skill.groupby("model"):
            ci = bootstrap_skill_ci(group["skill"], seed=cfg.run.seed)
            print(f"  {model:18s} {ci['estimate']:+.4f}  "
                  f"[{ci['lower']:+.4f}, {ci['upper']:+.4f}]  n={ci['n']}")
        print()

    for horizon in meta["protocol"]["horizons"]:
        blocks = blocks_for_testing(metrics, metric=args.metric, horizon=horizon)
        if blocks.shape[1] < 3:
            print(f"h={horizon}: need >=3 models for Friedman, have {blocks.shape[1]}\n")
            continue

        comparison = posthoc_comparison(blocks, alpha=args.alpha)
        comparison.to_csv(out_dir / f"posthoc_h{horizon}.csv", index=False)

        diagram = out_dir / f"critical_difference_h{horizon}.png"
        result = critical_difference_diagram(
            blocks, diagram, alpha=args.alpha,
            title=f"{args.metric}, horizon {horizon}")

        print(f"h={horizon}: Friedman chi2={result['statistic']:.2f} "
              f"p={result['p_value']:.2e}  CD={result['critical_difference']:.3f}  "
              f"n={result['n_blocks']} series")
        disagree = comparison[~comparison["agree"]]
        if len(disagree):
            print(f"  Nemenyi and Wilcoxon disagree on {len(disagree)} pair(s) "
                  f"- report this rather than choosing one:")
            for _, row in disagree.iterrows():
                print(f"    {row['model_a']} vs {row['model_b']}: "
                      f"nemenyi={row['nemenyi_significant']} "
                      f"wilcoxon={row['wilcoxon_significant']}")
        else:
            print("  both post-hocs agree on every pair")
        print(f"  diagram -> {diagram}")

        dm = _pairwise_dm(metrics, args.metric, horizon)
        dm.to_csv(out_dir / f"diebold_mariano_h{horizon}.csv", index=False)
        print(f"  Diebold-Mariano (HLN-corrected, h={horizon}) -> "
              f"{(dm['p_value'] < args.alpha).sum()}/{len(dm)} pairs separate\n")

    print(f"analysis -> {out_dir}")


def _pairwise_dm(metrics: pd.DataFrame, metric: str, horizon: int) -> pd.DataFrame:
    """DM on per-series scaled errors, with the small-sample correction the
    overlapping windows require."""
    wide = blocks_for_testing(metrics, metric=metric, horizon=horizon)
    rows = []
    for left, right in combinations(wide.columns, 2):
        result = diebold_mariano(wide[left].to_numpy(), wide[right].to_numpy(),
                                 horizon=horizon, power=1)
        rows.append({"model_a": left, "model_b": right, **result})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
