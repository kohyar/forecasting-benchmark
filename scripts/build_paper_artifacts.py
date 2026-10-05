"""Every figure and table the paper needs, from a finished results file.

    python scripts/build_paper_artifacts.py --run results/spins-weekly-v1

Reads the assembled run - so run `--collect-only` first if the tiers were run
separately, or this sees only the tier that wrote last. Nothing is recomputed:
the forecasts are already in the results file.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from tsbench.config import Config  # noqa: E402
from tsbench.data.sampling import load_sample  # noqa: E402
from tsbench.models.registry import default as default_registry  # noqa: E402
from tsbench.report.figures import (  # noqa: E402
    accuracy_vs_cost,
    coverage_calibration,
    error_distribution,
)
from tsbench.report.style import save_figure  # noqa: E402
from tsbench.stats.aggregate import (  # noqa: E402
    blocks_for_testing,
    critical_difference_diagram,
)
from tsbench.report.tables import (  # noqa: E402
    cost_per_1k_table,
    dataset_profile_table,
    main_accuracy_table,
    model_roster_table,
    ranking_flip_table,
    to_latex,
)


def _emit(frame: pd.DataFrame, out_dir: Path, stem: str, caption: str, label: str,
          float_format: str = "%.4f", fit_width: bool = False) -> None:
    frame.to_csv(out_dir / f"{stem}.csv")
    (out_dir / f"{stem}.tex").write_text(
        to_latex(frame, caption=caption, label=label, float_format=float_format,
                 fit_width=fit_width))
    print(f"  {stem:<28} {len(frame):>3} rows -> {stem}.tex, {stem}.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a results/<run-name> directory")
    ap.add_argument("--sample", help="sample_series.csv, when the path the run "
                                     "recorded is not reachable from here")
    ap.add_argument("--intermittent-bin", type=int, default=1,
                    help="lowest zero_bin counted as intermittent for T5")
    args = ap.parse_args()

    run_dir = Path(args.run)
    metrics = pd.read_parquet(run_dir / "metrics.parquet")
    timings = pd.read_parquet(run_dir / "timings.parquet")
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    cfg = Config.from_dict(meta["config"])
    horizons = tuple(cfg.protocol.horizons)

    out_dir = run_dir / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(metrics["model"].unique())
    # cfg.models.enabled reflects whichever tier wrote this file - --family
    # rewrites it before the config is stored - so the results are the only
    # honest roster. Print the family split instead: a missing tier shows there.
    by_family = metrics.groupby("family")["model"].nunique().to_dict()

    print(f"run {meta['run_name']}  {len(models)} models  horizons {list(horizons)}")
    print("  families: " + ", ".join(f"{k} {v}" for k, v in sorted(by_family.items()))
          + "   (a missing tier means --collect-only has not been run)")
    print(f"tables -> {out_dir}")

    accuracy = main_accuracy_table(metrics, horizons=horizons)
    _emit(accuracy, out_dir, "T7_main_accuracy",
          "Median accuracy by model and horizon.", "tab:main-accuracy",
          fit_width=True)

    # What each model actually spent. The timings carry it too, but a run
    # assembled from checkpoints predates any later tuning, so the recorded
    # budget is the one to trust - in T8's column as much as in T4's.
    tuning_path = run_dir / "tuning_best.json"
    budget = (json.loads(tuning_path.read_text()).get("budget", {})
              if tuning_path.exists() else {})
    if not budget:
        print("  NOTE: no tuning_best.json - T4 will report every model as untuned "
              "and T8 will show the trials annotation the timings carry")

    cost = cost_per_1k_table(timings, budget=budget or None)
    _emit(cost, out_dir, "T8_cost",
          "Compute cost of running each model once over 1,000 series "
          "(one fit plus one predict), and as a multiple of the cheapest model.",
          "tab:cost", float_format="%.3f", fit_width=True)

    registry = default_registry()
    roster = model_roster_table(registry, models, versions=meta.get("packages"),
                                cfg=cfg, budget=budget)
    _emit(roster, out_dir, "T4_model_roster",
          "Model roster: implementation, version and tuning treatment.",
          "tab:roster", fit_width=True)

    sample, _ = load_sample(args.sample or cfg.sampling.sample_path)
    profile = dataset_profile_table(sample, zero_share_bins=cfg.sampling.zero_share_bins)
    _emit(profile, out_dir, "T3_dataset_profile",
          "Profile of the benchmarked sample.", "tab:dataset")

    intermittent = sample.loc[sample["zero_bin"] >= args.intermittent_bin, "unique_id"]
    if len(intermittent) == 0:
        print("  T5 skipped: no series in the intermittent bins")
    else:
        flip = ranking_flip_table(metrics, series_ids=intermittent.tolist(),
                                  horizon=horizons[0])
        _emit(flip, out_dir, "T5_ranking_flip",
              f"Model ranking under MASE against sMAPE on the {len(intermittent)} "
              f"intermittent series, horizon {horizons[0]}.", "tab:ranking-flip")
        moved = int((flip["rank_delta"] != 0).sum())
        print(f"  T5: {moved}/{len(flip)} models change rank between the two metrics "
              f"({len(intermittent)} series)")

    print("figures ->", out_dir)
    families = metrics.groupby("model")["family"].first().to_dict()
    for horizon in horizons:
        # run_stats.py writes these; without them F1 draws a frontier that
        # implies a ranking the significance tests may not support
        dm_path = run_dir / "analysis" / f"diebold_mariano_h{horizon}.csv"
        dm = pd.read_csv(dm_path) if dm_path.exists() else None
        if dm is None:
            print(f"  note: no {dm_path.name} yet, so F1 h={horizon} omits the "
                  f"tie band. Run run_stats.py first for the full figure.")

        fig = accuracy_vs_cost(cost, accuracy, horizon=horizon, dm=dm,
                               highlight=("autoarima", "tabpfn_ts"))
        save_figure(fig, out_dir / f"F1_accuracy_vs_cost_h{horizon}")
        print(f"  F1_accuracy_vs_cost_h{horizon}")

        blocks = blocks_for_testing(metrics, metric="MASE", horizon=horizon)
        if blocks.shape[1] >= 3:
            critical_difference_diagram(
                blocks, out_dir / f"F7_critical_difference_h{horizon}.png")
            print(f"  F7_critical_difference_h{horizon}")

    save_figure(error_distribution(metrics, horizons, families=families),
                out_dir / "F8_error_distribution")
    print("  F8_error_distribution")

    save_figure(coverage_calibration(metrics, horizons, families=families),
                out_dir / "F_coverage_calibration")
    print("  F_coverage_calibration")

    print(f"\nThe significance tables behind F1's tie band live in "
          f"{run_dir / 'analysis'}, written by:\n"
          f"  python scripts/run_stats.py --run {run_dir}")


if __name__ == "__main__":
    main()
