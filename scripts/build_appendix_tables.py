"""The appendix tables: everything the main artefacts leave out.

    python scripts/build_appendix_tables.py --run results/spins-weekly-v1 \
        --scaling results/scaling

Reads the assembled run and the scaling report; nothing is recomputed.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from tsbench.config import Config
from tsbench.models.registry import default as default_registry
from tsbench.report.tables import declared_space, main_accuracy_table, to_latex

PROBABILISTIC = ("MASE", "RMSSE", "sMAPE", "CRPS", "pinball", "coverage_80")


def _emit(frame: pd.DataFrame, out_dir: Path, stem: str, caption: str, label: str,
          float_format: str = "%.4f", fit_width: bool = False) -> None:
    frame.to_csv(out_dir / f"{stem}.csv")
    (out_dir / f"{stem}.tex").write_text(
        to_latex(frame, caption=caption, label=label, float_format=float_format,
                 fit_width=fit_width))
    print(f"  {stem:<24} {len(frame):>3} rows -> {stem}.tex, {stem}.csv")


def fold_timings_table(timings: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Fit and predict seconds per fold, median over repeats, one horizon."""
    t = timings[(timings["status"] == "ok") & (timings["horizon"] == horizon)]
    fit = t.groupby(["model", "fold"])["fit_seconds"].median().unstack("fold")
    fit.columns = [f"fit f{c}" for c in fit.columns]
    predict = t.groupby(["model", "fold"])["predict_seconds"].median().unstack("fold")
    predict.columns = [f"predict f{c}" for c in predict.columns]
    out = fit.join(predict)
    out.index.name = "model"
    return out


def _selected(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def search_space_table(registry, names, cfg, best: dict) -> pd.DataFrame:
    """Every searchable hyperparameter with its range and the value chosen."""
    rows = []
    for name in names:
        space = declared_space(registry.get(name), cfg)
        chosen = best.get(name, {})
        for param, span in space.items():
            rows.append({"model": name, "hyperparameter": param, "range": span,
                         "selected": _selected(chosen.get(param))})
    return pd.DataFrame(rows).set_index("model")


def crossover_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    out = pd.DataFrame({
        "model": frame["model"],
        "horizon": frame["horizon"],
        "reference": frame["reference"],
        "first N at or below": frame["first_n_below"].map(
            lambda n: "none" if pd.isna(n) else f"{int(n):,}"),
        "MASE at 4,000": frame["at_max_n"],
        "gap at 4,000 (%)": frame["gap_at_max_pct"],
    })
    return out.set_index("model")


def paired_tests_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    out = frame.rename(columns={
        "median_first": "median first", "median_second": "median second",
        "median_paired_diff": "median paired diff",
        "share_first_better": "share first better", "p_value": "p",
    }).drop(columns=["n_series"])
    out["p"] = out["p"].map(lambda p: f"{p:.2g}")
    return out.set_index("model")


def packages_table(meta: dict) -> pd.DataFrame:
    rows = {"python": meta.get("python", "")}
    rows.update(meta.get("packages", {}))
    hardware = meta.get("hardware", {})
    if hardware.get("torch_version"):
        rows["torch (build)"] = hardware["torch_version"]
    out = pd.DataFrame({"version": pd.Series(rows)})
    out.index.name = "package"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a results/<run-name> directory")
    ap.add_argument("--scaling", help="the scaling report directory")
    args = ap.parse_args()

    run_dir = Path(args.run)
    metrics = pd.read_parquet(run_dir / "metrics.parquet")
    timings = pd.read_parquet(run_dir / "timings.parquet")
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    cfg = Config.from_dict(meta["config"])
    horizons = tuple(cfg.protocol.horizons)

    out_dir = run_dir / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run {meta['run_name']}  tables -> {out_dir}")

    present = [m for m in PROBABILISTIC if m in set(metrics["metric"].unique())]
    full = main_accuracy_table(metrics, metric_names=present, horizons=horizons)
    _emit(full, out_dir, "T10_full_metrics",
          "Median point and probabilistic accuracy by model and horizon. "
          "CRPS is twice the mean pinball loss over the nine quantiles; "
          "coverage is of the nominal 80% interval.",
          "tab:full-metrics", fit_width=True)

    folds = fold_timings_table(timings, horizons[0])
    _emit(folds, out_dir, "T11_fold_timings",
          f"Fit and prediction seconds per fold at horizon {horizons[0]}, "
          "median over repeats, for 1,000 series.",
          "tab:fold-timings", float_format="%.2f", fit_width=True)

    tuning_path = run_dir / "tuning_best.json"
    best = json.loads(tuning_path.read_text()).get("best", {}) if tuning_path.exists() else {}
    registry = default_registry()
    spaces = search_space_table(registry, sorted(metrics["model"].unique()), cfg, best)
    _emit(spaces, out_dir, "T12_search_spaces",
          "Declared search ranges and the configuration the 20-trial search "
          "selected.", "tab:search-spaces")

    _emit(packages_table(meta), out_dir, "T15_packages",
          "Package versions recorded by the run.", "tab:packages")

    if args.scaling:
        scaling = Path(args.scaling)
        _emit(crossover_table(scaling / "crossover.csv"), out_dir, "T13_crossover",
              "Crossover search on the fixed evaluation set: the AutoARIMA "
              "reference level, the first rung at or below it, and the gap at "
              "the largest rung.", "tab:crossover", float_format="%.3f")
        _emit(paired_tests_table(scaling / "paired_tests.csv"), out_dir,
              "T14_paired_tests",
              "Paired Wilcoxon signed-rank tests on per-series MASE (mean over "
              "folds and repeats) over the 500 shared series: each model at "
              "4,000 series against itself at 500, and each rung against "
              "AutoARIMA from the main run.", "tab:paired-tests",
              float_format="%.3f", fit_width=True)


if __name__ == "__main__":
    main()
