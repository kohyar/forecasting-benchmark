"""Publication tables from the long results file.

The output schema was chosen so a groupby lands directly on a table: no
reshaping between what the runner wrote and what goes in the paper.
"""
import numpy as np
import pandas as pd


def label_ablation_arms(metrics: pd.DataFrame) -> pd.DataFrame:
    """Fold the covariate arm into the model label.

    Every downstream table and test keys on `model`, so the with-covariate arm
    has to become its own model or a groupby averages the ablation away.
    """
    if "covariates" not in metrics.columns:
        return metrics

    out = metrics.copy()
    out["model"] = np.where(out["covariates"].fillna(False),
                            out["model"] + "+cov", out["model"])
    return out


def accuracy_table(metrics: pd.DataFrame, metric: str = "MASE") -> pd.DataFrame:
    """model x horizon, with the spread across repeats and folds."""
    sub = metrics[metrics["metric"] == metric]
    if sub.empty:
        raise ValueError(f"no rows for metric {metric!r}")

    table = (sub.groupby(["model", "horizon"])["value"]
             .agg(["median", "mean", "std", "count"])
             .unstack("horizon")
             .swaplevel(axis=1)
             .sort_index(axis=1))
    return table


def skill_scores(metrics: pd.DataFrame, baseline: str, metric: str = "MASE") -> pd.DataFrame:
    """Fractional improvement over a baseline, per series and horizon.

    Positive means better than the baseline.
    """
    sub = metrics[metrics["metric"] == metric]
    if baseline not in set(sub["model"]):
        raise ValueError(f"baseline {baseline!r} is not in the results")

    keys = ["unique_id", "horizon", "fold"]
    per = sub.groupby(["model"] + keys)["value"].mean().reset_index()
    ref = (per[per["model"] == baseline]
           .drop(columns="model").rename(columns={"value": "baseline"}))

    merged = per.merge(ref, on=keys, how="left")
    merged["skill"] = 1.0 - merged["value"] / merged["baseline"]
    merged.loc[~np.isfinite(merged["skill"]), "skill"] = np.nan
    return merged


def cost_table(metrics_or_timings: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """Fit and predict cost per model, never summed into one number.

    A fit shared across horizons is counted once - `fit_key` identifies it -
    so the totals are what the run actually cost.
    """
    t = metrics_or_timings
    t = t[t["status"] == "ok"] if "status" in t.columns else t

    fits = (t[~t["fit_reused"]].drop_duplicates("fit_key")
            .groupby("model")["fit_seconds"].agg(["sum", "median", "count"]))
    fits.columns = ["fit_seconds_total", "fit_seconds_median", "n_fits"]

    preds = t.groupby("model")["predict_seconds"].agg(["sum", "median"])
    preds.columns = ["predict_seconds_total", "predict_seconds_median"]

    extras = t.groupby("model").agg(
        family=("family", "first"),
        peak_memory_mb=("peak_memory_mb", "max"),
        n_series=("n_series", "max"),
        n_params=("n_params", "max"),
        device=("device", "first"),
        tuning_trials=("tuning_trials", "first"),
    )

    table = extras.join(fits).join(preds)
    table["total_seconds"] = table["fit_seconds_total"] + table["predict_seconds_total"]

    rate = getattr(getattr(cfg, "cost", None), "usd_per_hour", None) if cfg else None
    if rate:
        per_series = table["total_seconds"] / table["n_series"].replace(0, np.nan)
        table["usd_per_1k_series"] = per_series * 1000 * rate / 3600.0
        table["instance_type"] = cfg.cost.instance_type
    return table


def blocks_for_testing(metrics: pd.DataFrame, metric: str, horizon: int) -> pd.DataFrame:
    """series x model matrix for Friedman and the post-hocs.

    Repeats and folds are averaged within a series; the series stays the block,
    because that is the unit the tests treat as independent.
    """
    sub = metrics[(metrics["metric"] == metric) & (metrics["horizon"] == horizon)]
    if sub.empty:
        raise ValueError(f"no rows for metric {metric!r} at horizon {horizon}")

    wide = (sub.groupby(["unique_id", "model"])["value"].mean()
            .unstack("model"))
    return wide.dropna()


def critical_difference_diagram(blocks: pd.DataFrame, path, alpha: float = 0.05,
                                title: str = None):
    """Demsar-style diagram: mean ranks on an axis, cliques joined by a bar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tsbench.stats.tests import friedman_nemenyi

    result = friedman_nemenyi(blocks, alpha=alpha)
    ranks = result["mean_ranks"].sort_values()
    cd = result["critical_difference"]
    k = len(ranks)

    fig, ax = plt.subplots(figsize=(8, 1.6 + 0.32 * k))
    lo, hi = 0.8, k + 0.2
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.plot([lo, hi], [0.82, 0.82], color="black", linewidth=1)
    for tick in range(1, k + 1):
        ax.plot([tick, tick], [0.82, 0.85], color="black", linewidth=1)
        ax.text(tick, 0.88, str(tick), ha="center", va="bottom", fontsize=9)

    for i, (name, rank) in enumerate(ranks.items()):
        y = 0.72 - i * (0.62 / max(k, 1))
        ax.plot([rank, rank], [0.82, y], color="0.3", linewidth=1)
        ax.plot([rank, lo + 0.1], [y, y], color="0.3", linewidth=1)
        ax.text(lo + 0.05, y, f"{name} ({rank:.2f})", ha="right", va="center", fontsize=9)

    # cliques: consecutive models whose mean ranks differ by less than the CD
    values = ranks.to_numpy()
    bar_y = 0.79
    for start in range(k):
        end = start
        while end + 1 < k and values[end + 1] - values[start] <= cd:
            end += 1
        if end > start:
            ax.plot([values[start], values[end]], [bar_y, bar_y],
                    color="black", linewidth=3, solid_capstyle="butt")
            bar_y -= 0.03

    ax.text(hi, 0.95, f"CD = {cd:.2f}  (alpha={alpha}, n={result['n_blocks']})",
            ha="right", va="top", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return result
