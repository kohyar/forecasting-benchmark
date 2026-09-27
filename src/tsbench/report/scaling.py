"""The cardinality ladder: how accuracy and compute move with the series count.

Each rung is a separate run over a nested sample, so the frames arrive one per
run and are joined here on n_series. Two readings of the same runs:

- On the **fixed evaluation set** - the series every rung shares, which with
  nesting is the smallest rung - a local or zero-shot model produces the same
  forecasts at every N, so only a global model can move. That movement is what
  more training series bought, on the same held-out series, paired.
- On the **full sample** of each rung the medians also shift with which series
  were drawn. The zero-shot model fits nothing, so its spread across rungs is
  that composition effect alone, and a trained model's movement has to clear
  it before it means anything.

Compute per 1,000 series always comes from the full rung.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator

from tsbench.report.style import (
    FAMILIES,
    GRID,
    INK,
    INK_MUTED,
    apply_paper_style,
    family_style,
)
from tsbench.report.tables import cost_per_1k_table


def load_ladder(run_dirs) -> dict:
    """{n_series: rung} for every finished run directory, smallest first.

    A rung is finished when the runner wrote its metadata; a directory that
    holds only checkpoints is skipped and named, so a partial ladder still
    reports.
    """
    rungs = {}
    for run_dir in map(Path, run_dirs):
        meta_path = run_dir / "run_metadata.json"
        if not meta_path.exists():
            print(f"  skipping {run_dir}: no run_metadata.json yet (run not finished)")
            continue
        meta = json.loads(meta_path.read_text())
        n = int(meta["config"]["sampling"]["n_series"])
        if n in rungs:
            raise ValueError(f"two runs claim N={n}: {rungs[n]['path']} and {run_dir}")
        rungs[n] = {
            "path": run_dir,
            "meta": meta,
            "metrics": pd.read_parquet(run_dir / "metrics.parquet"),
            "timings": pd.read_parquet(run_dir / "timings.parquet"),
        }
    return dict(sorted(rungs.items()))


def common_series(rungs: dict) -> list:
    """The series every rung scored - the fixed evaluation set. With a nested
    ladder that is the smallest rung, but it is intersected rather than
    assumed, so a rung that lost a series to a failure still lines up."""
    sets = [set(r["metrics"]["unique_id"].unique()) for r in rungs.values()]
    return sorted(set.intersection(*sets)) if sets else []


def ladder_warnings(rungs: dict) -> list:
    """What would make the curve lie: rungs from different builds, models missing
    from some rungs, or units that failed."""
    notes = []
    commits = {n: (r["meta"].get("git_commit") or "")[:8] for n, r in rungs.items()}
    if len(set(commits.values())) > 1:
        notes.append("rungs were measured by different builds: "
                     + ", ".join(f"N={n} {c or '?'}" for n, c in commits.items()))
    rosters = {n: set(r["metrics"]["model"].unique()) for n, r in rungs.items()}
    everyone = set.union(*rosters.values()) if rosters else set()
    for n, roster in rosters.items():
        if roster != everyone:
            notes.append(f"N={n} lacks {', '.join(sorted(everyone - roster))}")
    for n, r in rungs.items():
        for model, p in r["meta"].get("progress", {}).items():
            if p.get("complete", 0) < p.get("expected", 0) or p.get("failed"):
                notes.append(f"N={n} {model}: {p['complete']}/{p['expected']} units, "
                             f"{p.get('failed', 0)} failed")
    return notes


def bootstrap_median_band(scope: pd.DataFrame, n_boot: int = 300, seed: int = 0,
                          level: float = 0.95) -> tuple:
    """Interval for the median when series, not rows, are the sampling unit.

    A series contributes one row per fold and repeat, and those rows share its
    scale; resampling rows would treat them as independent and shrink the band.
    """
    wide = scope.pivot_table(index="unique_id", columns=["fold", "repeat"],
                             values="value", aggfunc="first").to_numpy()
    if wide.shape[0] < 2:
        value = np.nanmedian(wide) if wide.size else np.nan
        return value, value
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, wide.shape[0], size=(n_boot, wide.shape[0]))
    medians = np.array([np.nanmedian(wide[idx].ravel()) for idx in draws])
    tail = (1 - level) / 2
    return float(np.quantile(medians, tail)), float(np.quantile(medians, 1 - tail))


def scaling_table(rungs: dict, metrics=("MASE", "RMSSE"), horizons=(4, 13),
                  band: bool = True, n_boot: int = 300, series=None) -> pd.DataFrame:
    """T9: one row per (model, n_series) - median accuracy, the compute of one
    fit plus one predict over that rung, and the same compute per 1,000 series.

    `series` restricts the accuracy medians to those ids (the fixed evaluation
    set); the compute columns always describe the whole rung, since that is
    what was run. `cost_per_1k_table` measures the sample it was given, so
    dividing by the rung's size is what makes the column comparable across
    rungs - whether it falls with N is the amortisation argument for global
    models.
    """
    keep = set(series) if series is not None else None
    rows = []
    for n, rung in rungs.items():
        m = rung["metrics"]
        if keep is not None:
            m = m[m["unique_id"].isin(keep)]
        cost = cost_per_1k_table(rung["timings"])
        for model in sorted(m["model"].unique()):
            mine = m[m["model"] == model]
            row = {"model": model, "n_series": n,
                   "family": mine["family"].iloc[0] if "family" in mine else "",
                   "eval_series": int(mine["unique_id"].nunique())}
            for metric in metrics:
                for h in horizons:
                    scope = mine[(mine["metric"] == metric) & (mine["horizon"] == h)]
                    row[f"{metric}_h{h}"] = scope["value"].median()
                    if band and metric == metrics[0]:
                        lo, hi = bootstrap_median_band(scope, n_boot=n_boot)
                        row[f"{metric}_h{h}_lo"], row[f"{metric}_h{h}_hi"] = lo, hi
            if model in cost.index:
                row["fit_s"] = cost.loc[model, "fit_s"]
                for h in horizons:
                    run_s = cost.loc[model, f"per1k_s_h{h}"]
                    row[f"run_s_h{h}"] = run_s
                    row[f"per1k_s_h{h}"] = run_s * 1000.0 / n
            rows.append(row)
    return pd.DataFrame(rows).set_index(["model", "n_series"]).sort_index()


def reference_levels(metrics: pd.DataFrame, model: str, metric: str = "MASE",
                     horizons=(4, 13), series=None) -> dict:
    """{horizon: median} of one model from another run - the level the ladder
    is read against, typically the best local model at the paper's N. Pass
    `series` to take it over the fixed evaluation set, so the line and the
    curve describe the same series."""
    scope = metrics[(metrics["model"] == model) & (metrics["metric"] == metric)]
    if series is not None:
        scope = scope[scope["unique_id"].isin(set(series))]
    return {h: float(scope[scope["horizon"] == h]["value"].median())
            for h in horizons if (scope["horizon"] == h).any()}


def crossovers(table: pd.DataFrame, reference: dict, metric: str = "MASE",
               horizons=(4, 13)) -> pd.DataFrame:
    """Where each model first beats the reference level, if it does at all.

    "No crossover up to N" is a result, not a gap: it says the reference holds
    its lead over the whole range the panel can test.
    """
    rows = []
    for model, group in table.groupby(level="model"):
        g = group.droplevel("model").sort_index()
        for h in horizons:
            column = f"{metric}_h{h}"
            level = reference.get(h)
            if column not in g or level is None:
                continue
            below = g.index[g[column] < level]
            last = g[column].iloc[-1]
            rows.append({
                "model": model, "horizon": h, "reference": level,
                "first_n_below": int(below[0]) if len(below) else None,
                "max_n": int(g.index[-1]), "at_max_n": last,
                "gap_at_max_pct": (last / level - 1.0) * 100.0,
            })
    return pd.DataFrame(rows).set_index(["model", "horizon"])


def noise_band(table: pd.DataFrame, model: str, metric: str, horizon: int) -> tuple:
    """Spread of a zero-shot model's median across rungs: with no fitting, the
    only thing that moves it is which series were drawn."""
    if model not in table.index.get_level_values("model"):
        return None
    values = table.xs(model, level="model")[f"{metric}_h{horizon}"].dropna()
    return (float(values.min()), float(values.max())) if len(values) else None


def scaling_figure(table: pd.DataFrame, horizons=(4, 13), metric: str = "MASE",
                   reference: dict | None = None, reference_label: str = "",
                   noise_model: str | None = "chronos2", cost: bool = True,
                   title: str = "Accuracy and compute against series count",
                   accuracy_label: str | None = None):
    """F9: accuracy (top) and, with `cost`, compute per 1,000 series (bottom)
    against series count, one column per horizon. Series count doubles per
    rung, so the axis is log2 with a tick at each rung and nothing between.

    On the fixed evaluation set the zero-shot band is a point, so pass
    `noise_model=None` there; on the full sample a fixed reference line is
    confounded by composition, so pass `reference=None` there.
    """
    apply_paper_style()
    sizes = sorted(table.index.get_level_values("n_series").unique())
    n_rows = 2 if cost else 1
    fig, axes = plt.subplots(n_rows, len(horizons),
                             figsize=(3.7 * len(horizons), 5.8 if cost else 3.4),
                             sharex=True)
    axes = np.asarray(axes).reshape(n_rows, len(horizons))
    families = set()
    band_handle = reference_handle = None

    for col, h in enumerate(horizons):
        top = axes[0, col]
        bottom = axes[1, col] if cost else None
        score, cost_column = f"{metric}_h{h}", f"per1k_s_h{h}"

        # both are explained in the legend: text on the axes lands on the line
        # labels whenever the band and the reference level sit close, which is
        # exactly the case the figure is for
        band = noise_band(table, noise_model, metric, h) if noise_model else None
        if band and band[1] > band[0]:
            band_handle = top.axhspan(*band, color=family_style("foundation")["color"],
                                      alpha=0.12, linewidth=0, zorder=0,
                                      label=f"{noise_model} spread across rungs "
                                            "(sampling noise)")
        if reference and h in reference:
            reference_handle = top.axhline(reference[h], color=INK_MUTED, linestyle="--",
                                           linewidth=1.0, zorder=1,
                                           label=reference_label or "reference")

        ends_top, ends_bottom = [], []
        for model, group in table.groupby(level="model"):
            g = group.droplevel("model").sort_index()
            family = g["family"].iloc[0] if "family" in g else ""
            families.add(family)
            style = family_style(family)
            x = g.index.to_numpy(dtype=float)
            panels = [(top, score, ends_top)]
            if cost:
                panels.append((bottom, cost_column, ends_bottom))
            for ax, column, ends in panels:
                if column not in g:
                    continue
                y = g[column].to_numpy(dtype=float)
                keep = ~np.isnan(y)
                if not keep.any():
                    continue
                ax.plot(x[keep], y[keep], marker=style["marker"], color=style["color"],
                        markersize=5, linewidth=1.2, markeredgecolor="white",
                        markeredgewidth=0.5, zorder=3)
                if ax is top and f"{score}_lo" in g:
                    ax.fill_between(x[keep], g[f"{score}_lo"].to_numpy()[keep],
                                    g[f"{score}_hi"].to_numpy()[keep],
                                    color=style["color"], alpha=0.15, linewidth=0, zorder=2)
                ends.append((model, x[keep][-1], y[keep][-1]))

        top.set_xscale("log", base=2)
        top.set_xlim(sizes[0] / 1.3, sizes[-1] * 2.4)
        last = bottom if cost else top
        for ax in ([top, bottom] if cost else [top]):
            ax.set_xticks(sizes)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v)):,}"))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(axis="both", color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
        top.set_title(f"horizon {h}", loc="left")
        last.set_xlabel("series in the sample (log scale)")
        if col == 0:
            top.set_ylabel(accuracy_label or f"median {metric}  (lower is better)")
        _label_line_ends(top, ends_top)
        if cost:
            bottom.set_yscale("log")
            if col == 0:
                bottom.set_ylabel("compute seconds per 1,000 series\n"
                                  "(one fit + one predict, log scale)")
            _label_line_ends(bottom, ends_bottom)

    handles = []
    for family in FAMILIES:
        if family in families:
            style = family_style(family)
            handles.append(plt.Line2D([], [], color=style["color"], marker=style["marker"],
                                      markersize=5, linewidth=1.2, label=family,
                                      markeredgecolor="white", markeredgewidth=0.5))
    handles += [h for h in (reference_handle, band_handle) if h is not None]
    # the two explanatory entries are long, so no panel has room for the
    # legend without covering its own axis; it goes under the figure instead
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.08 if cost else 0.14, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.4, handlelength=2.2)
    return fig


def _label_line_ends(ax, ends, min_gap_pt: float = 9.0) -> None:
    """Name each line at its right end, nudging labels apart where two lines
    finish close together - on the accuracy panel they often do."""
    if not ends:
        return
    ax.figure.canvas.draw()
    to_pt = 72.0 / ax.figure.dpi
    order = sorted(ends, key=lambda e: ax.transData.transform((e[1], e[2]))[1])
    placed = []
    for model, x, y in order:
        pixel_y = ax.transData.transform((x, y))[1] * to_pt
        if placed and pixel_y - placed[-1] < min_gap_pt:
            pixel_y = placed[-1] + min_gap_pt
        placed.append(pixel_y)
        offset = pixel_y - ax.transData.transform((x, y))[1] * to_pt
        ax.annotate(model, (x, y), textcoords="offset points", xytext=(6, offset),
                    fontsize=7.5, color=INK, va="center", zorder=4)


def per_series_scores(metrics: pd.DataFrame, model: str, metric: str, horizon: int,
                      series=None) -> pd.Series:
    """One value per series: folds and repeats averaged, as the paper's
    post-hoc tests do, because the series is the unit treated as independent."""
    scope = metrics[(metrics["model"] == model) & (metrics["metric"] == metric)
                    & (metrics["horizon"] == horizon)]
    if series is not None:
        scope = scope[scope["unique_id"].isin(set(series))]
    return scope.groupby("unique_id")["value"].mean()


def paired_tests(rungs: dict, series, metric: str = "MASE", horizons=(4, 13),
                 reference: pd.DataFrame | None = None,
                 reference_model: str = "autoarima") -> pd.DataFrame:
    """Two paired questions on the fixed evaluation set, per model and horizon.

    Did the largest rung beat the smallest on the same series - what more
    training series bought - and where does the largest rung stand against the
    reference model on those series. Wilcoxon signed-rank with the series as
    the block; a negative median difference favours the first named side.
    """
    from scipy import stats

    sizes = sorted(rungs)
    lo, hi = sizes[0], sizes[-1]
    rows = []

    def compare(a: pd.Series, b: pd.Series, model, h, label):
        joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        diff = (joined["a"] - joined["b"]).to_numpy()
        if len(diff) < 2 or np.allclose(diff, 0):
            p_value = 1.0
        else:
            p_value = float(stats.wilcoxon(diff).pvalue)
        rows.append({
            "model": model, "horizon": h, "comparison": label, "n_series": int(len(diff)),
            "median_first": float(joined["a"].median()),
            "median_second": float(joined["b"].median()),
            "median_paired_diff": float(np.median(diff)),
            "share_first_better": float((diff < 0).mean()),
            "p_value": p_value,
        })

    models = sorted(set.intersection(*[set(r["metrics"]["model"].unique())
                                       for r in rungs.values()]))
    for model in models:
        for h in horizons:
            largest = per_series_scores(rungs[hi]["metrics"], model, metric, h, series)
            smallest = per_series_scores(rungs[lo]["metrics"], model, metric, h, series)
            compare(largest, smallest, model, h, f"N={hi:,} vs N={lo:,}")
            if reference is not None:
                ref = per_series_scores(reference, reference_model, metric, h, series)
                if len(ref):
                    compare(largest, ref, model, h, f"N={hi:,} vs {reference_model}")
    return pd.DataFrame(rows).set_index(["model", "horizon", "comparison"])
