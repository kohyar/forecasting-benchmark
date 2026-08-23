"""The paper's tables, as DataFrames ready to emit."""
import pandas as pd

# backslash first, or the replacements introduced below get escaped again
_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _escape(text) -> str:
    out = str(text)
    for char, replacement in _LATEX_SPECIALS.items():
        out = out.replace(char, replacement)
    return out


def to_latex(frame: pd.DataFrame, caption: str, label: str,
             float_format: str = "%.4f") -> str:
    """A booktabs table. Model names carry underscores, so escaping is not
    optional - an unescaped one fails the LaTeX build rather than the proof.
    """
    escaped = frame.copy()
    escaped.index = pd.Index([_escape(v) for v in escaped.index],
                             name=_escape(frame.index.name) if frame.index.name else None)
    escaped.columns = [_escape(c) for c in escaped.columns]
    for column in escaped.columns:
        if escaped[column].dtype == object:
            escaped[column] = escaped[column].map(_escape)

    return escaped.to_latex(caption=_escape(caption), label=label,
                            float_format=float_format, escape=False,
                            position="htbp")


def main_accuracy_table(metrics: pd.DataFrame, metric_names=("MASE", "RMSSE", "sMAPE"),
                        horizons=(4, 13)) -> pd.DataFrame:
    """Every accuracy metric side by side, one row per model.

    Medians, matching what the runner already prints - a mean over MASE is
    dominated by a handful of near-zero-denominator series.
    """
    out = pd.DataFrame(index=pd.Index(sorted(metrics["model"].unique()), name="model"))
    if "family" in metrics.columns:
        out["family"] = metrics.groupby("model")["family"].first()

    for metric in metric_names:
        for horizon in horizons:
            scope = metrics[(metrics["metric"] == metric)
                            & (metrics["horizon"] == horizon)]
            out[f"{metric}_h{horizon}"] = scope.groupby("model")["value"].median()
    return out


def ranking_flip_table(metrics: pd.DataFrame, series_ids, horizon: int,
                       metric_pair=("MASE", "sMAPE")) -> pd.DataFrame:
    """Rank the same models under two metrics over the same series.

    The paper's claim is that the metric decides the ordering, so both columns
    have to come from one set of series - scoping to `series_ids` is what makes
    the comparison mean anything.
    """
    first, second = metric_pair
    scope = metrics[metrics["unique_id"].isin(series_ids)
                    & (metrics["horizon"] == horizon)]

    out = pd.DataFrame(index=pd.Index(sorted(scope["model"].unique()), name="model"))
    for metric in (first, second):
        value = scope[scope["metric"] == metric].groupby("model")["value"].median()
        out[metric] = value
        out[f"rank_{metric}"] = value.rank(method="min").astype(int)

    out["rank_delta"] = out[f"rank_{second}"] - out[f"rank_{first}"]
    return out.sort_values(f"rank_{first}")


def cost_per_1k_table(timings: pd.DataFrame) -> pd.DataFrame:
    """What one run of a model over 1k series costs: one fit plus one predict.

    Folds and repeats measure that same work several times over, so both are
    collapsed with a median rather than summed - a sum answers what the
    benchmark cost, which is a different number and belongs in the appendix.
    """
    t = timings[timings["status"] == "ok"] if "status" in timings.columns else timings

    fit = t[~t["fit_reused"]].groupby("model")["fit_seconds"].median()
    predict = t.groupby(["model", "horizon"])["predict_seconds"].median().unstack()

    out = pd.DataFrame({"fit_s": fit})
    for horizon in predict.columns:
        out[f"predict_s_h{horizon}"] = predict[horizon]
        out[f"per1k_s_h{horizon}"] = fit + predict[horizon]
    for horizon in predict.columns:
        per1k = out[f"per1k_s_h{horizon}"]
        out[f"rel_h{horizon}"] = per1k / per1k.min()

    for column, how in [("family", "first"), ("peak_memory_mb", "max"),
                        ("n_params", "max"), ("tuning_trials", "first")]:
        if column in t.columns:
            out[column] = t.groupby("model")[column].agg(how)
    return out


def model_roster_table(registry, names, params=None) -> pd.DataFrame:
    """T4: what each model actually is - implementation, version, whether it
    was tuned, and for the zero-shot models which checkpoint was loaded.
    """
    from importlib.metadata import PackageNotFoundError, version

    rows = []
    for name in names:
        adapter = registry.get(name)
        package = getattr(adapter, "package", "") or ""
        try:
            installed = version(package) if package else ""
        except PackageNotFoundError:
            installed = "not installed"
        rows.append({
            "model": name,
            "family": adapter.family,
            "package": package,
            "version": installed,
            "checkpoint": getattr(adapter, "checkpoint", "") or "",
            "tuned": "yes" if getattr(adapter, "tunable", True) else "zero-shot",
        })
    return pd.DataFrame(rows).set_index("model")


def dataset_profile_table(profile: pd.DataFrame, zero_share_bins=None) -> pd.DataFrame:
    """T3: the panel that was actually benchmarked, not the export it came from."""
    intermittent_at = (zero_share_bins or [0.0, 0.05])[1]
    intermittent = int((profile["zero_share"] >= intermittent_at).sum())

    rows = {
        "series": f"{len(profile):,}",
        "observations per series (median)": f"{profile['n_obs'].median():,.0f}",
        "observations per series (min)": f"{profile['n_obs'].min():,.0f}",
        f"intermittent series (zero share >= {intermittent_at})": f"{intermittent:,}",
        "share of series intermittent": f"{intermittent / len(profile):.1%}",
        "zero share (median)": f"{profile['zero_share'].median():.3f}",
        "zero share (max)": f"{profile['zero_share'].max():.3f}",
    }
    if "total_volume" in profile.columns:
        rows["total volume (sum)"] = f"{profile['total_volume'].sum():,.0f}"
    return pd.DataFrame({"value": pd.Series(rows)}).rename_axis("property")
