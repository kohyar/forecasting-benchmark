"""Significance testing for the benchmark.

Diebold-Mariano compares two forecasts on the same series; Friedman with
Nemenyi and Wilcoxon-Holm compare many models across many series. Both
post-hocs are reported: Nemenyi's mean-rank procedure is known to be unstable
when models are added or removed (Benavoli et al. 2016), and where the two
disagree that is worth saying rather than picking the friendlier one.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def diebold_mariano(errors_a, errors_b, horizon: int, power: int = 2,
                    harvey_correction: bool = True,
                    dependence_lags: int | None = None,
                    clusters=None) -> dict:
    """Pairwise equal-predictive-accuracy test.

    A negative statistic favours the first model. The inputs are one loss
    per observation for each forecast; the test is on the mean of their
    differential, and the only question is how the variance of that mean is
    estimated.

    Two dependence structures are supported. When the differentials are
    indexed by time, h-step errors from overlapping windows are serially
    correlated up to lag h-1, so `dependence_lags` autocovariances (default
    horizon-1) enter the variance and the Harvey-Leybourne-Newbold factor and
    t reference correct the small-sample size. When each differential is one
    series - folds and repeats already averaged within it - the observations
    have no order to be serially correlated along; pass `dependence_lags=0`,
    and if the series fall into groups that share shocks (one product across
    many markets) pass their group labels as `clusters`: the variance is then
    the cluster sandwich, the reference is t on (groups - 1) degrees of
    freedom, and the lag correction does not apply.
    """
    a = np.asarray(errors_a, dtype=float)
    b = np.asarray(errors_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"error arrays differ in shape: {a.shape} vs {b.shape}")

    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    labels = None if clusters is None else np.asarray(clusters)[ok]
    n = len(a)
    lags = max(int(horizon) - 1, 0) if dependence_lags is None else max(int(dependence_lags), 0)
    if labels is not None:
        lags, harvey_correction = 0, False

    result = {"n": n, "horizon": int(horizon), "power": power,
              "n_autocovariances": lags, "harvey_correction": harvey_correction,
              "n_clusters": int(len(np.unique(labels))) if labels is not None else n,
              "statistic": np.nan, "p_value": np.nan, "mean_loss_difference": np.nan}
    if n < 3:
        return result

    d = np.abs(a) ** power - np.abs(b) ** power
    d_bar = float(d.mean())
    result["mean_loss_difference"] = d_bar

    centred = d - d_bar
    if labels is not None:
        # sum the centred differentials within each cluster; clusters are
        # independent of one another, members of one are not
        sums = pd.Series(centred).groupby(labels).sum().to_numpy()
        variance = float(sums @ sums) / n
        df = result["n_clusters"] - 1
    else:
        variance = float(centred @ centred) / n
        for lag in range(1, lags + 1):
            if lag >= n:
                break
            variance += 2.0 * float(centred[lag:] @ centred[:-lag]) / n
        df = n - 1

    if variance <= 0 or df < 1:
        result["statistic"] = 0.0 if d_bar == 0 else np.nan
        result["p_value"] = 1.0 if d_bar == 0 else np.nan
        return result

    statistic = d_bar / np.sqrt(variance / n)

    if harvey_correction:
        h = lags + 1
        factor = (n + 1 - 2 * h + h * (h - 1) / n) / n
        statistic *= np.sqrt(max(factor, 1e-12))

    result["statistic"] = float(statistic)
    result["p_value"] = float(2 * stats.t.cdf(-abs(statistic), df=df))
    return result


def friedman_nemenyi(scores: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Friedman omnibus test with the Nemenyi post-hoc.

    `scores` is one row per block (series or dataset), one column per model,
    holding a loss where lower is better.
    """
    data = scores.dropna()
    n, k = data.shape
    if n < 2 or k < 3:
        raise ValueError(f"need >=2 blocks and >=3 models, got {n} x {k}")

    statistic, p_value = stats.friedmanchisquare(*[data[c].to_numpy() for c in data])
    ranks = data.rank(axis=1)
    mean_ranks = ranks.mean()

    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    critical_difference = float(q * np.sqrt(k * (k + 1) / (6 * n)))

    pairs = []
    for left, right in combinations(data.columns, 2):
        difference = float(mean_ranks[left] - mean_ranks[right])
        pairs.append({
            "model_a": left, "model_b": right,
            "rank_difference": difference,
            "abs_rank_difference": abs(difference),
            "significant": abs(difference) > critical_difference,
        })

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_blocks": int(n),
        "n_models": int(k),
        "alpha": alpha,
        "mean_ranks": mean_ranks,
        "critical_difference": critical_difference,
        "pairs": pd.DataFrame(pairs),
    }


def holm_adjust(p_values) -> list:
    """Holm step-down adjustment, monotone in the sorted p-values."""
    p = np.asarray(list(p_values), dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)

    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def wilcoxon_holm(scores: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests with Holm-adjusted p-values."""
    data = scores.dropna()
    rows = []
    for left, right in combinations(data.columns, 2):
        a, b = data[left].to_numpy(), data[right].to_numpy()
        if np.allclose(a, b):
            statistic, p_value = np.nan, 1.0
        else:
            statistic, p_value = stats.wilcoxon(a, b)
        rows.append({"model_a": left, "model_b": right,
                     "statistic": float(statistic) if np.isfinite(statistic) else np.nan,
                     "p_value": float(p_value),
                     "median_difference": float(np.median(a - b))})

    frame = pd.DataFrame(rows)
    frame["p_value_adjusted"] = holm_adjust(frame["p_value"])
    frame["significant"] = frame["p_value_adjusted"] < alpha
    return frame


def posthoc_comparison(scores: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Both post-hocs side by side, with an explicit agreement flag."""
    nemenyi = friedman_nemenyi(scores, alpha=alpha)["pairs"]
    wilcoxon = wilcoxon_holm(scores, alpha=alpha)

    merged = nemenyi.merge(wilcoxon, on=["model_a", "model_b"],
                           suffixes=("_nemenyi", "_wilcoxon"))
    merged = merged.rename(columns={"significant_nemenyi": "nemenyi_significant",
                                    "significant_wilcoxon": "wilcoxon_significant"})
    merged["agree"] = merged["nemenyi_significant"] == merged["wilcoxon_significant"]
    return merged


def bootstrap_skill_ci(values, n_resamples: int = 2000, alpha: float = 0.05,
                       seed: int = 0) -> dict:
    """Percentile bootstrap interval for a mean skill score."""
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return {"estimate": np.nan, "lower": np.nan, "upper": np.nan,
                "n": 0, "n_resamples": n_resamples}

    rng = np.random.default_rng(seed)
    draws = rng.choice(data, size=(n_resamples, len(data)), replace=True).mean(axis=1)
    lower, upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(data.mean()), "lower": float(lower),
            "upper": float(upper), "n": int(len(data)), "n_resamples": n_resamples}
