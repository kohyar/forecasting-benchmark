"""Turning the results file into the paper's figures and tables.

The cost unit is the subtle part: a fit is shared across horizons and repeated
for timing stability, so neither may be summed into a cost claim. These tests
pin that down, along with the frontier the signature figure draws.
"""
import matplotlib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tsbench.report.figures import (  # noqa: E402
    accuracy_vs_cost,
    models_tied_with_best,
    pareto_frontier,
)
from tsbench.report.style import FAMILIES, family_style, save_figure  # noqa: E402
from tsbench.report.tables import (  # noqa: E402
    cost_per_1k_table,
    dataset_profile_table,
    model_roster_table,
    main_accuracy_table,
    ranking_flip_table,
    to_latex,
)


@pytest.fixture
def timings():
    """Two models over 3 folds x 3 repeats x 2 horizons, shaped like the real
    table: one fit per (fold, repeat), reused by the second horizon.
    """
    rows = []
    for model, family, fit, predict in [("steady", "local", 100.0, 2.0),
                                        ("cheap", "foundation", 1.0, 0.5)]:
        for fold in range(3):
            for repeat in range(3):
                for horizon in (4, 13):
                    rows.append({
                        "model": model, "family": family, "fold": fold,
                        "repeat": repeat, "horizon": horizon,
                        "fit_seconds": fit, "predict_seconds": predict,
                        "fit_reused": horizon == 13,
                        "fit_key": f"{model}|f{fold}|r{repeat}|hall",
                        "peak_memory_mb": 100.0, "n_params": None,
                        "tuning_trials": 20, "n_series": 1000,
                        "status": "ok",
                    })
    return pd.DataFrame(rows)


def test_cost_per_1k_counts_one_fit_not_one_per_repeat(timings):
    """Repeats measure the same work three times for timing stability. Summing
    them would inflate the cost of running the model once by 3x.
    """
    cost = cost_per_1k_table(timings)

    assert cost.loc["steady", "fit_s"] == pytest.approx(100.0)


def test_cost_per_1k_counts_a_shared_fit_once(timings):
    """One fit serves both horizons, so the h=4 cost is fit + its own predict -
    not fit counted again for the second horizon.
    """
    cost = cost_per_1k_table(timings)

    assert cost.loc["steady", "per1k_s_h4"] == pytest.approx(102.0)


def test_cost_per_1k_relative_column_is_multiples_of_the_cheapest(timings):
    """The ratio is the hardware-robust claim; absolute seconds are not."""
    cost = cost_per_1k_table(timings)

    assert cost.loc["cheap", "rel_h4"] == pytest.approx(1.0)
    assert cost.loc["steady", "rel_h4"] == pytest.approx(102.0 / 1.5)


def test_pareto_frontier_drops_points_beaten_on_both_axes():
    """Minimising both cost and error: a point survives unless something else
    is at least as good on both and strictly better on one.
    """
    cost = [1.0, 2.0, 3.0, 0.5]
    error = [0.5, 0.6, 0.4, 0.9]

    assert list(pareto_frontier(cost, error)) == [True, False, True, True]


def test_pareto_frontier_keeps_the_cheaper_of_two_equal_error_points():
    cost = [1.0, 2.0]
    error = [0.5, 0.5]

    assert list(pareto_frontier(cost, error)) == [True, False]


@pytest.fixture
def flip_metrics():
    """Three models whose order under MASE is exactly reversed under sMAPE,
    plus one series outside the intermittent set to prove it is excluded.
    """
    rows = []
    scores = {"alpha": {"MASE": 0.5, "sMAPE": 30.0},
              "beta": {"MASE": 1.0, "sMAPE": 20.0},
              "gamma": {"MASE": 1.5, "sMAPE": 10.0}}
    for model, by_metric in scores.items():
        for metric, value in by_metric.items():
            for series in ("S0", "S1"):
                rows.append({"model": model, "unique_id": series, "horizon": 4,
                             "fold": 0, "repeat": 0, "metric": metric, "value": value})
            # a dense series with values that would reverse the ranking if used
            rows.append({"model": model, "unique_id": "DENSE", "horizon": 4,
                         "fold": 0, "repeat": 0, "metric": metric,
                         "value": 100.0 - by_metric[metric]})
    return pd.DataFrame(rows)


def test_ranking_flip_reports_both_ranks_for_the_same_models(flip_metrics):
    """The §4 argument is that the choice of metric, not the models, decides
    the ordering - so both rankings must come from one set of series.
    """
    table = ranking_flip_table(flip_metrics, series_ids=["S0", "S1"], horizon=4)

    assert table.loc["alpha", "rank_MASE"] == 1
    assert table.loc["alpha", "rank_sMAPE"] == 3


def test_ranking_flip_delta_is_positive_when_a_model_falls_under_smape(flip_metrics):
    table = ranking_flip_table(flip_metrics, series_ids=["S0", "S1"], horizon=4)

    assert table.loc["alpha", "rank_delta"] == 2
    assert table.loc["gamma", "rank_delta"] == -2


def test_ranking_flip_ignores_series_outside_the_given_set(flip_metrics):
    """DENSE would invert every ranking; scoping to series_ids must exclude it."""
    table = ranking_flip_table(flip_metrics, series_ids=["S0", "S1"], horizon=4)

    assert table.loc["alpha", "MASE"] == pytest.approx(0.5)


def test_to_latex_escapes_underscores_in_model_names():
    """Every second model is named like lightgbm_global; an unescaped
    underscore is a LaTeX compile error, not a typo in the PDF.
    """
    frame = pd.DataFrame({"value": [1.0]}, index=pd.Index(["lightgbm_global"], name="model"))

    out = to_latex(frame, caption="Cost", label="tab:cost")

    assert r"lightgbm\_global" in out
    assert "lightgbm_global" not in out.replace(r"lightgbm\_global", "")


def test_to_latex_uses_booktabs_rules_and_carries_caption_and_label():
    frame = pd.DataFrame({"value": [1.0]}, index=pd.Index(["naive"], name="model"))

    out = to_latex(frame, caption="Main accuracy", label="tab:accuracy")

    assert r"\toprule" in out and r"\midrule" in out and r"\bottomrule" in out
    assert "Main accuracy" in out and r"\label{tab:accuracy}" in out


def test_every_family_gets_a_distinct_colour():
    """Four families share one scatter, so the hues must be separable."""
    colours = {family_style(f)["color"] for f in FAMILIES}

    assert len(colours) == len(FAMILIES)


def test_every_family_gets_a_distinct_marker():
    """Colour alone disappears in a greyscale print edition; the marker is the
    redundant channel that keeps the figure readable there.
    """
    markers = {family_style(f)["marker"] for f in FAMILIES}

    assert len(markers) == len(FAMILIES)


def test_save_figure_writes_a_vector_and_a_raster_copy(tmp_path):
    """LaTeX wants the PDF; reviewers and slides want the PNG."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    save_figure(fig, tmp_path / "f1")
    plt.close(fig)

    assert (tmp_path / "f1.pdf").stat().st_size > 0
    assert (tmp_path / "f1.png").stat().st_size > 0


def test_save_figure_rasterises_at_print_resolution(tmp_path):
    """The submission checklist requires >=300 dpi.

    PNG stores resolution as whole pixels-per-metre, so 300 dpi comes back as
    299.9994 - round before comparing rather than reading it as a shortfall.
    """
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    save_figure(fig, tmp_path / "f1")
    plt.close(fig)

    assert round(Image.open(tmp_path / "f1.png").info["dpi"][0]) >= 300


def test_main_accuracy_table_has_one_column_per_metric_and_horizon(flip_metrics):
    """T7 carries the whole accuracy set side by side, so one call must cover
    every metric rather than one table per metric.
    """
    table = main_accuracy_table(flip_metrics, metric_names=("MASE", "sMAPE"),
                                horizons=(4,))

    assert list(table.columns) == ["MASE_h4", "sMAPE_h4"]


def test_main_accuracy_table_reports_the_median_over_series(flip_metrics):
    """alpha scores 0.5 on both real series and 99.5 on the dense one; the
    median is the statistic the console output already reports.
    """
    table = main_accuracy_table(flip_metrics, metric_names=("MASE",), horizons=(4,))

    assert table.loc["alpha", "MASE_h4"] == pytest.approx(0.5)


def test_accuracy_vs_cost_joins_frames_that_both_label_the_family(timings):
    """Both tables carry a `family` column, so a naive join collides on it."""
    cost = cost_per_1k_table(timings)
    accuracy = pd.DataFrame(
        {"MASE_h4": [0.5, 1.0], "family": ["local", "foundation"]},
        index=pd.Index(["steady", "cheap"], name="model"))

    fig = accuracy_vs_cost(cost, accuracy, horizon=4)
    plt.close(fig)


def test_model_roster_names_the_package_behind_every_model():
    """T4 has to say what each result was actually produced by; a blank
    implementation column is what reviewers ask about.
    """
    from tsbench.models.registry import default

    registry = default()
    roster = model_roster_table(registry, ["naive", "autoarima", "tft",
                                           "lightgbm_global", "prophet"])

    assert (roster["package"].str.len() > 0).all()


def test_dataset_profile_summarises_the_sampled_panel():
    """T3 describes what was actually benchmarked - the sample, not the raw
    export it was drawn from.
    """
    profile = pd.DataFrame({
        "unique_id": ["a", "b", "c"],
        "n_obs": [232, 232, 200],
        "zero_share": [0.0, 0.10, 0.30],
        "total_volume": [100.0, 200.0, 300.0],
    })

    table = dataset_profile_table(profile, zero_share_bins=[0.0, 0.05, 0.2, 1.01])

    assert table.loc["series", "value"] == "3"
    assert table.loc["intermittent series (zero share >= 0.05)", "value"] == "2"


def test_tie_region_holds_models_the_test_cannot_separate_from_the_best():
    """Diebold-Mariano cannot separate the top three, so a frontier drawn as if
    the ranking were real would overstate the result.
    """
    scores = pd.Series({"a": 0.60, "b": 0.61, "c": 0.80})
    dm = pd.DataFrame([
        {"model_a": "a", "model_b": "b", "p_value": 0.40},
        {"model_a": "a", "model_b": "c", "p_value": 0.001},
        {"model_a": "b", "model_b": "c", "p_value": 0.002},
    ])

    assert models_tied_with_best(scores, dm) == {"a", "b"}


def test_tie_region_is_just_the_best_model_when_everything_separates():
    scores = pd.Series({"a": 0.60, "b": 0.80})
    dm = pd.DataFrame([{"model_a": "a", "model_b": "b", "p_value": 0.001}])

    assert models_tied_with_best(scores, dm) == {"a"}
