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
from tsbench.tuning import declared_space  # noqa: E402
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


def test_to_latex_scales_a_wide_table_to_the_text_width():
    """The cost table carries twelve columns and overflows a portrait page at
    any readable font size, so it has to be scaled rather than shrunk.
    """
    frame = pd.DataFrame({f"c{i}": [1.0] for i in range(12)},
                         index=pd.Index(["naive"], name="model"))

    out = to_latex(frame, caption="Cost", label="tab:cost", fit_width=True)

    assert out.index(r"\resizebox") < out.index(r"\begin{tabular}")
    assert out.index(r"\end{tabular}") < out.rindex("}")


def test_to_latex_leaves_a_narrow_table_unscaled():
    frame = pd.DataFrame({"value": [1.0]}, index=pd.Index(["naive"], name="model"))

    out = to_latex(frame, caption="Profile", label="tab:profile")

    assert r"\resizebox" not in out


def test_to_latex_escapes_angle_brackets():
    """LaTeX's default encoding renders a bare > as an inverted question mark,
    so a threshold like "zero share >= 0.05" comes out as "zero share ?= 0.05".
    """
    frame = pd.DataFrame({"value": ["1"]},
                         index=pd.Index(["zero share >= 0.05"], name="property"))

    out = to_latex(frame, caption="Profile", label="tab:profile")

    assert r"\textgreater{}= 0.05" in out
    assert ">" not in out.replace(r"\textgreater{}", "")


def test_model_roster_prefers_the_version_recorded_by_the_run():
    """The version that produced a result is the one the run recorded, not
    whatever happens to be installed when the table is rebuilt - the foundation
    packages live only on the cluster.
    """
    from tsbench.models.registry import default

    roster = model_roster_table(default(), ["autoarima"],
                                versions={"statsforecast": "9.9.9-from-the-run"})

    assert roster.loc["autoarima", "version"] == "9.9.9-from-the-run"


def test_declared_space_reports_the_search_space_without_running_a_trial(config_dict):
    """The reviewer-facing question is what range each hyperparameter was
    searched over; recording a probe trial answers it without tuning anything.
    """
    from tsbench.config import Config
    from tsbench.models.registry import default

    cfg = Config.from_dict(config_dict)
    spaces = declared_space(default().get("lightgbm_global"), cfg)

    assert spaces["learning_rate"] == "0.01-0.2 log"
    assert spaces["n_estimators"] == "100-600"


def test_declared_space_is_empty_for_a_zero_shot_model(config_dict):
    from tsbench.config import Config
    from tsbench.models.registry import default

    cfg = Config.from_dict(config_dict)

    assert declared_space(default().get("chronos2"), cfg) == {}


# --- the cardinality ladder ----------------------------------------------------

def _rung(n, models, seed=0):
    """A finished rung: per-series MASE rows for 2 folds x 2 repeats x 2
    horizons, timings shaped like the real table, metadata with n_series."""
    ids = [f"S{i:05d}" for i in range(n)]
    metric_rows, timing_rows = [], []
    for model, family, level, fit, predict in models:
        for uid in ids:
            # a series keeps its own error across rungs, as a local or zero-shot
            # model's would; only the level (a function of n) can move
            rng = np.random.default_rng(int(uid[1:]) + seed)
            base = rng.lognormal(np.log(level), 0.3)
            for fold in range(2):
                for repeat in range(2):
                    for horizon in (4, 13):
                        metric_rows.append({
                            "model": model, "family": family, "fold": fold,
                            "repeat": repeat, "horizon": horizon, "unique_id": uid,
                            "metric": "MASE", "value": base * (1 + 0.1 * (horizon == 13)),
                        })
                        metric_rows.append({**metric_rows[-1], "metric": "RMSSE",
                                            "value": base * 0.9})
        for fold in range(2):
            for repeat in range(2):
                for horizon in (4, 13):
                    timing_rows.append({
                        "model": model, "family": family, "fold": fold,
                        "repeat": repeat, "horizon": horizon, "status": "ok",
                        "fit_seconds": fit, "predict_seconds": predict,
                        "fit_reused": horizon == 13, "n_series": n,
                    })
    return {
        "path": f"results/fake-n{n}",
        "meta": {"config": {"sampling": {"n_series": n}, "protocol": {"horizons": [4, 13]}},
                 "git_commit": "abc123", "progress": {}},
        "metrics": pd.DataFrame(metric_rows),
        "timings": pd.DataFrame(timing_rows),
    }


@pytest.fixture
def ladder():
    """A local model whose accuracy does not move with N, a global one that
    improves as N grows, and a zero-shot one that only jitters."""
    from tsbench.report.scaling import scaling_table

    rungs = {}
    for n in (500, 1000, 2000):
        rungs[n] = _rung(n, [
            ("steady", "local", 1.00, 20.0 * n / 1000, 1.0 * n / 1000),
            ("learner", "global", 1.20 * (1000 / n) ** 0.25, 60.0, 0.5 * n / 1000),
            ("zeroshot", "foundation", 1.05, 0.1, 2.0 * n / 1000),
        ])
    return rungs, scaling_table(rungs, band=True, n_boot=50)


def test_scaling_table_normalises_compute_to_1k_series(ladder):
    """The cost table measures the whole rung; the per-1k column has to divide
    that by the rung's size or the curve just restates N."""
    _, table = ladder

    steady = table.xs("steady", level="model")
    # a per-series model costs the same per 1k series at every N
    assert steady["per1k_s_h4"].round(6).nunique() == 1
    assert steady.loc[2000, "run_s_h4"] == pytest.approx(42.0)
    assert steady.loc[2000, "per1k_s_h4"] == pytest.approx(21.0)

    learner = table.xs("learner", level="model")
    # one fit amortised over more series: per-1k compute falls with N
    assert learner["per1k_s_h4"].is_monotonic_decreasing


def test_scaling_table_band_brackets_the_median(ladder):
    _, table = ladder
    assert (table["MASE_h4_lo"] <= table["MASE_h4"]).all()
    assert (table["MASE_h4"] <= table["MASE_h4_hi"]).all()
    # a wider sample gives a tighter interval on its median
    zeroshot = table.xs("zeroshot", level="model")
    widths = zeroshot["MASE_h4_hi"] - zeroshot["MASE_h4_lo"]
    assert widths.loc[2000] < widths.loc[500]


def test_crossovers_report_the_first_rung_below_the_reference(ladder):
    from tsbench.report.scaling import crossovers

    _, table = ladder
    steady_level = {4: float(table.loc[("steady", 1000), "MASE_h4"]),
                    13: float(table.loc[("steady", 1000), "MASE_h13"])}
    cross = crossovers(table, steady_level, horizons=(4, 13))

    learner = cross.loc[("learner", 4)]
    learner_curve = table.xs("learner", level="model")["MASE_h4"]
    below = learner_curve.index[learner_curve < steady_level[4]]
    if len(below):
        assert learner.first_n_below == below[0]
    else:
        assert pd.isna(learner.first_n_below)
        assert learner.max_n == 2000
    # a model that never dips under the line says so rather than guessing
    assert pd.isna(cross.loc[("zeroshot", 4), "first_n_below"])
    assert cross.loc[("zeroshot", 4), "gap_at_max_pct"] > 0


def test_noise_band_is_the_zero_shot_spread_across_rungs(ladder):
    from tsbench.report.scaling import noise_band

    _, table = ladder
    lo, hi = noise_band(table, "zeroshot", "MASE", 4)
    values = table.xs("zeroshot", level="model")["MASE_h4"]
    assert (lo, hi) == (values.min(), values.max())
    assert noise_band(table, "absent", "MASE", 4) is None


def test_scaling_figure_has_a_panel_pair_per_horizon(ladder, tmp_path):
    from tsbench.report.scaling import scaling_figure

    _, table = ladder
    fig = scaling_figure(table, horizons=(4, 13), reference={4: 1.0, 13: 1.1},
                         reference_label="steady at N=1,000", noise_model="zeroshot")
    assert len(fig.axes) == 4
    assert fig.axes[0].get_xscale() == "log"
    # shared x: only the bottom row carries tick labels, one per rung
    fig.canvas.draw()
    bottom_left = fig.axes[2]
    assert [t.get_text() for t in bottom_left.get_xticklabels()] == ["500", "1,000", "2,000"]
    written = save_figure(fig, tmp_path / "F9")
    assert all(p.exists() for p in written)
    plt.close(fig)


def test_ladder_warnings_name_mixed_builds_and_missing_models():
    from tsbench.report.scaling import ladder_warnings

    a = _rung(500, [("steady", "local", 1.0, 1.0, 1.0)])
    b = _rung(1000, [("steady", "local", 1.0, 1.0, 1.0), ("learner", "global", 1.0, 1.0, 1.0)])
    b["meta"]["git_commit"] = "fff999"
    b["meta"]["progress"] = {"learner": {"expected": 5, "complete": 4, "failed": 1}}

    notes = ladder_warnings({500: a, 1000: b})

    assert any("different builds" in n for n in notes)
    assert any("N=500 lacks learner" in n for n in notes)
    assert any("learner: 4/5 units, 1 failed" in n for n in notes)


def test_common_series_is_the_intersection_of_the_rungs(ladder):
    from tsbench.report.scaling import common_series

    rungs, _ = ladder
    shared = common_series(rungs)
    assert len(shared) == 500
    assert set(shared) == set(rungs[500]["metrics"]["unique_id"])


def test_fixed_evaluation_set_holds_local_and_zero_shot_models_flat(ladder):
    """On the same series a model that does not learn across series repeats
    itself at every N; only the global model can move."""
    from tsbench.report.scaling import common_series, scaling_table

    rungs, _ = ladder
    fixed = scaling_table(rungs, band=False, series=common_series(rungs))

    assert (fixed["eval_series"] == 500).all()
    for model in ("steady", "zeroshot"):
        assert fixed.xs(model, level="model")["MASE_h4"].round(9).nunique() == 1
    learner = fixed.xs("learner", level="model")["MASE_h4"]
    assert learner.is_monotonic_decreasing and learner.iloc[-1] < learner.iloc[0]


def test_reference_levels_can_be_taken_over_the_fixed_set(ladder):
    from tsbench.report.scaling import common_series, reference_levels

    rungs, _ = ladder
    metrics = rungs[2000]["metrics"]
    on_all = reference_levels(metrics, "steady", horizons=(4,))[4]
    on_shared = reference_levels(metrics, "steady", horizons=(4,),
                                 series=common_series(rungs))[4]
    expected = (metrics[(metrics["model"] == "steady") & (metrics["metric"] == "MASE")
                        & (metrics["horizon"] == 4)
                        & metrics["unique_id"].isin(common_series(rungs))]["value"].median())
    assert on_shared == pytest.approx(expected)
    assert on_shared != on_all


def test_scaling_figure_without_cost_has_one_row(ladder, tmp_path):
    from tsbench.report.scaling import scaling_figure

    _, table = ladder
    fig = scaling_figure(table, horizons=(4, 13), reference=None, noise_model="zeroshot",
                         cost=False, title="full sample")
    assert len(fig.axes) == 2
    fig.canvas.draw()
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["500", "1,000", "2,000"]
    plt.close(fig)


def test_paired_tests_separate_a_learner_from_a_model_that_cannot_move(ladder):
    from tsbench.report.scaling import common_series, paired_tests

    rungs, _ = ladder
    shared = common_series(rungs)
    paired = paired_tests(rungs, shared, horizons=(4,),
                          reference=rungs[2000]["metrics"], reference_model="steady")

    growth = paired.xs("N=2,000 vs N=500", level="comparison")
    # the same forecasts at both sizes: nothing to test, p is 1 by construction
    assert growth.loc[("steady", 4), "p_value"] == 1.0
    assert growth.loc[("zeroshot", 4), "median_paired_diff"] == 0.0
    # the learner improved on most series, and the test sees it
    learner = growth.loc[("learner", 4)]
    assert learner["median_paired_diff"] < 0
    assert learner["share_first_better"] > 0.5
    assert learner["p_value"] < 0.01
    assert learner["n_series"] == 500

    against = paired.xs("N=2,000 vs steady", level="comparison")
    assert against.loc[("steady", 4), "p_value"] == 1.0
    assert set(against.index.get_level_values("model")) == {"steady", "learner", "zeroshot"}


def test_cost_table_takes_the_trial_count_from_the_recorded_budget(timings):
    """Checkpoints from a build that applied the budget uniformly annotate every
    row with 20 trials; the recorded budget says what was actually searched."""
    with_annotation = cost_per_1k_table(timings)
    assert with_annotation.loc["cheap", "tuning_trials"] == 20

    corrected = cost_per_1k_table(timings, budget={"steady": 20})
    assert corrected.loc["steady", "tuning_trials"] == 20
    assert corrected.loc["cheap", "tuning_trials"] == 0
    assert corrected["tuning_trials"].dtype.kind == "i"
