"""Turning the long results file into the paper's tables.

The schema exists so a groupby produces a publication table directly. These
tests pin that promise down.
"""
import numpy as np
import pandas as pd
import pytest

from tsbench.config import Config
from tsbench.stats.aggregate import (
    accuracy_table,
    blocks_for_testing,
    cost_table,
    critical_difference_diagram,
    skill_scores,
)


@pytest.fixture
def metrics():
    rng = np.random.default_rng(0)
    rows = []
    for model, level in [("naive", 1.2), ("seasonal_naive", 1.0), ("autoets", 0.8)]:
        for horizon in (4, 13):
            for fold in range(3):
                for repeat in range(2):
                    for series in range(10):
                        rows.append({
                            "model": model, "family": "local", "fold": fold,
                            "horizon": horizon, "repeat": repeat,
                            "unique_id": f"S{series}", "metric": "MASE",
                            "value": level + rng.normal(0, 0.05),
                            "n_points": horizon, "n_undefined": 0,
                            "seed": 42 + repeat, "config_hash": "abc",
                            "run_name": "t", "tuning_trials": 20,
                        })
    return pd.DataFrame(rows)


@pytest.fixture
def timings():
    rows = []
    for model, fit, predict in [("naive", 1.0, 0.2), ("autoets", 30.0, 0.5),
                                ("chronos2", 0.0, 12.0)]:
        for fold in range(3):
            for horizon in (4, 13):
                rows.append({
                    "model": model,
                    "family": "foundation" if model == "chronos2" else "local",
                    "fold": fold, "horizon": horizon, "repeat": 0,
                    "fit_seconds": fit, "predict_seconds": predict,
                    "peak_memory_mb": 100.0, "n_series": 10, "n_params": None,
                    "device": "cpu", "status": "ok",
                    "fit_reused": horizon == 13,
                    "fit_key": f"{model}|f{fold}|r0|hall",
                    "tuning_trials": 0 if model == "chronos2" else 20,
                })
    return pd.DataFrame(rows)


def test_accuracy_table_is_models_by_horizon(metrics):
    table = accuracy_table(metrics, metric="MASE")

    assert list(table.index.names) == ["model"]
    assert set(table.columns.get_level_values(0)) == {4, 13}
    assert "median" in set(table.columns.get_level_values(1))
    assert table.loc["autoets", (4, "median")] < table.loc["naive", (4, "median")]


def test_accuracy_table_reports_spread_across_repeats(metrics):
    table = accuracy_table(metrics, metric="MASE")

    assert "std" in set(table.columns.get_level_values(1))
    assert (table.xs("std", level=1, axis=1) >= 0).all().all()


def test_skill_scores_are_relative_to_the_named_baseline(metrics):
    skill = skill_scores(metrics, baseline="seasonal_naive", metric="MASE")

    baseline = skill[skill["model"] == "seasonal_naive"]["skill"]
    assert baseline.abs().max() < 1e-9, "the baseline scores zero against itself"
    assert skill[skill["model"] == "autoets"]["skill"].mean() > 0, "better than baseline"
    assert skill[skill["model"] == "naive"]["skill"].mean() < 0


def test_cost_table_keeps_fit_and_predict_apart(timings):
    table = cost_table(timings)

    assert table.loc["chronos2", "fit_seconds_total"] == 0.0
    assert table.loc["chronos2", "predict_seconds_total"] > 0
    assert table.loc["autoets", "fit_seconds_total"] > 0


def test_cost_table_does_not_double_count_a_reused_fit(timings):
    table = cost_table(timings)

    # three folds, one fit each, 30s per fit
    assert table.loc["autoets", "fit_seconds_total"] == pytest.approx(90.0)


def test_cost_table_reports_dollars_when_a_rate_is_configured(timings, config_dict):
    config_dict["cost"] = {"usd_per_hour": 3.6, "instance_type": "g5.xlarge",
                           "runtime_version": "15.4-ml"}
    cfg = Config.from_dict(config_dict)

    table = cost_table(timings, cfg=cfg)

    assert "usd_per_1k_series" in table.columns
    assert table.loc["autoets", "usd_per_1k_series"] > 0


def test_cost_table_omits_dollars_when_no_rate_is_known(timings):
    table = cost_table(timings)

    assert "usd_per_1k_series" not in table.columns


def test_cost_table_carries_the_tuning_budget(timings):
    table = cost_table(timings)

    assert table.loc["chronos2", "tuning_trials"] == 0
    assert table.loc["autoets", "tuning_trials"] == 20


def test_blocks_matrix_is_series_by_model(metrics):
    blocks = blocks_for_testing(metrics, metric="MASE", horizon=4)

    assert set(blocks.columns) == {"naive", "seasonal_naive", "autoets"}
    assert len(blocks) == 10
    assert blocks.notna().all().all()


def test_blocks_average_over_repeats_not_over_series(metrics):
    blocks = blocks_for_testing(metrics, metric="MASE", horizon=4)
    manual = (metrics.query("model == 'autoets' and horizon == 4")
              .groupby("unique_id")["value"].mean())

    pd.testing.assert_series_equal(blocks["autoets"].sort_index(),
                                   manual.sort_index(), check_names=False)


def test_ablation_arms_are_separate_models_not_averaged_together(metrics):
    """Otherwise a groupby silently merges the with- and without-covariate
    arms into one number."""
    from tsbench.stats.aggregate import label_ablation_arms

    metrics = metrics.copy()
    metrics["covariates"] = False
    with_cov = metrics[metrics["model"] == "autoets"].copy()
    with_cov["covariates"] = True
    with_cov["value"] = with_cov["value"] * 0.5
    both = pd.concat([metrics, with_cov], ignore_index=True)

    labelled = label_ablation_arms(both)
    table = accuracy_table(labelled, metric="MASE")

    assert "autoets+cov" in table.index
    assert "autoets" in table.index
    assert table.loc["autoets+cov", (4, "median")] < table.loc["autoets", (4, "median")]


def test_labelling_leaves_a_single_arm_run_untouched(metrics):
    from tsbench.stats.aggregate import label_ablation_arms

    metrics = metrics.assign(covariates=False)

    assert set(label_ablation_arms(metrics)["model"]) == set(metrics["model"])


def test_critical_difference_diagram_is_written(metrics, tmp_path):
    blocks = blocks_for_testing(metrics, metric="MASE", horizon=4)
    path = tmp_path / "cd.png"

    critical_difference_diagram(blocks, path)

    assert path.exists() and path.stat().st_size > 1000
