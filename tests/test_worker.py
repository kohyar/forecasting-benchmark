"""Running each (model, fold) in its own process.

Two reasons, one of them not optional: adapters pull in libraries that cannot
coexist in a single process (lightgbm and torch each bundle an OpenMP runtime
and segfault when both are exercised), and a segfault or an OOM kill cannot be
caught by try/except - only by watching a child process exit.
"""
import numpy as np
import pandas as pd
import pytest

from tsbench.config import Config
from tsbench.data.loader import normalize_panel
from tsbench.models import registry
from tsbench.models.base import ModelAdapter
from tsbench.runner import BenchmarkRunner

pytestmark = pytest.mark.slow


class Constant(ModelAdapter):
    name = "constant"
    family = "baseline"

    def fit(self, train_df):
        self._ids = sorted(train_df["unique_id"].unique())
        self._last = train_df["ds"].max()

    def predict(self, horizon):
        ds = pd.date_range(self._last + pd.Timedelta(days=7), periods=horizon, freq="7D")
        return pd.DataFrame({
            "unique_id": np.repeat(self._ids, horizon),
            "ds": np.tile(ds.to_numpy(), len(self._ids)),
            "yhat": 5.0,
        })


class HardCrash(Constant):
    """Dies the way a real OOM or a segfault does - no exception to catch."""

    name = "hard_crash"

    def fit(self, train_df):
        import os
        os._exit(139)


@pytest.fixture
def cfg(config_dict, tmp_path):
    config_dict["run"]["output_dir"] = str(tmp_path)
    config_dict["run"]["n_repeats"] = 1
    config_dict["run"]["execution"] = "subprocess"
    return Config.from_dict(config_dict)


@pytest.fixture
def panel(raw_frame, cfg):
    return normalize_panel(raw_frame, cfg)


def test_subprocess_execution_matches_in_process_results(config_dict, panel, tmp_path):
    config_dict["run"]["output_dir"] = str(tmp_path / "a")
    config_dict["run"]["n_repeats"] = 1
    config_dict["run"]["execution"] = "inprocess"
    in_proc = BenchmarkRunner(Config.from_dict(config_dict)).run(panel, models=["naive"])

    config_dict["run"]["output_dir"] = str(tmp_path / "b")
    config_dict["run"]["execution"] = "subprocess"
    sub = BenchmarkRunner(Config.from_dict(config_dict)).run(panel, models=["naive"])

    key = ["model", "fold", "horizon", "unique_id", "metric"]
    a = in_proc.metrics.sort_values(key).reset_index(drop=True)
    b = sub.metrics.sort_values(key).reset_index(drop=True)

    pd.testing.assert_series_equal(a["value"], b["value"], check_exact=False, rtol=1e-9)


def test_a_hard_crash_is_recorded_and_the_run_continues(cfg, panel):
    reg = registry.Registry()
    reg.register(HardCrash)
    reg.register(Constant)

    result = BenchmarkRunner(cfg, registry=reg).run(panel, models=["hard_crash", "constant"])

    failed = result.timings[result.timings["status"] == "failed"]
    assert len(failed) > 0
    assert "hard_crash" in set(failed["model"])
    assert "139" in failed.iloc[0]["error"] or "exit" in failed.iloc[0]["error"].lower()
    assert "constant" in set(result.timings[result.timings["status"] == "ok"]["model"])


def test_a_crash_contributes_no_metric_rows(cfg, panel):
    reg = registry.Registry()
    reg.register(HardCrash)
    reg.register(Constant)

    result = BenchmarkRunner(cfg, registry=reg).run(panel, models=["hard_crash", "constant"])

    assert "hard_crash" not in set(result.metrics["model"])
    assert "constant" in set(result.metrics["model"])


def test_the_child_loads_only_its_own_adapter_module(cfg, panel):
    """The whole point: a lightgbm child must not import torch, and vice versa."""
    result = BenchmarkRunner(cfg).run(panel, models=["naive"])
    row = result.timings[result.timings["status"] == "ok"].iloc[0]
    loaded = row["worker_modules"].split(",")

    assert row["worker_adapter_module"].endswith("statsforecast_adapters")
    assert "torch" not in loaded
    assert "lightgbm" not in loaded


def test_execution_mode_is_recorded_in_the_metadata(cfg, panel):
    meta = BenchmarkRunner(cfg).run(panel, models=["naive"]).metadata

    assert meta["execution"] == "subprocess"
