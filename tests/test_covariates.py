"""The covariate ablation must actually change something.

A model that declares covariate support but ignores it would produce two
identical arms and an ablation that proves nothing.

Runs through the worker, like everything else: fitting lightgbm in a process
that has already imported torch segfaults.
"""
import pandas as pd
import pytest

from tsbench.config import Config
from tsbench.data.loader import normalize_panel
from tsbench.models import registry as registry_module
from tsbench.runner import BenchmarkRunner

pytestmark = pytest.mark.slow

FAST = {"max_steps": 3, "input_size": 8}


def _capable(cfg):
    reg = registry_module.default()
    return [n for n in reg.available_names() if reg.get(n)(cfg).supports_covariates()]


@pytest.fixture
def cfg(config_dict, tmp_path):
    config_dict["run"]["output_dir"] = str(tmp_path)
    config_dict["run"]["n_repeats"] = 1
    config_dict["run"]["execution"] = "subprocess"
    config_dict["data"]["season_length"] = 4
    config_dict["models"]["covariate_ablation"] = True
    return Config.from_dict(config_dict)


@pytest.fixture
def panel(cfg):
    """Needs more than one year of history: the _Yago covariates are 52-week
    lags, so a shorter fixture would hand every model a column of zeros."""
    import numpy as np

    from tests.conftest import raw_rows, weeks

    ds = weeks("2022-01-09", 130)
    rng = np.random.default_rng(0)
    season = 40 * np.sin(2 * np.pi * np.arange(130) / 52)
    frames = [
        raw_rows("SUB_A", "BOSTON, MA - MULO", ds, 500 + season + rng.normal(0, 5, 130)),
        raw_rows("SUB_B", "ALBANY, NY - MULO", ds, 300 + season + rng.normal(0, 5, 130)),
        raw_rows("SUB_C", "CHICAGO, IL - MULO", ds,
                 200 + np.arange(130) * 1.5 + rng.normal(0, 5, 130)),
    ]
    return normalize_panel(pd.concat(frames, ignore_index=True), cfg)


def test_at_least_one_model_declares_covariate_support(cfg):
    assert _capable(cfg), "the ablation needs a capable model"


def test_each_capable_model_runs_both_arms_and_they_differ(cfg, panel):
    """Same model, same data, covariates the only change."""
    for name in _capable(cfg):
        result = BenchmarkRunner(cfg).run(panel, models=[name],
                                          tuned_params={name: dict(FAST)})
        failed = result.timings[result.timings["status"] == "failed"]
        assert failed.empty, f"{name}: {failed.iloc[0]['error'] if len(failed) else ''}"
        assert set(result.timings["covariates"]) == {False, True}, name

        mase = (result.metrics[result.metrics["metric"] == "MASE"]
                .groupby("covariates")["value"].mean())
        assert not mase.isna().any(), name
        assert abs(mase[True] - mase[False]) > 1e-9, f"{name} ignores its covariates"


def test_the_arms_are_distinguishable_in_the_output(cfg, panel):
    name = _capable(cfg)[0]
    result = BenchmarkRunner(cfg).run(panel, models=[name],
                                      tuned_params={name: dict(FAST)})

    assert "covariates" in result.metrics.columns
    assert "covariates" in result.timings.columns
    key = ["model", "covariates", "fold", "horizon", "unique_id", "metric"]
    assert not result.metrics.duplicated(key + ["repeat"]).any()
