"""Equal-budget hyperparameter tuning.

The most scrutinised part of a benchmark paper: every tunable model gets the
same number of trials, that number appears in every output row, and zero-shot
models report 0 rather than a blank. Tuning happens strictly before the first
forecast origin, so no test window is ever used to choose a hyperparameter.
"""
import numpy as np
import pandas as pd
import pytest

from tsbench.config import Config
from tsbench.data.loader import normalize_panel
from tsbench.models import registry as registry_module
from tsbench.models.base import ModelAdapter
from tsbench.tuning import tune_all, tuning_origin


class Tunable(ModelAdapter):
    name = "tunable"
    family = "local"
    seen_max_ds = []

    def tuning_space(self, trial):
        return {"scale": trial.suggest_float("scale", 0.5, 2.0)}

    def fit(self, train_df):
        Tunable.seen_max_ds.append(train_df["ds"].max())
        self._ids = sorted(train_df["unique_id"].unique())
        self._last = train_df["ds"].max()
        self._level = train_df.groupby("unique_id")["y"].last()

    def predict(self, horizon):
        ds = pd.date_range(self._last + pd.Timedelta(days=7), periods=horizon, freq="7D")
        scale = self.params.get("scale", 1.0)
        return pd.DataFrame({
            "unique_id": np.repeat(self._ids, horizon),
            "ds": np.tile(ds.to_numpy(), len(self._ids)),
            "yhat": np.repeat(self._level.loc[self._ids].to_numpy() * scale, horizon),
        })


class NotTunable(Tunable):
    name = "not_tunable"
    family = "foundation"
    tunable = False

    def tuning_space(self, trial):
        return {}


class Flaky(Tunable):
    name = "flaky"
    calls = []

    def fit(self, train_df):
        Flaky.calls.append(1)
        if len(Flaky.calls) % 2 == 0:
            raise RuntimeError("trial blew up")
        super().fit(train_df)


@pytest.fixture(autouse=True)
def reset():
    Tunable.seen_max_ds = []
    Flaky.calls = []


@pytest.fixture
def cfg(config_dict, tmp_path):
    config_dict["run"]["output_dir"] = str(tmp_path)
    config_dict["tuning"]["budget_trials"] = 4
    return Config.from_dict(config_dict)


@pytest.fixture
def panel(raw_frame, cfg):
    return normalize_panel(raw_frame, cfg)


@pytest.fixture
def reg():
    r = registry_module.Registry()
    for adapter in (Tunable, NotTunable, Flaky):
        r.register(adapter)
    return r


def test_tuning_origin_sits_before_the_first_forecast_origin(cfg, panel):
    from tsbench.eval.splitter import RollingOriginSplitter

    first = RollingOriginSplitter(cfg).origins(panel)[0]

    assert tuning_origin(panel, cfg) < first


def test_every_tunable_model_gets_the_same_budget(cfg, panel, reg):
    result = tune_all(panel, cfg, ["tunable", "flaky"], registry=reg)

    for name in ("tunable", "flaky"):
        assert result.trials[result.trials["model"] == name]["trial"].nunique() == 4
        assert result.budget[name] == 4


def test_zero_shot_models_get_no_trials_and_report_zero(cfg, panel, reg):
    result = tune_all(panel, cfg, ["not_tunable"], registry=reg)

    assert result.budget["not_tunable"] == 0
    assert result.best["not_tunable"] == {}
    assert result.trials[result.trials["model"] == "not_tunable"].empty


def test_tuning_never_sees_data_at_or_after_the_tuning_origin(cfg, panel, reg):
    tune_all(panel, cfg, ["tunable"], registry=reg)
    origin = tuning_origin(panel, cfg)

    assert Tunable.seen_max_ds
    assert all(ds <= origin for ds in Tunable.seen_max_ds)


def test_the_same_seed_reproduces_the_same_choice(cfg, panel, reg):
    first = tune_all(panel, cfg, ["tunable"], registry=reg)
    second = tune_all(panel, cfg, ["tunable"], registry=reg)

    assert first.best["tunable"] == second.best["tunable"]


def test_a_failing_trial_does_not_abort_the_search(cfg, panel, reg):
    result = tune_all(panel, cfg, ["flaky"], registry=reg)
    trials = result.trials[result.trials["model"] == "flaky"]

    assert (trials["status"] == "failed").any()
    assert (trials["status"] == "ok").any()
    assert result.best["flaky"], "a best configuration was still found"


def test_the_trial_record_is_written_for_the_paper(cfg, panel, reg, tmp_path):
    result = tune_all(panel, cfg, ["tunable"], registry=reg)
    saved = pd.read_parquet(result.path)

    assert set(saved.columns) >= {"model", "trial", "value", "params", "status",
                                  "config_hash", "seed", "objective"}
    assert (saved["config_hash"] == cfg.hash).all()


def test_best_params_beat_the_worst_trial(cfg, panel, reg):
    result = tune_all(panel, cfg, ["tunable"], registry=reg)
    trials = result.trials.query("model == 'tunable' and status == 'ok'")

    assert result.best_value["tunable"] == pytest.approx(trials["value"].min())
