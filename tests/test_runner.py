"""The runner: execute (model, fold, horizon), measure it, survive failures,
and make every row traceable.
"""
import numpy as np
import pandas as pd
import pytest

from tsbench.config import Config
from tsbench.data.loader import normalize_panel
from tsbench.models import registry
from tsbench.models.base import ModelAdapter
from tsbench.runner import BenchmarkRunner


@pytest.fixture
def cfg(config_dict, tmp_path):
    config_dict["run"]["output_dir"] = str(tmp_path)
    config_dict["run"]["n_repeats"] = 2
    # These tests use adapters defined inside the test module, which a worker
    # process cannot import; subprocess execution is covered in test_worker.py.
    config_dict["run"]["execution"] = "inprocess"
    return Config.from_dict(config_dict)


@pytest.fixture
def panel(raw_frame, cfg):
    return normalize_panel(raw_frame, cfg)


class Constant(ModelAdapter):
    name = "constant"
    family = "baseline"

    def fit(self, train_df):
        self._ids = sorted(train_df["unique_id"].unique())
        self._last = train_df["ds"].max()
        self._seen_max_ds = train_df["ds"].max()

    def predict(self, horizon):
        ds = pd.date_range(self._last + pd.Timedelta(days=7), periods=horizon, freq="7D")
        return pd.DataFrame({
            "unique_id": np.repeat(self._ids, horizon),
            "ds": np.tile(ds.to_numpy(), len(self._ids)),
            "yhat": 5.0,
        })


class Exploding(Constant):
    name = "exploding"

    def fit(self, train_df):
        raise MemoryError("CUDA out of memory")


class Malformed(Constant):
    name = "malformed"

    def predict(self, horizon):
        return super().predict(horizon).head(1)


class Refitting(Constant):
    name = "refitting"
    horizon_is_fit_time = True


@pytest.fixture
def reg():
    r = registry.Registry()
    for adapter in (Constant, Exploding, Malformed, Refitting):
        r.register(adapter)
    return r


@pytest.fixture
def runner(cfg, reg):
    return BenchmarkRunner(cfg, registry=reg)


def test_produces_metric_rows_for_every_model_fold_horizon_series(runner, panel):
    result = runner.run(panel, models=["constant"])
    m = result.metrics

    assert set(m["model"]) == {"constant"}
    assert set(m["fold"]) == {0, 1}
    assert set(m["horizon"]) == {1, 3}
    assert set(m["unique_id"]) == set(panel["unique_id"])
    assert "MASE" in set(m["metric"])


def test_timings_keep_fit_and_predict_apart(runner, panel):
    t = runner.run(panel, models=["constant"]).timings

    assert {"fit_seconds", "predict_seconds"} <= set(t.columns)
    assert (t["fit_seconds"] >= 0).all()
    assert (t["predict_seconds"] > 0).all()
    assert t["peak_memory_mb"].notna().all()
    assert t["device"].notna().all()
    assert (t["n_series"] == 3).all()


def test_each_repeat_gets_its_own_seed(runner, panel):
    t = runner.run(panel, models=["constant"]).timings

    assert set(t["repeat"]) == {0, 1}
    assert t.groupby("repeat")["seed"].nunique().eq(1).all()
    assert t["seed"].nunique() == 2


def test_every_row_carries_its_provenance(runner, panel, cfg):
    result = runner.run(panel, models=["constant"],
                        tuning_budget={"constant": cfg.tuning.budget_trials})

    for df in (result.metrics, result.timings):
        assert (df["config_hash"] == cfg.hash).all()
        assert (df["tuning_trials"] == cfg.tuning.budget_trials).all()
        assert df["run_name"].notna().all()


def test_a_failing_model_records_a_failure_and_the_run_continues(runner, panel):
    result = runner.run(panel, models=["exploding", "constant"])

    failed = result.timings[result.timings["status"] == "failed"]
    assert len(failed) > 0
    assert "CUDA out of memory" in failed.iloc[0]["error"]
    assert "MemoryError" in failed.iloc[0]["traceback"]
    assert "constant" in set(result.timings[result.timings["status"] == "ok"]["model"])


def test_a_failing_model_contributes_no_metric_rows(runner, panel):
    result = runner.run(panel, models=["exploding", "constant"])

    assert "exploding" not in set(result.metrics["model"])


def test_a_malformed_prediction_is_caught_as_a_failure(runner, panel):
    result = runner.run(panel, models=["malformed"])

    failed = result.timings[result.timings["status"] == "failed"]
    assert len(failed) > 0
    assert "series" in failed.iloc[0]["error"]


def test_fit_is_reused_across_horizons_when_the_model_allows_it(runner, panel):
    t = runner.run(panel, models=["constant"]).timings
    fold0 = t[(t["fold"] == 0) & (t["repeat"] == 0)]

    assert fold0["fit_key"].nunique() == 1, "one fit serves both horizons"
    assert fold0["fit_reused"].sum() == 1


def test_models_that_bake_in_the_horizon_are_refitted(runner, panel):
    t = runner.run(panel, models=["refitting"]).timings
    fold0 = t[(t["fold"] == 0) & (t["repeat"] == 0)]

    assert fold0["fit_key"].nunique() == 2
    assert not fold0["fit_reused"].any()


def test_no_model_ever_sees_data_at_or_after_its_origin(cfg, panel, reg):
    """End-to-end guard: a spy records the frame it was actually handed."""
    seen = []

    class Spy(Constant):
        name = "spy"

        def fit(self, train_df):
            seen.append(train_df["ds"].max())
            super().fit(train_df)

    reg.register(Spy)
    result = BenchmarkRunner(cfg, registry=reg).run(panel, models=["spy"])
    origins = sorted(result.timings["origin"].unique())

    assert seen
    assert all(s <= max(origins) for s in seen)
    assert set(pd.to_datetime(seen)) <= set(pd.to_datetime(origins))


def test_checkpoint_lets_a_crashed_run_resume(cfg, panel, reg, tmp_path):
    first = BenchmarkRunner(cfg, registry=reg)
    first.run(panel, models=["constant"])

    calls = []

    class Counting(Constant):
        name = "counting"

        def fit(self, train_df):
            calls.append(1)
            super().fit(train_df)

    reg.register(Counting)
    resumed = BenchmarkRunner(cfg, registry=reg).run(panel, models=["constant", "counting"])

    assert len(calls) > 0, "the new model runs"
    assert set(resumed.metrics["model"]) == {"constant", "counting"}
    assert (resumed.timings[resumed.timings["model"] == "constant"]["resumed"]).all()


def test_a_changed_config_does_not_reuse_stale_checkpoints(cfg, panel, reg, config_dict, tmp_path):
    """Otherwise a protocol change silently inherits the old run's numbers."""
    BenchmarkRunner(cfg, registry=reg).run(panel, models=["constant"])

    config_dict["run"]["output_dir"] = str(tmp_path)
    config_dict["run"]["n_repeats"] = 2
    config_dict["run"]["execution"] = "inprocess"
    config_dict["protocol"]["horizons"] = [1, 2]
    changed = Config.from_dict(config_dict)
    result = BenchmarkRunner(changed, registry=reg).run(panel, models=["constant"])

    assert changed.hash != cfg.hash
    assert not result.timings["resumed"].any()
    assert set(result.timings["horizon"]) == {1, 2}


def test_a_training_frame_that_stops_short_of_the_origin_is_refused(cfg, reg):
    """A model anchored before the origin forecasts the wrong weeks. That was
    a real failure mode; it must be loud, not silent."""
    import numpy as np
    from tsbench.data.prepare import impute_gaps
    from tsbench.eval.splitter import ProtocolError, RollingOriginSplitter
    from tsbench.runner import assert_frame_reaches_origin

    ds = pd.date_range("2022-01-09", periods=30, freq="7D")
    frame = pd.DataFrame({"unique_id": "A", "ds": ds, "y": np.arange(30.0)})
    frame.loc[frame.index[-3:], "y"] = np.nan
    truncated = frame.iloc[:-3]

    assert_frame_reaches_origin(impute_gaps(frame)[0], ds[-1])

    with pytest.raises(ProtocolError, match="origin"):
        assert_frame_reaches_origin(truncated, ds[-1])


class WithCovariates(Constant):
    name = "with_covariates"
    family = "global"

    def supports_covariates(self):
        return True


def test_capable_models_run_with_and_without_covariates(config_dict, panel, reg, tmp_path):
    """The ablation is a core result: same model, covariates the only change."""
    config_dict["run"]["output_dir"] = str(tmp_path)
    config_dict["run"]["execution"] = "inprocess"
    config_dict["run"]["n_repeats"] = 1
    config_dict["models"]["covariate_ablation"] = True
    cfg = Config.from_dict(config_dict)
    reg.register(WithCovariates)

    result = BenchmarkRunner(cfg, registry=reg).run(panel, models=["with_covariates"])

    assert set(result.timings["covariates"]) == {False, True}
    assert set(result.metrics["covariates"]) == {False, True}


def test_models_without_covariate_support_run_once(config_dict, panel, reg, tmp_path):
    config_dict["run"]["output_dir"] = str(tmp_path)
    config_dict["run"]["execution"] = "inprocess"
    config_dict["run"]["n_repeats"] = 1
    config_dict["models"]["covariate_ablation"] = True
    cfg = Config.from_dict(config_dict)

    result = BenchmarkRunner(cfg, registry=reg).run(panel, models=["constant"])

    assert set(result.timings["covariates"]) == {False}


def test_the_ablation_is_off_by_default(runner, panel):
    result = runner.run(panel, models=["constant"])

    assert set(result.timings["covariates"]) == {False}


def test_the_two_ablation_arms_are_checkpointed_apart(config_dict, panel, reg, tmp_path):
    config_dict["run"]["output_dir"] = str(tmp_path)
    config_dict["run"]["execution"] = "inprocess"
    config_dict["run"]["n_repeats"] = 1
    config_dict["models"]["covariate_ablation"] = True
    cfg = Config.from_dict(config_dict)
    reg.register(WithCovariates)

    BenchmarkRunner(cfg, registry=reg).run(panel, models=["with_covariates"])
    resumed = BenchmarkRunner(cfg, registry=reg).run(panel, models=["with_covariates"])

    assert set(resumed.timings["covariates"]) == {False, True}
    assert resumed.timings["resumed"].all()


def test_results_are_written_as_long_parquet(runner, panel, cfg):
    result = runner.run(panel, models=["constant"])

    metrics = pd.read_parquet(result.paths["metrics"])
    timings = pd.read_parquet(result.paths["timings"])

    assert len(metrics) == len(result.metrics)
    assert len(timings) == len(result.timings)


def test_run_metadata_supports_the_reproducibility_statement(runner, panel):
    meta = runner.run(panel, models=["constant"]).metadata

    assert meta["config_hash"]
    assert meta["seed"] == 42
    assert meta["packages"]["pandas"]
    assert meta["hardware"]["cpu_count"] >= 1
    assert "timestamp" in meta
    assert "git_commit" in meta
    assert meta["imputation"]["method"] == "linear"


def test_a_groupby_produces_a_publication_table_without_reshaping(runner, panel):
    m = runner.run(panel, models=["constant"]).metrics

    table = (m[m["metric"] == "MASE"]
             .groupby(["model", "horizon"])["value"].median().unstack())

    assert list(table.columns) == [1, 3]
    assert table.index.tolist() == ["constant"]


def test_config_model_params_reach_the_adapter(config_dict, panel, reg, tmp_path):
    """models.params in the config must actually be applied, not decorative."""
    seen = {}

    class Recording(Constant):
        name = "recording"

        def fit(self, train_df):
            seen.update(self.params)
            super().fit(train_df)

    config_dict["run"]["output_dir"] = str(tmp_path)
    config_dict["run"]["execution"] = "inprocess"
    config_dict["run"]["n_repeats"] = 1
    cfg = Config.from_dict(config_dict)
    reg.register(Recording)

    BenchmarkRunner(cfg, registry=reg).run(
        panel, models=["recording"], tuned_params={"recording": {"alpha": 7}})

    assert seen["alpha"] == 7


def test_preload_runs_before_the_timed_fit(cfg, panel, reg):
    """Library imports and CUDA context creation belong to the process, not
    the model; preload() gives adapters a place to pay them untimed."""
    order = []

    class Preloading(Constant):
        name = "preloading"

        @classmethod
        def preload(cls):
            order.append("preload")

        def fit(self, train_df):
            order.append("fit")
            super().fit(train_df)

    reg.register(Preloading)
    BenchmarkRunner(cfg, registry=reg).run(panel, models=["preloading"])

    assert order[:2] == ["preload", "fit"]
    assert order.count("preload") == order.count("fit")


def test_package_versions_resolve_import_names_to_distributions(monkeypatch):
    """The adapters declare import names, but version() wants a distribution
    name - chronos ships as chronos-forecasting, toto as toto-ts. Looking up
    the import name directly records nothing for four of the five foundation
    backends.
    """
    import tsbench.runner as runner

    monkeypatch.setattr(runner, "TRACKED_PACKAGES", ["chronos", "toto", "absent"])
    monkeypatch.setattr(runner, "packages_distributions",
                        lambda: {"chronos": ["chronos-forecasting"],
                                 "toto": ["toto-ts"]})

    def fake_version(dist):
        if dist == "chronos-forecasting":
            return "1.5.2"
        if dist == "toto-ts":
            return "0.2.1"
        raise runner.PackageNotFoundError(dist)

    monkeypatch.setattr(runner, "version", fake_version)

    assert runner._package_versions() == {
        "chronos": "1.5.2", "toto": "0.2.1", "absent": None}


def test_a_horizon_at_fit_model_is_told_its_horizon_before_fit(runner, panel, reg):
    """These models cannot build the architecture until the horizon is known.
    If nobody supplies it, the work lands on the first predict and the timings
    report training as inference - which is how a global model comes to look
    almost free to train.
    """
    seen = []

    class Recording(ModelAdapter):
        name = "recording"
        family = "global"
        horizon_is_fit_time = True

        def fit(self, train_df):
            seen.append(("fit", self.fit_horizon))
            self._ids = sorted(train_df["unique_id"].unique())
            self._last = train_df["ds"].max()

        def predict(self, horizon):
            seen.append(("predict", horizon))
            ds = pd.date_range(self._last + pd.Timedelta(days=7), periods=horizon,
                               freq="7D")
            return pd.DataFrame([{"unique_id": u, "ds": d, "yhat": 1.0}
                                 for u in self._ids for d in ds])

    reg.register(Recording)
    runner.run(panel, models=["recording"])

    fits = [h for stage, h in seen if stage == "fit"]
    assert fits and all(h is not None for h in fits), \
        "fit ran without a horizon, so training could not happen under the fit timer"
    assert set(fits) == set(runner.cfg.protocol.horizons)
