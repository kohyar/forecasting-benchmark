"""The zero-shot path.

The model weights are not available here, so what is tested is everything the
harness owns: context construction, the no-op fit, quantile assembly, and the
fact that a zero fit time survives into the results instead of being folded
into predict.
"""
import numpy as np
import pandas as pd
import pytest

from tsbench.config import Config
from tsbench.data.loader import normalize_panel
from tsbench.models import registry as registry_module
from tsbench.models.base import AdapterError, validate_prediction
from tsbench.models.foundation_adapters import FoundationAdapter

WEEK = pd.Timedelta(days=7)


class StubFoundation(FoundationAdapter):
    """Stands in for a real checkpoint: records what it was shown."""

    name = "stub_foundation"
    package = "tsbench"

    def _forecast(self, contexts, horizon):
        self.seen = [np.array(c) for c in contexts]
        levels = len(self.quantile_levels)
        base = np.array([c[-1] for c in contexts])[:, None, None]
        offsets = np.linspace(-1, 1, levels)[None, None, :]
        return np.broadcast_to(base + offsets,
                               (len(contexts), horizon, levels)).copy()


@pytest.fixture
def cfg(config_dict):
    return Config.from_dict(config_dict)


@pytest.fixture
def panel(raw_frame, cfg):
    return normalize_panel(raw_frame, cfg)


@pytest.fixture
def model(cfg):
    return StubFoundation(cfg, params={"context_length": 20})


def test_fit_trains_nothing_and_is_fast(model, panel):
    import time

    start = time.perf_counter()
    model.fit(panel)

    assert time.perf_counter() - start < 1.0
    assert not hasattr(model, "_trained")


def test_context_is_the_tail_of_the_training_window(model, panel):
    model.fit(panel)
    uid = "SUB_A@@BOSTON, MA - MULO"
    expected = (panel[panel["unique_id"] == uid].sort_values("ds")["y"]
                .dropna().to_numpy()[-20:])

    np.testing.assert_allclose(model._context[uid], expected)


def test_context_never_reaches_past_the_training_frame(cfg, panel):
    from tsbench.eval.splitter import RollingOriginSplitter

    fold = next(iter(RollingOriginSplitter(cfg).split(panel)))
    model = StubFoundation(cfg, params={"context_length": 500})
    model.fit(fold.train)
    model.predict(3)

    for uid, context in model._context.items():
        history = fold.train[fold.train["unique_id"] == uid]["y"].dropna().to_numpy()
        assert len(context) <= len(history)
        np.testing.assert_allclose(context, history[-len(context):])


def test_prediction_satisfies_the_contract(model, panel):
    model.fit(panel)
    pred = model.predict(4)

    expected_ds = pd.date_range(panel["ds"].max() + WEEK, periods=4, freq="7D")
    validate_prediction(pred, sorted(panel["unique_id"].unique()), expected_ds)


def test_quantiles_are_emitted_and_ordered(model, panel):
    model.fit(panel)
    pred = model.predict(4)

    q_cols = [c for c in pred.columns if c.startswith("yhat_q")]
    assert len(q_cols) == len(model.quantile_levels)
    values = pred[sorted(q_cols, key=lambda c: int(c[6:]))].to_numpy()
    assert (np.diff(values, axis=1) >= -1e-9).all()


def test_point_forecast_is_the_median(model, panel):
    model.fit(panel)
    pred = model.predict(2)

    np.testing.assert_allclose(pred["yhat"], pred["yhat_q50"])


def test_a_wrong_shaped_forecast_is_rejected(cfg, panel):
    class Broken(StubFoundation):
        name = "broken"

        def _forecast(self, contexts, horizon):
            return np.zeros((len(contexts), horizon, 2))

    model = Broken(cfg)
    model.fit(panel)

    with pytest.raises(AdapterError, match="shape"):
        model.predict(3)


def test_zero_shot_models_are_not_tunable(model):
    assert model.tunable is False
    assert model.tuning_space(trial=None) == {}


def test_zero_fit_time_survives_into_the_results(config_dict, raw_frame, tmp_path):
    """The asymmetry between fit and predict is the finding, not an artefact."""
    from tsbench.models import registry
    from tsbench.runner import BenchmarkRunner

    config_dict["run"]["output_dir"] = str(tmp_path)
    config_dict["run"]["n_repeats"] = 1
    config_dict["run"]["execution"] = "inprocess"
    cfg = Config.from_dict(config_dict)
    panel = normalize_panel(raw_frame, cfg)

    reg = registry.Registry()
    reg.register(StubFoundation)
    result = BenchmarkRunner(cfg, registry=reg).run(panel, models=["stub_foundation"])

    ok = result.timings[result.timings["status"] == "ok"]
    assert (ok["fit_seconds"] < 0.5).all(), "no training happened"
    assert (ok["predict_seconds"] > 0).all()
    assert "fit_seconds" in ok and "predict_seconds" in ok
    assert (ok["tuning_trials"] == 0).all(), "zero-shot budget is reported, not blank"
    assert set(result.metrics["metric"]) >= {"MASE", "CRPS", "pinball"}


def test_every_foundation_adapter_is_registered_and_reports_its_package():
    reg = registry_module.default()
    foundation = [n for n in reg.names() if reg.get(n).family == "foundation"]

    assert {"chronos2", "timesfm", "toto", "tabpfn_ts", "ttm", "tirex"} <= set(foundation)
    for name in foundation:
        status = reg.status(name)
        assert status["available"] or status["disabled_reason"]


def test_bf16_is_used_only_where_the_gpu_supports_it(monkeypatch):
    """T4 (sm_75) has no bf16; Chronos-2 must fall back to float32 there
    rather than crash or silently upcast."""
    import torch

    from tsbench.models.foundation_adapters import preferred_dtype

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert preferred_dtype("cuda") == torch.float32

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert preferred_dtype("cuda") == torch.bfloat16

    assert preferred_dtype("cpu") == torch.float32
    assert preferred_dtype("mps") == torch.float32


def test_emulated_bf16_does_not_count(monkeypatch):
    """T4 reports bf16 support *with emulation*; that path is slow or breaks
    in some kernels, so only native support may select bf16."""
    import torch

    from tsbench.models.foundation_adapters import preferred_dtype

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def emulated_only(including_emulation=True):
        return including_emulation
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", emulated_only)
    assert preferred_dtype("cuda") == torch.float32

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: True)
    assert preferred_dtype("cuda") == torch.bfloat16


def test_bf16_probe_tolerates_an_older_torch_signature(monkeypatch):
    import torch

    from tsbench.models.foundation_adapters import preferred_dtype

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)  # no kwarg

    assert preferred_dtype("cuda") == torch.bfloat16
