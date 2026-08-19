"""Tensor plumbing of the five foundation adapters, against fakes that return
exactly the shapes the installed libraries return (read from their sources:
chronos-forecasting 2.3.1, timesfm 2.0.2, toto-ts 0.2.0, tabpfn-time-series
1.2.0, granite-tsfm 0.3.8).

This cannot prove the real models run - only that, given their documented
outputs, the adapters slice, order and reshape them correctly. Each fake also
records the arguments it received, so the call signatures are pinned too.
"""
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch

from tsbench.config import Config
from tsbench.models import foundation_adapters as fa

N, H, Q = 3, 4, 9
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@pytest.fixture
def cfg(config_dict):
    config_dict["metrics"]["quantile_levels"] = LEVELS
    return Config.from_dict(config_dict)


@pytest.fixture
def contexts():
    rng = np.random.default_rng(0)
    return [rng.normal(100, 5, 60 + i * 10) for i in range(N)]


def _fake_module(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _install(monkeypatch, name, module):
    monkeypatch.setitem(sys.modules, name, module)


def _expect_shape(out):
    assert out.shape == (N, H, Q), out.shape
    assert np.isfinite(out).all()


# --- Chronos-2 -------------------------------------------------------------

def test_chronos2_unpacks_a_list_of_per_series_tensors(monkeypatch, cfg, contexts):
    calls = {}

    class Pipe:
        @classmethod
        def from_pretrained(cls, name, **kw):
            calls["from_pretrained"] = (name, kw)
            return cls()

        def predict_quantiles(self, inputs, prediction_length, quantile_levels, **kw):
            calls["inputs"] = inputs
            calls["kw"] = dict(prediction_length=prediction_length, quantile_levels=quantile_levels, **kw)
            # one (n_variates=1, h, q) tensor per input, values encode series index
            return ([torch.full((1, prediction_length, len(quantile_levels)), float(i))
                     for i in range(len(inputs))],
                    [torch.zeros(1, prediction_length) for _ in inputs])

    _install(monkeypatch, "chronos", _fake_module("chronos", Chronos2Pipeline=Pipe))
    model = fa.Chronos2Adapter(cfg, device="cpu")
    out = model._forecast(contexts, H)

    _expect_shape(out)
    assert len(calls["inputs"]) == N and calls["inputs"][0].ndim == 1
    assert calls["kw"]["prediction_length"] == H
    assert calls["kw"]["quantile_levels"] == LEVELS
    assert (out[2] == 2.0).all(), "series order preserved"


# --- TimesFM ---------------------------------------------------------------

def test_timesfm_selects_the_deciles_and_drops_the_mean_head(monkeypatch, cfg, contexts):
    calls = {}

    class ForecastConfig:
        def __init__(self, **kw):
            calls["config"] = kw

    class Model:
        @classmethod
        def from_pretrained(cls, name, **kw):
            calls["from_pretrained"] = (name, kw)
            return cls()

        def compile(self, fc):
            calls["compiled"] = True

        def forecast(self, horizon, inputs):
            n = len(inputs)
            # (n, horizon, 10): head 0 = mean (marked 99), heads 1..9 = deciles = their index
            q = np.tile(np.arange(10, dtype=float), (n, horizon, 1))
            q[:, :, 0] = 99.0
            return np.zeros((n, horizon)), q

    _install(monkeypatch, "timesfm",
             _fake_module("timesfm", TimesFM_2p5_200M_torch=Model, ForecastConfig=ForecastConfig))
    out = fa.TimesFMAdapter(cfg, device="cpu")._forecast(contexts, H)

    _expect_shape(out)
    assert calls["from_pretrained"][1]["torch_compile"] is False
    assert calls["config"]["fix_quantile_crossing"] is True
    assert (out[:, :, 0] == 1.0).all() and (out[:, :, 8] == 9.0).all(), "q10..q90 = heads 1..9"
    assert not (out == 99.0).any(), "the mean head is never used as a quantile"


# --- Toto ------------------------------------------------------------------

def test_toto_puts_each_series_in_the_batch_not_the_variate_axis(monkeypatch, cfg, contexts):
    calls = {}

    class Backbone:
        pass

    class Toto:
        model = Backbone()

        @classmethod
        def load_from_checkpoint(cls, path, map_location="cpu", strict=True, **kw):
            calls["checkpoint_dir"] = path
            return cls()

        @classmethod
        def from_pretrained(cls, *a, **kw):
            raise TypeError("hub mixin path must not be used")

        def to(self, device):
            return self

        def eval(self):
            return self

    class Forecast:
        def __init__(self, samples):
            self.samples = samples

    class Forecaster:
        def __init__(self, backbone):
            calls["backbone"] = backbone

        def forecast(self, inputs, prediction_length, num_samples, samples_per_batch, **kw):
            calls["inputs"] = inputs
            calls["num_samples"] = num_samples
            calls["samples_per_batch"] = samples_per_batch
            b, v, _ = inputs.series.shape
            # samples encode the batch index so ordering can be checked
            base = torch.arange(b, dtype=torch.float32).view(b, 1, 1, 1)
            return Forecast(base + torch.zeros(b, v, prediction_length, num_samples))

    class MaskedTimeseries:
        def __init__(self, **kw):
            self.__dict__.update(kw)
            self.series = kw["series"]

    _install(monkeypatch, "huggingface_hub",
             _fake_module("huggingface_hub",
                          snapshot_download=lambda repo, **kw: f"/fake/snapshots/{repo}"))
    _install(monkeypatch, "toto", _fake_module("toto"))
    _install(monkeypatch, "toto.model", _fake_module("toto.model"))
    _install(monkeypatch, "toto.model.toto", _fake_module("toto.model.toto", Toto=Toto))
    _install(monkeypatch, "toto.inference", _fake_module("toto.inference"))
    _install(monkeypatch, "toto.inference.forecaster",
             _fake_module("toto.inference.forecaster", TotoForecaster=Forecaster))
    _install(monkeypatch, "toto.data", _fake_module("toto.data"))
    _install(monkeypatch, "toto.data.util", _fake_module("toto.data.util"))
    _install(monkeypatch, "toto.data.util.dataset",
             _fake_module("toto.data.util.dataset", MaskedTimeseries=MaskedTimeseries))

    out = fa.TotoAdapter(cfg, device="cpu")._forecast(contexts, H)

    _expect_shape(out)
    inp = calls["inputs"]
    assert inp.series.shape == (N, 1, max(len(c) for c in contexts)), "batch=series, variates=1"
    assert inp.padding_mask.shape == inp.series.shape
    assert inp.time_interval_seconds.shape == (N, 1)
    assert calls["num_samples"] % calls["samples_per_batch"] == 0
    assert (out[1] == 1.0).all(), "series order preserved"
    assert isinstance(calls["backbone"], Backbone), "forecaster wraps model.model"
    assert calls["checkpoint_dir"] == "/fake/snapshots/Datadog/Toto-Open-Base-1.0"


# --- TabPFN-TS ------------------------------------------------------------

def test_tabpfn_reindexes_the_item_timestamp_frame_and_reads_string_quantile_columns(
        monkeypatch, cfg, contexts):
    calls = {}

    class Mode:
        LOCAL = "local"

    class Pipeline:
        def __init__(self, **kw):
            calls["init"] = kw

        def predict_df(self, context_df, future_df=None, prediction_length=None, quantiles=None):
            calls["context_cols"] = list(context_df.columns)
            calls["future"] = future_df
            # returned scrambled, indexed by (item_id, timestamp), string quantile columns
            rows = future_df.sample(frac=1.0, random_state=1).copy()
            for q in quantiles:
                rows[str(q)] = rows["item_id"].map({uid: i for i, uid in enumerate(sorted(rows["item_id"].unique()))}).astype(float) + q
            rows["target"] = rows["0.5"]
            return rows.set_index(["item_id", "timestamp"])

    _install(monkeypatch, "tabpfn_time_series",
             _fake_module("tabpfn_time_series", TabPFNMode=Mode, TabPFNTSPipeline=Pipeline))
    _install(monkeypatch, "tabpfn_time_series.defaults",
             _fake_module("tabpfn_time_series.defaults", TABPFN_V3_TS_CHECKPOINT="ckpt-x"))

    model = fa.TabPFNTSAdapter(cfg, device="cpu")
    model.fit(pd.DataFrame({
        "unique_id": np.repeat([f"S{i}" for i in range(N)], 60),
        "ds": np.tile(pd.date_range("2024-01-07", periods=60, freq="7D"), N),
        "y": np.concatenate([c[:60] for c in contexts]),
    }))
    out = model._forecast([c[:60] for c in contexts], H)

    _expect_shape(out)
    assert set(calls["context_cols"]) == {"item_id", "timestamp", "target"}
    assert len(calls["future"]) == N * H
    assert model.resolved_checkpoint == "ckpt-x"
    # series i, quantile q -> i + q, in the adapter's own series order
    assert out[1, 0, 0] == pytest.approx(1 + 0.1)
    assert out[2, 3, 8] == pytest.approx(2 + 0.9)


# --- TTM -------------------------------------------------------------------

def test_ttm_uses_get_model_and_widens_the_point_forecast(monkeypatch, cfg, contexts):
    calls = {}

    class Out:
        def __init__(self, n, fl):
            self.prediction_outputs = torch.arange(n, dtype=torch.float32).view(n, 1, 1).expand(n, fl, 1)

    class Cfg:
        context_length = 90

    class Model:
        config = Cfg()

        def to(self, d):
            return self

        def eval(self):
            return self

        def __call__(self, past_values):
            calls["past_values"] = past_values
            return Out(past_values.shape[0], 16)

    def get_model(repo, context_length=None, prediction_length=None, return_model_key=False, **kw):
        calls.setdefault("get_model", []).append((repo, context_length, prediction_length, return_model_key))
        return "90-30-ft-r2.1" if return_model_key else Model()

    _install(monkeypatch, "tsfm_public", _fake_module("tsfm_public"))
    _install(monkeypatch, "tsfm_public.toolkit", _fake_module("tsfm_public.toolkit"))
    _install(monkeypatch, "tsfm_public.toolkit.get_model",
             _fake_module("tsfm_public.toolkit.get_model", get_model=get_model))

    model = fa.TTMAdapter(cfg, device="cpu")
    out = model._forecast(contexts, H)

    _expect_shape(out)
    assert calls["past_values"].shape == (N, 90, 1), "padded/trimmed to the model's context"
    assert model.resolved_checkpoint.endswith("@90-30-ft-r2.1")
    assert (np.diff(out, axis=2) >= 0).all(), "widened quantiles are ordered"
    np.testing.assert_allclose(out[:, :, 4], np.tile(np.arange(N)[:, None], (1, H)),
                               atol=1e-6, err_msg="median = point")
