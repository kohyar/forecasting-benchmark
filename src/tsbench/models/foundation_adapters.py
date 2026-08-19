"""Zero-shot foundation models.

fit() trains nothing: it captures the context window and loads the checkpoint.
So for these models `fit_seconds` is setup (weight loading, seconds) rather
than training (minutes to hours), and `predict_seconds` is pure inference -
the number practitioners care about most, kept free of disk I/O. The tuning
budget is 0 by definition.

Everything shared - context construction, horizon handling, quantile assembly,
validation - lives in FoundationAdapter and is covered by tests. Each model
contributes only `_forecast()`, which is the part that depends on a specific
library API and can only be verified where the weights are available.
"""
import numpy as np
import pandas as pd

from tsbench.models.base import AdapterError, ModelAdapter
from tsbench.models.registry import register


def preferred_dtype(device: str):
    """bf16 only where the GPU supports it natively, float32 everywhere else.

    A T4 (sm_75) reports bf16 support *with emulation*, which is slow and
    breaks in some kernels, so emulated support does not count.
    """
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        return torch.float32
    try:
        native = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # older torch: no emulation flag, answer is native-only
        native = torch.cuda.is_bf16_supported()
    return torch.bfloat16 if native else torch.float32


class FoundationAdapter(ModelAdapter):
    family = "foundation"
    tunable = False           # zero-shot: the results report a budget of 0
    package = ""
    #: the modules _load() imports - imported untimed by preload()
    preload_modules = ()

    @classmethod
    def preload(cls):
        """Import the library and create the device context untimed, so
        fit_seconds is checkpoint loading and nothing else."""
        import importlib

        import torch

        for module in cls.preload_modules or (cls.package,):
            importlib.import_module(module)
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
    checkpoint = ""
    #: how much history the model is shown, in weeks
    default_context = 104
    #: set by adapters that resolve a concrete checkpoint at run time
    resolved_checkpoint = None

    @classmethod
    def is_available(cls):
        import importlib.util

        if importlib.util.find_spec(cls.package.split(".")[0]) is None:
            return False, f"{cls.package} is not installed"
        return True, ""

    def supports_quantiles(self) -> bool:
        return True

    @property
    def context_length(self) -> int:
        return int(self.params.get("context_length", self.default_context))

    def fit(self, train_df: pd.DataFrame) -> None:
        """No training. Capture the context each series ends on, then load
        the checkpoint so predict() measures inference alone."""
        df = train_df[["unique_id", "ds", "y"]].dropna(subset=["y"])
        if df.empty:
            raise AdapterError(f"{self.name}: no observations to build a context from")

        df = df.sort_values(["unique_id", "ds"])
        self._ids = sorted(df["unique_id"].unique())
        self._last = df["ds"].max()
        self._context = {
            uid: g["y"].to_numpy(dtype=float)[-self.context_length:]
            for uid, g in df.groupby("unique_id", sort=True)
        }
        self._max_horizon = max(self.cfg.protocol.horizons)
        self._load()
        self._warm_up()

    def _load(self) -> None:
        """Load weights / build the pipeline. Called once, from fit()."""

    def _warm_up(self) -> None:
        """One series, one step: pays first-call costs (lazy weight loading,
        kernel selection, graph capture) inside fit() so predict() measures
        steady-state inference. Milliseconds of work, not a forecast."""
        first_id = self._ids[0]
        ids, self._ids = self._ids, [first_id]
        try:
            self._forecast([self._context[first_id]], 1)
        finally:
            self._ids = ids

    def predict(self, horizon: int) -> pd.DataFrame:
        contexts = [self._context[uid] for uid in self._ids]
        quantiles = self._forecast(contexts, horizon)

        levels = list(self.quantile_levels)
        expected = (len(self._ids), horizon, len(levels))
        if quantiles.shape != expected:
            raise AdapterError(
                f"{self.name}: expected quantile array of shape {expected}, "
                f"got {quantiles.shape}")

        ds = pd.date_range(self._last + pd.Timedelta(days=7), periods=horizon, freq="7D")
        out = pd.DataFrame({
            "unique_id": np.repeat(self._ids, horizon),
            "ds": np.tile(ds.to_numpy(), len(self._ids)),
        })

        flat = quantiles.reshape(-1, len(levels))
        flat = np.sort(flat, axis=1)          # guard against crossed quantiles
        for i, q in enumerate(levels):
            out[f"yhat_q{int(q * 100)}"] = flat[:, i]

        median = levels.index(0.5) if 0.5 in levels else len(levels) // 2
        out["yhat"] = flat[:, median]
        return out

    def _forecast(self, contexts: list, horizon: int) -> np.ndarray:
        """Return an array shaped (n_series, horizon, n_quantile_levels).

        The only model-specific code. Unverified until run where the weights
        are available.
        """
        raise NotImplementedError

    def _torch_device(self):
        return {"cuda": "cuda", "mps": "mps"}.get(self.device, "cpu")


@register
class Chronos2Adapter(FoundationAdapter):
    name = "chronos2"
    package = "chronos"
    preload_modules = ("chronos",)
    checkpoint = "amazon/chronos-2"

    def _load(self):
        from chronos import Chronos2Pipeline

        self._pipe = Chronos2Pipeline.from_pretrained(
            self.params.get("checkpoint", self.checkpoint),
            device_map=self._torch_device(),
            torch_dtype=preferred_dtype(self.device),
        )

    def _forecast(self, contexts, horizon):
        # inputs: one 1-d array per series -> one (n_variates=1, h, q) tensor each
        quantiles, _mean = self._pipe.predict_quantiles(
            [np.asarray(c, dtype=np.float32) for c in contexts],
            prediction_length=horizon,
            quantile_levels=list(self.quantile_levels),
            batch_size=int(self.params.get("batch_size", 256)),
        )
        stacked = np.stack([q.float().cpu().numpy() for q in quantiles])   # (n, 1, h, q)
        return stacked[:, 0, :, :]


@register
class TimesFMAdapter(FoundationAdapter):
    name = "timesfm"
    package = "timesfm"
    preload_modules = ("timesfm",)
    checkpoint = "google/timesfm-2.5-200m-pytorch"

    def _load(self):
        import timesfm

        # torch.compile off: minutes of compile latency that would otherwise
        # land in the first timed call.
        self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            self.params.get("checkpoint", self.checkpoint), torch_compile=False)
        # Compiled once for the longest horizon; shorter horizons decode less.
        self._model.compile(timesfm.ForecastConfig(
            max_context=self.context_length,          # rounded up to the patch size internally
            max_horizon=self._max_horizon,            # rounded up to the output patch internally
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            fix_quantile_crossing=True,
            per_core_batch_size=int(self.params.get("batch_size", 32)),
        ))

    def _forecast(self, contexts, horizon):
        _point, quantile = self._model.forecast(horizon=horizon, inputs=list(contexts))
        # (n, horizon, 10): index 0 is the mean, 1..9 are the deciles q10..q90
        quantile = np.asarray(quantile)[:, :horizon, :]
        wanted = [int(round(q * 10)) for q in self.quantile_levels]
        return quantile[:, :, wanted]


@register
class TotoAdapter(FoundationAdapter):
    name = "toto"
    package = "toto"
    # toto/__init__ is nearly empty; the heavy imports (lightning, gluonts,
    # rotary embeddings) live under these
    preload_modules = ("toto.model.toto", "toto.inference.forecaster",
                       "toto.data.util.dataset", "huggingface_hub")
    checkpoint = "Datadog/Toto-Open-Base-1.0"

    def _load(self):
        from toto.inference.forecaster import TotoForecaster
        from toto.model.toto import Toto

        model = _load_toto(Toto, self.params.get("checkpoint", self.checkpoint),
                           self._torch_device())
        self._forecaster = TotoForecaster(model.model)

    def _forecast(self, contexts, horizon):
        import torch
        from toto.data.util.dataset import MaskedTimeseries

        device = self._torch_device()
        forecaster = self._forecaster

        # Each series is its own batch element with a single variate. Stacking
        # them as variates of one input would let unrelated series attend to
        # each other - Toto is multivariate by design.
        width = max(len(c) for c in contexts)
        padded = np.stack([np.pad(c, (width - len(c), 0), mode="edge") for c in contexts])
        valid = np.stack([np.r_[np.zeros(width - len(c), bool), np.ones(len(c), bool)]
                          for c in contexts])
        series = torch.tensor(padded, dtype=torch.float32, device=device).unsqueeze(1)  # (n, 1, T)
        mask = torch.tensor(valid, device=device).unsqueeze(1)

        inputs = MaskedTimeseries(
            series=series,
            padding_mask=mask,
            id_mask=torch.zeros_like(series, dtype=torch.int64),
            timestamp_seconds=torch.zeros_like(series, dtype=torch.int64),
            time_interval_seconds=torch.full((series.shape[0], 1), 7 * 86400,
                                             dtype=torch.int64, device=device),
        )
        num_samples = int(self.params.get("num_samples", 256))
        per_batch = int(self.params.get("samples_per_batch", 32))
        num_samples -= num_samples % per_batch     # must be divisible
        with torch.no_grad():
            forecast = forecaster.forecast(
                inputs, prediction_length=horizon,
                num_samples=num_samples, samples_per_batch=per_batch)

        samples = forecast.samples[:, 0].float().cpu().numpy()   # (n, h, samples)
        return np.quantile(samples, list(self.quantile_levels), axis=-1).transpose(1, 2, 0)


@register
class TabPFNTSAdapter(FoundationAdapter):
    name = "tabpfn_ts"
    package = "tabpfn_time_series"
    preload_modules = ("tabpfn_time_series", "tabpfn")
    checkpoint = "tabpfn-ts (local TabPFN checkpoint chosen by the package)"

    def _load(self):
        from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

        self._pipeline = TabPFNTSPipeline(
            tabpfn_mode=TabPFNMode.LOCAL,
            max_context_length=int(self.params.get("max_context_length", 4096)),
        )
        try:
            from tabpfn_time_series.defaults import TABPFN_V3_TS_CHECKPOINT
            self.resolved_checkpoint = TABPFN_V3_TS_CHECKPOINT
        except ImportError:
            pass
        # TabPFN loads its weights lazily on the first prediction; the base
        # class's one-series warm-up in fit() pulls that out of predict().

    def _forecast(self, contexts, horizon):
        pred = self._predict_frame(self._ids, contexts, horizon)
        columns = []
        for q in self.quantile_levels:
            for candidate in (str(q), q, f"{q:.1f}", str(float(q))):
                if candidate in pred.columns:
                    columns.append(candidate)
                    break
            else:
                raise AdapterError(f"tabpfn_ts: no column for quantile {q}; have {list(pred.columns)}")
        return pred[columns].to_numpy(dtype=float).reshape(
            len(self._ids), horizon, len(self.quantile_levels))

    def _predict_frame(self, ids, contexts, horizon):
        end = self._last
        step = pd.Timedelta(days=7)
        context_rows, future_rows = [], []
        for uid, context in zip(ids, contexts):
            hist = pd.date_range(end=end, periods=len(context), freq="7D")
            context_rows.append(pd.DataFrame(
                {"item_id": uid, "timestamp": hist, "target": np.asarray(context, dtype=float)}))
            future_rows.append(pd.DataFrame(
                {"item_id": uid, "timestamp": pd.date_range(end + step, periods=horizon, freq="7D")}))
        context_df = pd.concat(context_rows, ignore_index=True)
        future_df = pd.concat(future_rows, ignore_index=True)

        pred = self._pipeline.predict_df(context_df, future_df=future_df,
                                         quantiles=[float(q) for q in self.quantile_levels])

        # Rows come back indexed by (item_id, timestamp); quantile columns are
        # named by level, as strings in current versions.
        pred = pred.reset_index()
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        pred = pred.set_index(["item_id", "timestamp"]).sort_index()
        wanted = pd.MultiIndex.from_product(
            [list(ids), pd.date_range(end + step, periods=horizon, freq="7D")],
            names=["item_id", "timestamp"])
        return pred.reindex(wanted)


@register
class TTMAdapter(FoundationAdapter):
    name = "ttm"
    package = "tsfm_public"
    preload_modules = ("tsfm_public.toolkit.get_model",)
    checkpoint = "ibm-granite/granite-timeseries-ttm-r2"
    # TTM ships fixed context lengths (52/90/180/360/512 in r2.1); 90 is the
    # nearest to the ~104-week context the other zero-shot models see, and
    # every eligible series has at least 104 weeks of history.
    default_context = 90

    def _load(self):
        from tsfm_public.toolkit.get_model import get_model

        repo = self.params.get("checkpoint", self.checkpoint)
        # get_model picks the pretrained variant whose context/forecast lengths
        # cover the longest horizon; shorter horizons are sliced from it.
        select = dict(context_length=self.context_length,
                      prediction_length=self._max_horizon, freq="W")
        key = get_model(repo, return_model_key=True, **select)
        self.resolved_checkpoint = f"{repo}@{key}"
        model = get_model(repo, **select)
        self._model = model.to(self._torch_device()).eval()

    def _forecast(self, contexts, horizon):
        import torch

        model = self._model
        device = self._torch_device()
        width = int(model.config.context_length)
        padded = np.stack([np.pad(c, (max(width - len(c), 0), 0), mode="edge")[-width:]
                           for c in contexts])
        batch = torch.tensor(padded, dtype=torch.float32, device=device).unsqueeze(-1)  # (n, T, 1)

        # Frequency-prefix-tuned variants (the "-ft-" checkpoints) require a
        # frequency token; others ignore it. Weekly is 9 in tsfm's mapping.
        freq_token = torch.full((batch.shape[0],), _TTM_WEEKLY_TOKEN,
                                dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(past_values=batch, freq_token=freq_token).prediction_outputs  # (n, fl, 1)
        point = out[:, :horizon, 0].cpu().numpy()

        # TTM is a point forecaster; widen it into quantiles with the residual
        # spread of the context so CRPS and coverage are defined. Reported as a
        # limitation rather than passed off as a native predictive distribution.
        spread = np.array([np.std(np.diff(c)) if len(c) > 1 else 0.0 for c in contexts])
        z = np.array([_normal_quantile(q) for q in self.quantile_levels])
        return point[:, :, None] + spread[:, None, None] * z[None, None, :]


_TTM_WEEKLY_TOKEN = 9   # tsfm_public DEFAULT_FREQUENCY_MAPPING["W"]


def _load_toto(toto_cls, repo_or_dir: str, device: str):
    """Load Toto without its hub mixin.

    Toto's `_from_pretrained` was written for huggingface_hub < 1.0 and
    requires keyword arguments hub 1.x no longer passes. Its own
    `load_from_checkpoint(directory)` reads config.json and the safetensors
    file itself, so fetch the snapshot and hand it the directory.
    """
    import os

    if os.path.isdir(repo_or_dir):
        local_dir = repo_or_dir
    else:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(repo_or_dir, allow_patterns=["*.json", "*.safetensors"])
    model = toto_cls.load_from_checkpoint(local_dir, map_location=device, strict=False)
    return model.to(device).eval()


def _normal_quantile(q: float) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(q)
