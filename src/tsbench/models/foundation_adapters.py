"""Zero-shot foundation models.

fit() is a no-op that captures the context window - these models are not
trained here, and their tuning budget is 0 by definition. The cost that
matters for them is inference, which is why fit and predict timings are never
summed.

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
        """No training. Capture the context each series ends on."""
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
    checkpoint = "amazon/chronos-2"

    def _forecast(self, contexts, horizon):
        from chronos import Chronos2Pipeline

        pipe = Chronos2Pipeline.from_pretrained(
            self.params.get("checkpoint", self.checkpoint),
            device_map=self._torch_device(),
            torch_dtype=preferred_dtype(self.device),
        )
        # inputs: one 1-d array per series -> one (n_variates=1, h, q) tensor each
        quantiles, _mean = pipe.predict_quantiles(
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
    checkpoint = "google/timesfm-2.5-200m-pytorch"

    def _forecast(self, contexts, horizon):
        import timesfm

        # torch.compile off: it adds minutes of compile latency to the first
        # call and would land inside the timed region.
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            self.params.get("checkpoint", self.checkpoint), torch_compile=False)
        model.compile(timesfm.ForecastConfig(
            max_context=self.context_length,          # rounded up to the patch size internally
            max_horizon=horizon,                      # rounded up to the output patch internally
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            fix_quantile_crossing=True,
            per_core_batch_size=int(self.params.get("batch_size", 32)),
        ))
        _point, quantile = model.forecast(horizon=horizon, inputs=list(contexts))
        # (n, horizon, 10): index 0 is the mean, 1..9 are the deciles q10..q90
        quantile = np.asarray(quantile)[:, :horizon, :]
        wanted = [int(round(q * 10)) for q in self.quantile_levels]
        return quantile[:, :, wanted]


@register
class TotoAdapter(FoundationAdapter):
    name = "toto"
    package = "toto"
    checkpoint = "Datadog/Toto-Open-Base-1.0"

    def _forecast(self, contexts, horizon):
        import torch
        from toto.data.util.dataset import MaskedTimeseries
        from toto.inference.forecaster import TotoForecaster
        from toto.model.toto import Toto

        device = self._torch_device()
        model = Toto.from_pretrained(self.params.get("checkpoint", self.checkpoint))
        model.to(device).eval()
        forecaster = TotoForecaster(model.model)

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
    checkpoint = "tabpfn-ts (local TabPFN checkpoint chosen by the package)"

    def _forecast(self, contexts, horizon):
        from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

        end = self._last
        step = pd.Timedelta(days=7)
        context_rows, future_rows = [], []
        for uid, context in zip(self._ids, contexts):
            hist = pd.date_range(end=end, periods=len(context), freq="7D")
            context_rows.append(pd.DataFrame(
                {"item_id": uid, "timestamp": hist, "target": np.asarray(context, dtype=float)}))
            future_rows.append(pd.DataFrame(
                {"item_id": uid, "timestamp": pd.date_range(end + step, periods=horizon, freq="7D")}))
        context_df = pd.concat(context_rows, ignore_index=True)
        future_df = pd.concat(future_rows, ignore_index=True)

        pipeline = TabPFNTSPipeline(
            tabpfn_mode=TabPFNMode.LOCAL,
            max_context_length=int(self.params.get("max_context_length", 4096)),
        )
        try:
            from tabpfn_time_series.defaults import TABPFN_V3_TS_CHECKPOINT
            self.resolved_checkpoint = TABPFN_V3_TS_CHECKPOINT
        except ImportError:
            pass
        pred = pipeline.predict_df(context_df, future_df=future_df,
                                   quantiles=[float(q) for q in self.quantile_levels])

        # Rows come back indexed by (item_id, timestamp); quantile columns are
        # named by level, as strings in current versions.
        pred = pred.reset_index()
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        pred = pred.set_index(["item_id", "timestamp"]).sort_index()
        wanted = pd.MultiIndex.from_product(
            [self._ids, pd.date_range(end + step, periods=horizon, freq="7D")],
            names=["item_id", "timestamp"])
        pred = pred.reindex(wanted)

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


@register
class TTMAdapter(FoundationAdapter):
    name = "ttm"
    package = "tsfm_public"
    checkpoint = "ibm-granite/granite-timeseries-ttm-r2"
    # TTM ships fixed context lengths (52/90/180/360/512 in r2.1); 90 is the
    # nearest to the ~104-week context the other zero-shot models see, and
    # every eligible series has at least 104 weeks of history.
    default_context = 90

    def _forecast(self, contexts, horizon):
        import torch
        from tsfm_public.toolkit.get_model import get_model

        repo = self.params.get("checkpoint", self.checkpoint)
        # get_model picks the pretrained variant whose context/forecast lengths
        # cover the request and trims the forecast to `horizon`.
        key = get_model(repo, context_length=self.context_length,
                        prediction_length=horizon, return_model_key=True)
        self.resolved_checkpoint = f"{repo}@{key}"
        model = get_model(repo, context_length=self.context_length, prediction_length=horizon)
        device = self._torch_device()
        model = model.to(device).eval()

        width = int(model.config.context_length)
        padded = np.stack([np.pad(c, (max(width - len(c), 0), 0), mode="edge")[-width:]
                           for c in contexts])
        batch = torch.tensor(padded, dtype=torch.float32, device=device).unsqueeze(-1)  # (n, T, 1)

        with torch.no_grad():
            out = model(past_values=batch).prediction_outputs      # (n, fl, 1)
        point = out[:, :horizon, 0].cpu().numpy()

        # TTM is a point forecaster; widen it into quantiles with the residual
        # spread of the context so CRPS and coverage are defined. Reported as a
        # limitation rather than passed off as a native predictive distribution.
        spread = np.array([np.std(np.diff(c)) if len(c) > 1 else 0.0 for c in contexts])
        z = np.array([_normal_quantile(q) for q in self.quantile_levels])
        return point[:, :, None] + spread[:, None, None] * z[None, None, :]


def _normal_quantile(q: float) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(q)
