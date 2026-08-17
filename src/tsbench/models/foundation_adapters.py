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


class FoundationAdapter(ModelAdapter):
    family = "foundation"
    tunable = False           # zero-shot: the results report a budget of 0
    package = ""
    checkpoint = ""
    #: how much history the model is shown, in weeks
    default_context = 104

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
        import torch
        from chronos import BaseChronosPipeline

        pipe = BaseChronosPipeline.from_pretrained(
            self.params.get("checkpoint", self.checkpoint),
            device_map=self._torch_device(),
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )
        quantiles, _mean = pipe.predict_quantiles(
            context=[torch.tensor(c, dtype=torch.float32) for c in contexts],
            prediction_length=horizon,
            quantile_levels=list(self.quantile_levels),
        )
        return quantiles.float().cpu().numpy()


@register
class TimesFMAdapter(FoundationAdapter):
    name = "timesfm"
    package = "timesfm"
    checkpoint = "google/timesfm-2.5-200m-pytorch"

    def _forecast(self, contexts, horizon):
        import timesfm

        model = timesfm.TimesFM_2p5_200M_torch()
        model.load_checkpoint()
        model.compile(timesfm.ForecastConfig(
            max_context=self.context_length,
            max_horizon=horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
        ))
        _point, quantile = model.forecast(horizon=horizon, inputs=contexts)
        # TimesFM returns 10 heads: mean followed by deciles q10..q90
        quantile = np.asarray(quantile)
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
        model.to(device)
        forecaster = TotoForecaster(model.model)

        width = max(len(c) for c in contexts)
        padded = np.stack([np.pad(c, (width - len(c), 0), mode="edge") for c in contexts])
        series = torch.tensor(padded, dtype=torch.float32, device=device).unsqueeze(0)

        inputs = MaskedTimeseries(
            series=series,
            padding_mask=torch.full_like(series, True, dtype=torch.bool),
            id_mask=torch.zeros_like(series),
            timestamp_seconds=torch.zeros_like(series),
            time_interval_seconds=torch.full((1, series.shape[1]), 7 * 86400,
                                             device=device),
        )
        forecast = forecaster.forecast(
            inputs, prediction_length=horizon,
            num_samples=self.params.get("num_samples", 256))
        samples = forecast.samples.squeeze(0).float().cpu().numpy()   # (series, h, samples)
        return np.quantile(samples, list(self.quantile_levels), axis=-1).transpose(1, 2, 0)


@register
class TabPFNTSAdapter(FoundationAdapter):
    name = "tabpfn_ts"
    package = "tabpfn_time_series"
    checkpoint = "tabpfn-v2"

    def _forecast(self, contexts, horizon):
        from tabpfn_time_series import TabPFNMode, TabPFNTimeSeriesPredictor

        end = self._last
        train_rows, test_rows = [], []
        for uid, context in zip(self._ids, contexts):
            hist = pd.date_range(end=end, periods=len(context), freq="7D")
            train_rows.append(pd.DataFrame({"item_id": uid, "timestamp": hist,
                                            "target": context}))
            future = pd.date_range(end + pd.Timedelta(days=7), periods=horizon, freq="7D")
            test_rows.append(pd.DataFrame({"item_id": uid, "timestamp": future,
                                           "target": np.nan}))

        predictor = TabPFNTimeSeriesPredictor(
            tabpfn_mode=TabPFNMode.LOCAL,
            config={"tabpfn_output_selection": [f"quantile_{q:.1f}"
                                                for q in self.quantile_levels]},
        )
        pred = predictor.predict(pd.concat(train_rows, ignore_index=True),
                                 pd.concat(test_rows, ignore_index=True))
        cols = [str(q) for q in self.quantile_levels]
        return (pred[cols].to_numpy()
                .reshape(len(self._ids), horizon, len(self.quantile_levels)))


@register
class TTMAdapter(FoundationAdapter):
    name = "ttm"
    package = "tsfm_public"
    checkpoint = "ibm-granite/granite-timeseries-ttm-r2"
    # TTM is trained at fixed context lengths; 52 weeks is the closest fit
    default_context = 52

    def _forecast(self, contexts, horizon):
        import torch
        from tsfm_public import TinyTimeMixerForPrediction

        device = self._torch_device()
        model = TinyTimeMixerForPrediction.from_pretrained(
            self.params.get("checkpoint", self.checkpoint),
            context_length=self.context_length,
            prediction_length=horizon,
            head_dropout=0.0,
        ).to(device).eval()

        width = self.context_length
        padded = np.stack([np.pad(c, (max(width - len(c), 0), 0), mode="edge")[-width:]
                           for c in contexts])
        batch = torch.tensor(padded, dtype=torch.float32, device=device).unsqueeze(-1)

        with torch.no_grad():
            point = model(past_values=batch).prediction_outputs.squeeze(-1).cpu().numpy()

        # TTM is a point forecaster; widen it into quantiles with the residual
        # spread of the context so CRPS and coverage are defined. Reported as a
        # limitation rather than passed off as a native predictive distribution.
        spread = np.array([np.std(np.diff(c)) if len(c) > 1 else 0.0 for c in contexts])
        z = np.array([_normal_quantile(q) for q in self.quantile_levels])
        return point[:, :, None] + spread[:, None, None] * z[None, None, :]


def _normal_quantile(q: float) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(q)
