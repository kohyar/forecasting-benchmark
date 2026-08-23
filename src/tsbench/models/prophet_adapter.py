"""Prophet, fitted per series.

weekly_seasonality is off and yearly_seasonality on: the data is weekly, so
there is no within-week pattern to fit, and the cycle that matters is annual.
"""
import numpy as np
import pandas as pd

from tsbench.models.base import AdapterError, ModelAdapter
from tsbench.models.registry import register

WEEK = pd.Timedelta(days=7)


@register
class ProphetAdapter(ModelAdapter):
    name = "prophet"
    package = "prophet"
    family = "local"

    @classmethod
    def preload(cls):
        import prophet  # noqa: F401

    @classmethod
    def is_available(cls):
        try:
            from prophet import Prophet  # noqa: F401
        except ImportError as exc:
            return False, f"prophet is not installed ({exc})"
        return True, ""

    def _make(self):
        from prophet import Prophet

        return Prophet(
            weekly_seasonality=False,   # weekly data has no within-week pattern
            yearly_seasonality=True,
            daily_seasonality=False,
            uncertainty_samples=self.params.get("uncertainty_samples", 200),
            seasonality_mode=self.params.get("seasonality_mode", "additive"),
            changepoint_prior_scale=self.params.get("changepoint_prior_scale", 0.05),
            seasonality_prior_scale=self.params.get("seasonality_prior_scale", 10.0),
        )

    def tuning_space(self, trial) -> dict:
        return {
            "changepoint_prior_scale": trial.suggest_float(
                "changepoint_prior_scale", 0.001, 0.5, log=True),
            "seasonality_prior_scale": trial.suggest_float(
                "seasonality_prior_scale", 0.01, 10.0, log=True),
            "seasonality_mode": trial.suggest_categorical(
                "seasonality_mode", ["additive", "multiplicative"]),
        }

    def fit(self, train_df: pd.DataFrame) -> None:
        import logging

        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        df = train_df[["unique_id", "ds", "y"]].dropna(subset=["y"])
        if df.empty:
            raise AdapterError(f"{self.name}: no observations to fit")

        self._models, self._failed = {}, []
        for uid, g in df.groupby("unique_id", sort=True):
            try:
                model = self._make()
                model.fit(g[["ds", "y"]].reset_index(drop=True))
                self._models[uid] = model
            except Exception as exc:
                self._failed.append((uid, str(exc)))

        if not self._models:
            raise AdapterError(f"{self.name}: every series failed to fit")
        self._ids = sorted(df["unique_id"].unique())
        self._last = df["ds"].max()

    def predict(self, horizon: int) -> pd.DataFrame:
        ds = pd.date_range(self._last + WEEK, periods=horizon, freq="7D")
        future = pd.DataFrame({"ds": ds})

        frames = []
        for uid in self._ids:
            model = self._models.get(uid)
            if model is None:
                continue
            samples = model.predictive_samples(future)["yhat"]
            out = pd.DataFrame({"unique_id": uid, "ds": ds,
                                "yhat": samples.mean(axis=1)})
            for q in self.quantile_levels:
                out[f"yhat_q{int(q * 100)}"] = np.quantile(samples, q, axis=1)
            frames.append(out)
        return pd.concat(frames, ignore_index=True)

    def supports_quantiles(self) -> bool:
        return True
