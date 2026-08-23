"""Adapters for statsforecast: the baselines and the local models.

season_length comes from the config on every model. The library's defaults
assume daily or monthly data and would silently fit the wrong cycle.
"""
import numpy as np
import pandas as pd

from tsbench.data.schema import assert_weekly
from tsbench.models.base import AdapterError, ModelAdapter
from tsbench.models.registry import register

# statsforecast expresses intervals as central levels; these four cover
# q10..q90 in steps of 10, with q50 taken from the point forecast.
_LEVELS = [80, 60, 40, 20]


def _level_to_quantiles(level: int) -> tuple:
    tail = (100 - level) / 2
    return round(tail), round(100 - tail)


class StatsForecastAdapter(ModelAdapter):
    package = "statsforecast"
    """Shared plumbing. Subclasses supply `_model()` only."""

    @classmethod
    def preload(cls):
        import statsforecast  # noqa: F401  (numba JIT caches warm on first import)

    def _model(self):
        raise NotImplementedError

    @classmethod
    def is_available(cls):
        try:
            import statsforecast  # noqa: F401
        except ImportError as exc:
            return False, f"statsforecast is not installed ({exc})"
        return True, ""

    def fit(self, train_df: pd.DataFrame) -> None:
        from statsforecast import StatsForecast

        df = train_df[["unique_id", "ds", "y"]].dropna(subset=["y"])
        if df.empty:
            raise AdapterError(f"{self.name}: no observations to fit")

        self._freq = assert_weekly(df["ds"])
        self._sf = StatsForecast(
            models=[self._model()],
            freq=self._freq,
            n_jobs=self.cfg.run.n_jobs,
        )
        self._sf.fit(df)
        self._alias = self._sf.models[0].alias

    def predict(self, horizon: int) -> pd.DataFrame:
        levels = _LEVELS if self.supports_quantiles() else None
        raw = self._sf.predict(h=horizon, level=levels)
        return self._to_canonical(raw, levels)

    def _to_canonical(self, raw: pd.DataFrame, levels) -> pd.DataFrame:
        out = raw[["unique_id", "ds"]].copy()
        out["yhat"] = raw[self._alias].to_numpy()

        if not levels:
            return out

        out["yhat_q50"] = out["yhat"]
        for level in levels:
            lo_q, hi_q = _level_to_quantiles(level)
            out[f"yhat_q{lo_q}"] = raw[f"{self._alias}-lo-{level}"].to_numpy()
            out[f"yhat_q{hi_q}"] = raw[f"{self._alias}-hi-{level}"].to_numpy()

        q_cols = sorted((c for c in out.columns if c.startswith("yhat_q")),
                        key=lambda c: int(c[len("yhat_q"):]))
        # Interval methods can cross on short or flat history; sorting is the
        # standard repair and keeps the pinball/CRPS definitions valid.
        out[q_cols] = np.sort(out[q_cols].to_numpy(), axis=1)
        return out

    def supports_quantiles(self) -> bool:
        return True

    def supports_covariates(self) -> bool:
        return False


@register
class NaiveAdapter(StatsForecastAdapter):
    name = "naive"
    family = "baseline"

    def _model(self):
        from statsforecast.models import Naive
        return Naive()


@register
class SeasonalNaiveAdapter(StatsForecastAdapter):
    name = "seasonal_naive"
    family = "baseline"

    def _model(self):
        from statsforecast.models import SeasonalNaive
        # Also the MASE denominator's basis - a wrong season here would make
        # the baseline and the scaling disagree.
        return SeasonalNaive(season_length=self.season_length)


@register
class AutoETSAdapter(StatsForecastAdapter):
    name = "autoets"
    family = "local"

    def _model(self):
        from statsforecast.models import AutoETS
        return AutoETS(season_length=self.season_length, **self.params)


@register
class AutoARIMAAdapter(StatsForecastAdapter):
    name = "autoarima"
    family = "local"

    def _model(self):
        from statsforecast.models import AutoARIMA
        return AutoARIMA(season_length=self.season_length, **self.params)


@register
class AutoThetaAdapter(StatsForecastAdapter):
    name = "autotheta"
    family = "local"

    def _model(self):
        from statsforecast.models import AutoTheta
        return AutoTheta(season_length=self.season_length, **self.params)
