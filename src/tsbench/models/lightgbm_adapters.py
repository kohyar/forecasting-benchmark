"""LightGBM twice: fitted per series, and fitted across all series.

Same algorithm, same features, same hyperparameters, same tuning budget. Scope
is the only variable - that is what makes this an ablation rather than two
entries in a table. Both adapters build their MLForecast from the one function
below, so the configurations cannot drift apart.
"""
import numpy as np
import pandas as pd

from tsbench.data.schema import assert_weekly
from tsbench.models.base import AdapterError, ModelAdapter
from tsbench.models.registry import register

DEFAULT_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "verbosity": -1,
}


def _feature_recipe(season_length: int) -> dict:
    """One recipe for both scopes."""
    from mlforecast.lag_transforms import RollingMean

    return {
        "lags": [1, 2, 3, 4, 13, 26, season_length],
        "lag_transforms": {1: [RollingMean(4), RollingMean(13), RollingMean(season_length)]},
        "date_features": ["week", "month", "quarter"],
    }


def _build(models, season_length: int, freq: str, num_threads: int):
    from mlforecast import MLForecast

    return MLForecast(models=models, freq=freq, num_threads=num_threads,
                      **_feature_recipe(season_length))


class LightGBMBase(ModelAdapter):
    family = "global"

    @classmethod
    def is_available(cls):
        try:
            import lightgbm  # noqa: F401
            import mlforecast  # noqa: F401
        except (ImportError, OSError) as exc:
            return False, f"lightgbm/mlforecast unavailable ({exc})"
        return True, ""

    def _estimator(self):
        from lightgbm import LGBMRegressor
        params = {**DEFAULT_PARAMS, **self.params}
        return LGBMRegressor(random_state=self.params.get("random_state", 0), **params)

    def tuning_space(self, trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        }

    def supports_covariates(self) -> bool:
        return True

    def _prepare(self, train_df: pd.DataFrame) -> pd.DataFrame:
        df = train_df[["unique_id", "ds", "y"]].dropna(subset=["y"])
        if df.empty:
            raise AdapterError(f"{self.name}: no observations to fit")
        self._freq = assert_weekly(df["ds"])
        return df

    @staticmethod
    def _rename(pred: pd.DataFrame, alias: str) -> pd.DataFrame:
        out = pred[["unique_id", "ds"]].copy()
        out["yhat"] = pred[alias].to_numpy()
        return out


@register
class LightGBMGlobalAdapter(LightGBMBase):
    """One model fitted across every series."""

    name = "lightgbm_global"
    family = "global"

    def fit(self, train_df: pd.DataFrame) -> None:
        df = self._prepare(train_df)
        self._model = _build([self._estimator()], self.season_length, self._freq,
                             num_threads=max(self.cfg.run.n_jobs, 1))
        self._model.fit(df, static_features=[])
        self._alias = next(iter(self._model.models))

    def predict(self, horizon: int) -> pd.DataFrame:
        return self._rename(self._model.predict(h=horizon), self._alias)

    @property
    def n_params(self):
        booster = getattr(self._model.models_[self._alias], "booster_", None)
        return booster.num_trees() if booster is not None else None


@register
class LightGBMLocalAdapter(LightGBMBase):
    """The same model fitted once per series."""

    name = "lightgbm_local"
    family = "local"

    def fit(self, train_df: pd.DataFrame) -> None:
        df = self._prepare(train_df)
        self._models = {}
        self._failed = []
        for uid, g in df.groupby("unique_id", sort=True):
            try:
                model = _build([self._estimator()], self.season_length, self._freq,
                               num_threads=1)
                model.fit(g, static_features=[])
                self._models[uid] = model
                self._alias = next(iter(model.models))
            except Exception as exc:  # a single short series must not sink the fold
                self._failed.append((uid, str(exc)))
        if not self._models:
            raise AdapterError(f"{self.name}: every series failed to fit")
        self._ids = sorted(df["unique_id"].unique())

    def predict(self, horizon: int) -> pd.DataFrame:
        out = []
        for uid in self._ids:
            model = self._models.get(uid)
            if model is None:
                continue
            out.append(self._rename(model.predict(h=horizon), self._alias))
        return pd.concat(out, ignore_index=True)

    @property
    def n_params(self):
        trees = [m.models_[self._alias].booster_.num_trees() for m in self._models.values()]
        return int(np.sum(trees)) if trees else None
