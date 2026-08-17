"""The one interface every model presents, whatever library is behind it.

Foundation models have a no-op fit(); local models fit per series internally.
That difference stays inside the adapter - the runner never sees it.
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

FAMILIES = ("baseline", "local", "global", "foundation")


class AdapterError(RuntimeError):
    pass


class ModelAdapter(ABC):
    name: str = ""
    family: str = ""
    # Zero-shot models get a tuning budget of 0 by definition; the results
    # report that rather than leaving it blank.
    tunable: bool = True
    # Neural models bake the horizon into the architecture, so they refit per
    # horizon; everything else fits once and predicts at both.
    horizon_is_fit_time: bool = False

    def __init__(self, cfg, params: dict | None = None, device: str = "cpu"):
        self.cfg = cfg
        self.params = dict(params or {})
        self.device = device
        self.season_length = cfg.data.season_length
        self.quantile_levels = list(cfg.metrics.quantile_levels)

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        """Train. Zero-shot models leave this empty."""

    @abstractmethod
    def predict(self, horizon: int) -> pd.DataFrame:
        """unique_id, ds, yhat [, yhat_q10 ... yhat_q90]."""

    def supports_quantiles(self) -> bool:
        return False

    def supports_covariates(self) -> bool:
        return False

    @classmethod
    def is_available(cls) -> tuple:
        """(available, reason). Adapters whose library is missing report it
        rather than raising at import time."""
        return True, ""

    def tuning_space(self, trial) -> dict:
        """Optuna search space. Empty means nothing to tune - zero-shot models
        report a budget of 0 rather than leaving it blank."""
        return {}

    @property
    def n_params(self):
        return None

    def set_future_covariates(self, future_df: pd.DataFrame) -> None:
        """Known-future values for the forecast window, from the splitter."""
        self._future = future_df


def validate_prediction(pred: pd.DataFrame, expected_ids, expected_ds,
                        quantile_levels=None) -> None:
    """Guard the runner against a misbehaving adapter."""
    if not isinstance(pred, pd.DataFrame):
        raise AdapterError(f"predict() must return a DataFrame, got {type(pred).__name__}")

    missing_cols = {"unique_id", "ds", "yhat"} - set(pred.columns)
    if missing_cols:
        raise AdapterError(f"predict() is missing column(s): {sorted(missing_cols)}")

    if pred.duplicated(["unique_id", "ds"]).any():
        n = int(pred.duplicated(["unique_id", "ds"]).sum())
        raise AdapterError(f"predict() returned {n} duplicate (unique_id, ds) row(s)")

    got_ids, want_ids = set(pred["unique_id"]), set(expected_ids)
    if got_ids != want_ids:
        raise AdapterError(
            f"predict() returned {len(got_ids)} series, expected {len(want_ids)}: "
            f"missing {sorted(want_ids - got_ids)[:3]}, unexpected {sorted(got_ids - want_ids)[:3]}")

    got_ds, want_ds = set(pd.to_datetime(pred["ds"])), set(pd.to_datetime(expected_ds))
    if got_ds != want_ds:
        raise AdapterError(
            f"predict() returned unexpected timestamp(s): "
            f"missing {sorted(want_ds - got_ds)[:3]}, unexpected {sorted(got_ds - want_ds)[:3]}")

    if pred["yhat"].isna().all():
        raise AdapterError("predict() returned all-NaN forecasts")

    q_cols = sorted((c for c in pred.columns if c.startswith("yhat_q")),
                    key=lambda c: int(c[len("yhat_q"):]))
    if len(q_cols) > 1:
        values = pred[q_cols].to_numpy(dtype=float)
        ok = np.isfinite(values).all(axis=1)
        if (np.diff(values[ok], axis=1) < -1e-9).any():
            raise AdapterError(f"predict() returned crossed quantiles across {q_cols}")
