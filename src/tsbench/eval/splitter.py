"""Rolling-origin evaluation with an expanding training window.

Origins are spaced `step` weeks apart, which is deliberately shorter than the
longest horizon: spacing them a full horizon apart would consume too much
history and leave the earliest fold under two seasonal cycles. The cost is
overlapping test windows and therefore correlated fold errors, which the
statistical tests must correct for - `Fold.metadata` carries the spacing so
that correction can be justified in the paper.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tsbench.config import Config
from tsbench.data.schema import (
    PLANNED_PRICE,
    PRICE,
    YAGO,
    YAGO_LAG_WEEKS,
    covariate_groups,
)

WEEK = pd.Timedelta(days=7)


class ProtocolError(ValueError):
    pass


@dataclass
class Fold:
    fold_id: int
    origin: pd.Timestamp
    train: pd.DataFrame
    metadata: dict
    _panel: pd.DataFrame = field(repr=False, default=None)
    _target: str = field(repr=False, default="Dollars")

    def test(self, horizon: int) -> pd.DataFrame:
        """Actuals for the forecast window. Missing weeks stay as NaN rows so
        every series contributes exactly `horizon` rows."""
        lo, hi = self.origin + WEEK, self.origin + horizon * WEEK
        window = self._grid(lo, hi)
        actual = self._panel[self._panel["ds"].between(lo, hi)]
        return window.merge(actual, on=["unique_id", "ds"], how="left")

    def future_covariates(self, horizon: int) -> pd.DataFrame:
        """Everything a model may legitimately see for the forecast window.

        The _Yago lags are rebuilt from the training window rather than read
        from the export's future rows - same values, but by construction no
        row dated after the origin is ever touched.
        """
        lo, hi = self.origin + WEEK, self.origin + horizon * WEEK
        future = self._grid(lo, hi)

        for col in YAGO:
            base = col[: -len("_Yago")]
            source = "y" if base == self._target else base
            if source not in self.train.columns:
                continue
            hist = self.train.set_index(["unique_id", "ds"])[source]
            keys = pd.MultiIndex.from_arrays(
                [future["unique_id"], future["ds"] - YAGO_LAG_WEEKS * WEEK])
            future[col] = hist.reindex(keys).to_numpy()

        if PRICE in self.train.columns:
            last_price = (self.train.dropna(subset=[PRICE]).sort_values("ds")
                          .groupby("unique_id")[PRICE].last())
            future[PLANNED_PRICE] = future["unique_id"].map(last_price)

        future["week_of_year"] = future["ds"].dt.isocalendar().week.astype(int)

        statics = [c for c in covariate_groups(self._target).static if c in self.train.columns]
        if statics:
            per_series = self.train.groupby("unique_id", as_index=False)[statics].last()
            future = future.merge(per_series, on="unique_id", how="left")
        return future

    def _grid(self, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
        ids = self.train["unique_id"].unique()
        ds = pd.date_range(lo, hi, freq="7D")
        return pd.DataFrame({
            "unique_id": np.repeat(ids, len(ds)),
            "ds": np.tile(ds.to_numpy(), len(ids)),
        })


class RollingOriginSplitter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.folds = cfg.protocol.folds
        self.step = cfg.protocol.step
        self.horizons = list(cfg.protocol.horizons)
        self.max_horizon = max(self.horizons)
        self.min_train_weeks = cfg.protocol.min_train_weeks
        self.season_length = cfg.data.season_length

    def origins(self, panel: pd.DataFrame) -> list:
        last = panel["ds"].max() - self.max_horizon * WEEK
        first = last - (self.folds - 1) * self.step * WEEK
        earliest_data = panel["ds"].min()

        if first <= earliest_data:
            raise ProtocolError(
                f"history too short for {self.folds} folds of step {self.step} with horizon "
                f"{self.max_horizon}: earliest origin {first.date()} precedes the first "
                f"observation {earliest_data.date()}")
        return [first + k * self.step * WEEK for k in range(self.folds)]

    def split(self, panel: pd.DataFrame):
        for fold_id, origin in enumerate(self.origins(panel)):
            train = panel[panel["ds"] <= origin].reset_index(drop=True)
            yield Fold(
                fold_id=fold_id,
                origin=origin,
                train=train,
                metadata={
                    "fold_id": fold_id,
                    "origin": origin,
                    "folds": self.folds,
                    "step_weeks": self.step,
                    "horizons": self.horizons,
                    "season_length": self.season_length,
                    "overlapping_test_windows": self.step < self.max_horizon,
                    "n_train_weeks": int((origin - panel["ds"].min()).days // 7 + 1),
                },
                _panel=panel,
                _target=self.cfg.data.target,
            )

    def eligibility(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Per-series verdict on whether the protocol can run on it."""
        origins = self.origins(panel)
        earliest, last = origins[0], origins[-1]
        span_end = last + self.max_horizon * WEEK

        observed = panel[panel["y"].notna()]
        train_weeks = (observed[observed["ds"] <= earliest]
                       .groupby("unique_id")["ds"].count())
        last_obs = observed.groupby("unique_id")["ds"].max()

        report = pd.DataFrame({"unique_id": panel["unique_id"].unique()}).set_index("unique_id")
        report["train_weeks_at_earliest_origin"] = train_weeks.reindex(report.index).fillna(0).astype(int)
        report["last_observation"] = last_obs.reindex(report.index)
        report["meets_min_train"] = report["train_weeks_at_earliest_origin"] >= self.min_train_weeks
        report["covers_evaluation_span"] = report["last_observation"] >= span_end
        report["eligible"] = report["meets_min_train"] & report["covers_evaluation_span"]
        return report.reset_index()

    def assert_protocol_feasible(self, panel: pd.DataFrame) -> dict:
        """Fail before a 12-hour run rather than during it."""
        report = self.eligibility(panel)
        short = report[~report["meets_min_train"]]
        uncovered = report[~report["covers_evaluation_span"]]

        if len(short) or len(uncovered):
            raise ProtocolError(
                f"{len(short)} series below min_train_weeks={self.min_train_weeks} "
                f"(minimum found: {int(report['train_weeks_at_earliest_origin'].min())} weeks); "
                f"{len(uncovered)} series do not cover the evaluation span. "
                f"Drop them with eligibility() before running.")
        return {
            "n_series": len(report),
            "min_train_weeks_found": int(report["train_weeks_at_earliest_origin"].min()),
            "origins": [o.date().isoformat() for o in self.origins(panel)],
        }
