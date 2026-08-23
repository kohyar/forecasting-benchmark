"""Global neural models via neuralforecast.

input_size is one seasonal cycle (52 weeks), not the 336 that long-horizon
benchmarks use: series here hold ~200 observations, so a 336-week context
would leave nothing to train on. The horizon is baked into the architecture,
so these models refit per horizon - hence horizon_is_fit_time.
"""
import numpy as np
import pandas as pd

from tsbench.data.schema import PLANNED_PRICE, YAGO, assert_weekly
from tsbench.models.base import AdapterError, ModelAdapter
from tsbench.models.registry import register

_ACCELERATOR = {"cpu": "cpu", "mps": "mps", "cuda": "gpu"}


def _warm_torch_device():
    """Create the CUDA/MPS context now so the first timed fit does not pay it."""
    import torch

    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.zeros(1, device="mps")


def _quantile_column(alias: str, q: float) -> str:
    """MQLoss names outputs by central level: q=0.1 -> lo-80.0, q=0.9 -> hi-80.0."""
    if abs(q - 0.5) < 1e-9:
        return f"{alias}-median"
    level = round(abs(q - 0.5) * 200, 1)
    side = "lo" if q < 0.5 else "hi"
    return f"{alias}-{side}-{level}"


class NeuralAdapter(ModelAdapter):
    family = "global"
    package = "neuralforecast"
    horizon_is_fit_time = True

    @classmethod
    def preload(cls):
        import neuralforecast  # noqa: F401
        _warm_torch_device()

    #: neuralforecast class name, resolved lazily so the import stays optional
    model_class = ""

    @classmethod
    def is_available(cls):
        try:
            import neuralforecast  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            return False, f"neuralforecast/torch unavailable ({exc})"
        return True, ""

    def supports_quantiles(self) -> bool:
        return True

    def supports_covariates(self) -> bool:
        return False

    def _model_kwargs(self) -> dict:
        return {}

    def _build(self, horizon: int):
        import neuralforecast.models as nfm
        from neuralforecast.losses.pytorch import MQLoss

        cls = getattr(nfm, self.model_class)
        kwargs = {
            "h": horizon,
            "input_size": self.params.get("input_size", self.season_length),
            "max_steps": self.params.get("max_steps", 300),
            "learning_rate": self.params.get("learning_rate", 1e-3),
            "batch_size": self.params.get("batch_size", 32),
            # Series span six orders of magnitude, so a global model needs
            # per-series scaling or the large ones dominate the loss.
            "scaler_type": self.params.get("scaler_type", "robust"),
            "loss": MQLoss(quantiles=list(self.quantile_levels)),
            "enable_progress_bar": False,
            "logger": False,
            "accelerator": _ACCELERATOR.get(self.device, "cpu"),
            "random_seed": self.params.get("random_seed", 1),
            **self._model_kwargs(),
        }
        if self.supports_covariates() and self._use_covariates():
            kwargs["futr_exog_list"] = list(self._futr_cols)
        return cls(**kwargs)

    def _use_covariates(self) -> bool:
        return bool(self.params.get("use_covariates", False))

    def tuning_space(self, trial) -> dict:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "max_steps": trial.suggest_int("max_steps", 100, 800, step=100),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "scaler_type": trial.suggest_categorical("scaler_type", ["standard", "robust"]),
        }

    def fit(self, train_df: pd.DataFrame) -> None:
        from neuralforecast import NeuralForecast

        cols = ["unique_id", "ds", "y"]
        self._futr_cols = []
        if self.supports_covariates() and self._use_covariates():
            self._futr_cols = [c for c in YAGO + [PLANNED_PRICE, "week_of_year"]
                               if c in train_df.columns]
            cols += self._futr_cols

        df = train_df[cols].dropna(subset=["y"]).reset_index(drop=True)
        if df.empty:
            raise AdapterError(f"{self.name}: no observations to fit")
        if self._futr_cols:
            df[self._futr_cols] = df[self._futr_cols].fillna(0.0)

        self._freq = assert_weekly(df["ds"])
        self._horizon = None
        self._df = df
        self._nf = None

    def predict(self, horizon: int) -> pd.DataFrame:
        from neuralforecast import NeuralForecast

        if self._nf is None or self._horizon != horizon:
            self._nf = NeuralForecast(models=[self._build(horizon)], freq=self._freq)
            self._nf.fit(self._df, verbose=False)
            self._horizon = horizon

        futr_df = None
        if self._futr_cols:
            futr = getattr(self, "_future", None)
            if futr is None:
                raise AdapterError(f"{self.name}: future covariates were not supplied")
            futr_df = futr[["unique_id", "ds"] + self._futr_cols].copy()
            futr_df[self._futr_cols] = futr_df[self._futr_cols].fillna(0.0)

        raw = self._nf.predict(futr_df=futr_df)
        return self._to_canonical(raw)

    def _to_canonical(self, raw: pd.DataFrame) -> pd.DataFrame:
        alias = self.model_class
        out = raw[["unique_id", "ds"]].copy()
        out["yhat"] = raw[f"{alias}-median"].to_numpy()

        q_cols = []
        for q in self.quantile_levels:
            col = _quantile_column(alias, q)
            name = f"yhat_q{int(q * 100)}"
            out[name] = raw[col].to_numpy()
            q_cols.append(name)

        # Quantile heads can cross before convergence; sorting is the standard
        # repair and keeps pinball and CRPS well defined.
        out[q_cols] = np.sort(out[q_cols].to_numpy(), axis=1)
        return out

    @property
    def n_params(self):
        if self._nf is None:
            return None
        return int(sum(p.numel() for p in self._nf.models[0].parameters()))


@register
class NHITSAdapter(NeuralAdapter):
    name = "nhits"
    model_class = "NHITS"


@register
class PatchTSTAdapter(NeuralAdapter):
    name = "patchtst"
    model_class = "PatchTST"

    def _model_kwargs(self) -> dict:
        return {"patch_len": self.params.get("patch_len", 4),
                "stride": self.params.get("stride", 2)}


@register
class DLinearAdapter(NeuralAdapter):
    name = "dlinear"
    model_class = "DLinear"


@register
class TFTAdapter(NeuralAdapter):
    name = "tft"
    model_class = "TFT"

    def supports_covariates(self) -> bool:
        return True

    def _model_kwargs(self) -> dict:
        return {"hidden_size": self.params.get("hidden_size", 64),
                "n_head": self.params.get("n_head", 4)}

    def tuning_space(self, trial) -> dict:
        return {**super().tuning_space(trial),
                "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128])}
