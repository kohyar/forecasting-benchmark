"""Execute the benchmark.

One model failing must not kill the run, and a crash must resume rather than
restart - twelve hours lost to an OOM at model nine is twelve hours gone. Every
row carries the config hash, seed and tuning budget so it traces back to a run.
"""
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from tsbench import __version__
from tsbench.config import Config
from tsbench.data.prepare import impute_gaps
from tsbench.eval.metrics import denominators_for_protocol, evaluate
from tsbench.eval.splitter import ProtocolError, RollingOriginSplitter
from tsbench.measure import Measurement, device_info, measure, resolve_device
from tsbench.models import registry as registry_module
from tsbench.models.base import validate_prediction
from tsbench.seeding import set_seeds

TRACKED_PACKAGES = [
    "pandas", "numpy", "scipy", "statsforecast", "mlforecast", "neuralforecast",
    "lightgbm", "prophet", "torch", "optuna", "mlflow",
]


@dataclass
class RunResult:
    metrics: pd.DataFrame
    timings: pd.DataFrame
    metadata: dict
    paths: dict = field(default_factory=dict)


class BenchmarkRunner:
    def __init__(self, cfg: Config, registry=None, mlflow_enabled: bool = True):
        self.cfg = cfg
        self.registry = registry or registry_module.default()
        self.device = resolve_device(cfg.run.device)
        self.out_dir = Path(cfg.run.output_dir) / cfg.run.name
        # Keyed by config hash: a protocol change must not inherit the previous
        # run's completed work.
        self.checkpoint_dir = self.out_dir / "checkpoints" / cfg.hash
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.work_root = self.out_dir / "work"
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.mlflow_enabled = mlflow_enabled

    # -- orchestration ----------------------------------------------------

    def run(self, panel: pd.DataFrame, models=None, tuned_params=None) -> RunResult:
        models = list(models or self.cfg.models.enabled)
        tuned_params = tuned_params or {}

        splitter = RollingOriginSplitter(self.cfg)
        denominators = denominators_for_protocol(panel, self.cfg, splitter)
        folds = list(splitter.split(panel))

        metrics, timings = [], []
        for name in models:
            for fold in folds:
                done = self._load_checkpoint(name, fold.fold_id)
                if done is not None:
                    metrics.append(done["metrics"])
                    timings.append(done["timings"])
                    continue

                m, t = self._run_model_fold(name, fold, denominators,
                                            tuned_params.get(name, {}))
                t = t.assign(resumed=False)
                self._save_checkpoint(name, fold.fold_id, m, t)
                metrics.append(m)
                timings.append(t)

        metrics = _concat(metrics)
        timings = _concat(timings)
        metadata = self._metadata(panel, splitter, models)
        paths = self._write(metrics, timings, metadata)
        self._log_mlflow(metrics, timings, metadata)
        return RunResult(metrics=metrics, timings=timings, metadata=metadata, paths=paths)

    def _run_model_fold(self, name: str, fold, denominators, params: dict):
        train, self._impute_report = impute_gaps(fold.train)
        assert_frame_reaches_origin(train, fold.origin)
        adapter_cls = self.registry.get(name)
        horizon_at_fit = getattr(adapter_cls, "horizon_is_fit_time", False)
        horizons = list(self.cfg.protocol.horizons)

        metrics, timings = [], []
        for repeat in range(self.cfg.run.n_repeats):
            seed = self.cfg.run.seed + repeat
            groups = [[h] for h in horizons] if horizon_at_fit else [horizons]

            for group in groups:
                fit_key = f"{name}|f{fold.fold_id}|r{repeat}|h{group[0] if horizon_at_fit else 'all'}"
                m, t = self._fit_and_predict(
                    name, adapter_cls, params, train, fold, group, repeat, seed,
                    denominators, fit_key)
                metrics.extend(m)
                timings.extend(t)
        return _concat(metrics), pd.DataFrame(timings)

    def _fit_and_predict(self, name, adapter_cls, params, train, fold, horizons,
                         repeat, seed, denominators, fit_key):
        set_seeds(seed)
        common = {
            "model": name,
            "family": adapter_cls.family,
            "fold": fold.fold_id,
            "origin": fold.origin,
            "repeat": repeat,
            "seed": seed,
            "fit_key": fit_key,
            "config_hash": self.cfg.hash,
            "run_name": self.cfg.run.name,
            "tuning_trials": (self.cfg.tuning.budget_trials
                              if getattr(adapter_cls, "tunable", True) else 0),
        }
        return self._execute(name, adapter_cls, params, train, fold, horizons,
                             denominators, common)

    def _execute(self, name, adapter_cls, params, train, fold, horizons,
                 denominators, common):
        if self.cfg.run.execution == "subprocess":
            return self._execute_subprocess(name, adapter_cls, params, train, fold,
                                            horizons, denominators, common)
        return self._execute_inprocess(name, adapter_cls, params, train, fold,
                                       horizons, denominators, common)

    def _execute_subprocess(self, name, adapter_cls, params, train, fold, horizons,
                            denominators, common):
        """Hand the work to a child process so a segfault or an OOM kill is a
        recorded failure rather than the end of the run."""
        workdir = Path(tempfile.mkdtemp(prefix=f"tsbench-{name}-", dir=self.work_root))
        try:
            train.to_parquet(workdir / "train.parquet", index=False)
            for h in horizons:
                fold.future_covariates(h).to_parquet(
                    workdir / f"future_{h}.parquet", index=False)
            (workdir / "spec.json").write_text(json.dumps({
                "config": self.cfg.to_dict(),
                "module": adapter_cls.__module__,
                "class": adapter_cls.__name__,
                "params": params,
                "horizons": list(horizons),
                "device": self.device,
                "seed": common["seed"],
            }, default=str))

            proc = subprocess.run(
                [sys.executable, "-m", "tsbench.worker", str(workdir)],
                capture_output=True, text=True)

            if proc.returncode != 0:
                return [], self._worker_failure(workdir, proc, horizons, common, train)

            result = json.loads((workdir / "result.json").read_text())
            return self._collect_worker_output(workdir, result, name, fold, horizons,
                                               denominators, common, train)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _worker_failure(self, workdir, proc, horizons, common, train) -> list:
        error_path = workdir / "error.json"
        if error_path.exists():
            detail = json.loads(error_path.read_text())
            error, tb = detail["error"], detail["traceback"]
        else:
            signal = f"worker exited with code {proc.returncode}"
            if proc.returncode < 0 or proc.returncode == 139:
                signal += " (segfault or killed - not catchable in-process)"
            error, tb = signal, (proc.stderr or "")[-4000:]
        return [{**common, "horizon": h, "status": "failed", "stage": "worker",
                 "error": error, "traceback": tb, "worker_modules": None,
                 **_empty_measurement(self.device, train)} for h in horizons]

    def _collect_worker_output(self, workdir, result, name, fold, horizons,
                               denominators, common, train):
        metrics, timings = [], []
        modules = ",".join(result.get("modules", []))
        adapter_module = result.get("adapter_module", "")
        for i, h in enumerate(horizons):
            try:
                pred = pd.read_parquet(workdir / f"pred_{h}.parquet")
                expected_ds = pd.date_range(fold.origin + pd.Timedelta(days=7),
                                            periods=h, freq="7D")
                validate_prediction(pred, sorted(train["unique_id"].unique()), expected_ds)

                rows = self._score(pred, fold, h, denominators, name, common)
                metrics.append(rows)

                per_h = result["horizons"][str(h)]
                timings.append({
                    **common, "horizon": h, "status": "ok", "stage": "", "error": "",
                    "traceback": "", "fit_reused": i > 0,
                    "fit_seconds": result["fit_seconds"],
                    "predict_seconds": per_h["predict_seconds"],
                    "peak_memory_mb": max(result["fit_peak_memory_mb"],
                                          per_h["predict_peak_memory_mb"]),
                    "n_series": result["n_series"],
                    "n_params": result.get("n_params"),
                    "n_jobs": self.cfg.run.n_jobs,
                    "worker_modules": modules,
                    "worker_adapter_module": adapter_module,
                    **device_info(result.get("device", self.device)),
                })
            except Exception as exc:
                timings.append({**common, "horizon": h, "status": "failed",
                                "stage": "collect",
                                "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc(),
                                "worker_modules": modules,
                                **_empty_measurement(self.device, train)})
        return metrics, timings

    def _score(self, pred, fold, horizon, denominators, name, common):
        actual = fold.test(horizon)[["unique_id", "ds", "y"]]
        rows = evaluate(actual, pred, denominators,
                        model=name, fold=fold.fold_id, horizon=horizon,
                        quantile_levels=self.cfg.metrics.quantile_levels,
                        coverage_levels=self.cfg.metrics.coverage_levels)
        for col in ("repeat", "seed", "config_hash", "run_name", "tuning_trials", "family"):
            rows[col] = common[col]
        return rows

    def _execute_inprocess(self, name, adapter_cls, params, train, fold, horizons,
                           denominators, common):
        metrics, timings = [], []
        try:
            model = adapter_cls(self.cfg, params=params, device=self.device)
            with measure(self.device) as fit_m:
                model.fit(train)
        except Exception as exc:
            for h in horizons:
                timings.append({**common, "horizon": h, "status": "failed",
                                "stage": "fit", "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc(),
                                "worker_modules": None,
                                **_empty_measurement(self.device, train)})
            return metrics, timings

        for i, h in enumerate(horizons):
            try:
                model.set_future_covariates(fold.future_covariates(h))
                with measure(self.device) as pred_m:
                    pred = model.predict(h)

                expected_ds = pd.date_range(fold.origin + pd.Timedelta(days=7),
                                            periods=h, freq="7D")
                validate_prediction(pred, sorted(train["unique_id"].unique()), expected_ds)

                metrics.append(self._score(pred, fold, h, denominators, name, common))

                timings.append({
                    **common, "horizon": h, "status": "ok", "stage": "", "error": "",
                    "traceback": "",
                    "fit_reused": i > 0,
                    **Measurement.combine(fit_m, pred_m, self.device,
                                          n_series=train["unique_id"].nunique(),
                                          n_params=model.n_params),
                    "n_jobs": self.cfg.run.n_jobs,
                    "worker_modules": None,
                })
            except Exception as exc:
                timings.append({**common, "horizon": h, "status": "failed",
                                "stage": "predict",
                                "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc(),
                                "worker_modules": None,
                                **_empty_measurement(self.device, train)})
        return metrics, timings

    # -- checkpointing ----------------------------------------------------

    def _checkpoint_paths(self, model: str, fold: int) -> tuple:
        stem = f"{model}__fold{fold}"
        return (self.checkpoint_dir / f"{stem}__metrics.parquet",
                self.checkpoint_dir / f"{stem}__timings.parquet")

    def _save_checkpoint(self, model, fold, metrics, timings) -> None:
        m_path, t_path = self._checkpoint_paths(model, fold)
        if len(metrics):
            metrics.to_parquet(m_path, index=False)
        timings.to_parquet(t_path, index=False)

    def _load_checkpoint(self, model, fold):
        m_path, t_path = self._checkpoint_paths(model, fold)
        if not t_path.exists():
            return None
        metrics = pd.read_parquet(m_path) if m_path.exists() else pd.DataFrame()
        timings = pd.read_parquet(t_path)
        return {"metrics": metrics, "timings": timings.assign(resumed=True)}

    # -- output -----------------------------------------------------------

    def _metadata(self, panel, splitter, models) -> dict:
        return {
            "run_name": self.cfg.run.name,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config_hash": self.cfg.hash,
            "config": self.cfg.to_dict(),
            "seed": self.cfg.run.seed,
            "n_repeats": self.cfg.run.n_repeats,
            "execution": self.cfg.run.execution,
            "tuning_trials": self.cfg.tuning.budget_trials,
            "tsbench_version": __version__,
            "git_commit": _git_commit(),
            "packages": _package_versions(),
            "hardware": device_info(self.device),
            "python": platform.python_version(),
            "models": models,
            "model_status": [self.registry.status(m) for m in models
                             if m in self.registry.names()],
            "panel": {
                "n_series": int(panel["unique_id"].nunique()),
                "n_rows": int(len(panel)),
                "start": str(panel["ds"].min().date()),
                "end": str(panel["ds"].max().date()),
            },
            "protocol": {
                "folds": self.cfg.protocol.folds,
                "step_weeks": self.cfg.protocol.step,
                "horizons": self.cfg.protocol.horizons,
                "season_length": self.cfg.data.season_length,
                "origins": [str(o.date()) for o in splitter.origins(panel)],
                "overlapping_test_windows": self.cfg.protocol.step < max(self.cfg.protocol.horizons),
            },
            "imputation": getattr(self, "_impute_report", {"method": "linear"}),
        }

    def _write(self, metrics, timings, metadata) -> dict:
        paths = {
            "metrics": self.out_dir / "metrics.parquet",
            "timings": self.out_dir / "timings.parquet",
            "metadata": self.out_dir / "run_metadata.json",
        }
        if len(metrics):
            metrics.to_parquet(paths["metrics"], index=False)
        if len(timings):
            timings.to_parquet(paths["timings"], index=False)
        paths["metadata"].write_text(json.dumps(metadata, indent=2, default=str))
        return paths

    def _log_mlflow(self, metrics, timings, metadata) -> None:
        if not self.mlflow_enabled or not len(metrics):
            return
        try:
            import mlflow
        except ImportError:
            return
        try:
            mlflow.set_experiment(self.cfg.run.name)
            with mlflow.start_run(run_name=f"{self.cfg.run.name}-{metadata['config_hash']}"):
                mlflow.log_params({
                    "config_hash": self.cfg.hash,
                    "seed": self.cfg.run.seed,
                    "tuning_trials": self.cfg.tuning.budget_trials,
                    "season_length": self.cfg.data.season_length,
                    "folds": self.cfg.protocol.folds,
                })
                summary = (metrics[metrics["metric"] == "MASE"]
                           .groupby(["model", "horizon"])["value"].median())
                for (model, horizon), value in summary.items():
                    if np.isfinite(value):
                        mlflow.log_metric(f"MASE_{model}_h{horizon}", float(value))
        except Exception:
            # Parquet is the system of record; MLflow is a convenience.
            pass


def assert_frame_reaches_origin(train: pd.DataFrame, origin) -> None:
    """Every series must carry a value at the origin.

    Libraries anchor their forecast on each series' last observation, so a
    series that stops short silently forecasts the wrong weeks.
    """
    observed = train[train["y"].notna()]
    last = observed.groupby("unique_id")["ds"].max()
    short = last[last < origin]
    if len(short):
        raise ProtocolError(
            f"{len(short)} series end before the origin {pd.Timestamp(origin).date()} "
            f"(earliest {short.min().date()}); they would anchor their forecast on the "
            f"wrong week. Example: {short.index[0]}")


def _empty_measurement(device: str, train: pd.DataFrame) -> dict:
    return {
        "fit_seconds": np.nan, "predict_seconds": np.nan, "peak_memory_mb": np.nan,
        "n_series": int(train["unique_id"].nunique()), "n_params": None,
        "fit_reused": False, **device_info(device),
    }


def _package_versions() -> dict:
    out = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent).decode().strip()
    except Exception:
        return ""


def _concat(frames) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
