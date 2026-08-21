"""Execute the benchmark.

One model failing must not kill the run, and a crash must resume rather than
restart - twelve hours lost to an OOM at model nine is twelve hours gone. Every
row carries the config hash, seed and tuning budget so it traces back to a run.
"""
import json
import os
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
from tsbench.measure import (
    Measurement,
    device_info,
    measure,
    peak_rss_mb,
    resolve_device,
)
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
    def __init__(self, cfg: Config, registry=None, mlflow_enabled: bool = True,
                 retry_failed: bool = True, force=()):
        self.cfg = cfg
        self.registry = registry or registry_module.default()
        self.device = resolve_device(cfg.run.device)
        self.out_dir = Path(cfg.run.output_dir) / cfg.run.name
        # checkpoints/<model>/<result_key>/fold<N>[__cov]__{metrics,timings}.parquet
        # The key covers only what determines *that* model's results, so a fix
        # to one model - or a move to another machine - never invalidates the
        # others' completed work.
        self.checkpoint_root = self.out_dir / "checkpoints"
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        # Scratch for worker hand-off lives on local disk, never under results:
        # results may sit on a network volume where thousands of small
        # transient files are slow, and scratch must not survive a crash.
        self.work_root = Path(cfg.run.work_dir or tempfile.gettempdir()) / "tsbench-work"
        self.mlflow_enabled = mlflow_enabled
        # A failed checkpoint is retried on the next run by default: a failure
        # usually means something to fix, and after the fix the model must run
        # again without anyone deleting files by hand.
        self.retry_failed = retry_failed
        # Models whose existing checkpoints are discarded before running -
        # for re-validating a model after its adapter changed.
        self.force = set(force or ())
        self._git_commit = _git_commit()

    def result_key(self, model: str, params: dict | None = None) -> str:
        return self.cfg.result_key(model, params)

    # -- orchestration ----------------------------------------------------

    def run(self, panel: pd.DataFrame, models=None, tuned_params=None) -> RunResult:
        models = list(models or self.cfg.models.enabled)
        tuned_params = tuned_params or {}

        splitter = RollingOriginSplitter(self.cfg)
        denominators = denominators_for_protocol(panel, self.cfg, splitter)
        folds = list(splitter.split(panel))

        metrics, timings = [], []
        for name in models:
            params = tuned_params.get(name, {})
            key = self.result_key(name, params)
            if name in self.force:
                self._discard_checkpoints(name, key)
            for use_covariates in self._covariate_arms(name):
                for fold in folds:
                    done = self._load_checkpoint(name, key, fold.fold_id, use_covariates)
                    if done is not None:
                        metrics.append(done["metrics"])
                        timings.append(done["timings"])
                        continue

                    m, t = self._run_model_fold(name, fold, denominators, params,
                                                use_covariates, key)
                    t = t.assign(resumed=False)
                    self._save_checkpoint(name, key, fold.fold_id, use_covariates, m, t)
                    metrics.append(m)
                    timings.append(t)
                    _progress(name, fold.fold_id, len(folds), use_covariates, t)

            # Flush after every model so partial results are on disk if the
            # run dies or the operator kills it.
            self._write(_concat(metrics), _concat(timings),
                        self._metadata(panel, splitter, models, tuned_params))

        metrics = _concat(metrics)
        timings = _concat(timings)
        metadata = self._metadata(panel, splitter, models, tuned_params)
        paths = self._write(metrics, timings, metadata)
        self._log_mlflow(metrics, timings, metadata)
        return RunResult(metrics=metrics, timings=timings, metadata=metadata, paths=paths)

    def _covariate_arms(self, name: str) -> list:
        """Capable models run twice when the ablation is on; everything else
        runs once, without."""
        if not self.cfg.models.covariate_ablation:
            return [False]
        adapter = self.registry.get(name)(self.cfg)
        return [False, True] if adapter.supports_covariates() else [False]

    def _run_model_fold(self, name: str, fold, denominators, params: dict,
                        use_covariates: bool = False, result_key: str = ""):
        train, self._impute_report = impute_gaps(fold.train)
        assert_frame_reaches_origin(train, fold.origin)
        adapter_cls = self.registry.get(name)
        horizon_at_fit = getattr(adapter_cls, "horizon_is_fit_time", False)
        horizons = list(self.cfg.protocol.horizons)
        params = {**params, "use_covariates": use_covariates} if use_covariates else params

        metrics, timings = [], []
        for repeat in range(self.cfg.run.n_repeats):
            seed = self.cfg.run.seed + repeat
            groups = [[h] for h in horizons] if horizon_at_fit else [horizons]

            for group in groups:
                fit_key = f"{name}|f{fold.fold_id}|r{repeat}|h{group[0] if horizon_at_fit else 'all'}"
                m, t = self._fit_and_predict(
                    name, adapter_cls, params, train, fold, group, repeat, seed,
                    denominators, fit_key, use_covariates, result_key)
                metrics.extend(m)
                timings.extend(t)
        return _concat(metrics), pd.DataFrame(timings)

    def _fit_and_predict(self, name, adapter_cls, params, train, fold, horizons,
                         repeat, seed, denominators, fit_key, use_covariates=False,
                         result_key=""):
        set_seeds(seed)
        common = {
            "model": name,
            "covariates": use_covariates,
            "result_key": result_key,
            "git_commit": self._git_commit,
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
        self.work_root.mkdir(parents=True, exist_ok=True)
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
                    "peak_rss_mb": result.get("peak_rss_mb"),
                    "checkpoint": result.get("checkpoint"),
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
        for col in ("repeat", "seed", "config_hash", "run_name", "tuning_trials",
                    "family", "covariates", "result_key", "git_commit"):
            rows[col] = common[col]
        return rows

    def _execute_inprocess(self, name, adapter_cls, params, train, fold, horizons,
                           denominators, common):
        metrics, timings = [], []
        try:
            adapter_cls.preload()
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
                    "peak_rss_mb": peak_rss_mb(),
                    "checkpoint": (getattr(model, "resolved_checkpoint", None)
                                   or getattr(model, "checkpoint", None)),
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

    def _checkpoint_paths(self, model: str, key: str, fold: int, use_covariates: bool) -> tuple:
        stem = f"fold{fold}" + ("__cov" if use_covariates else "")
        folder = self.checkpoint_root / model / key
        return folder / f"{stem}__metrics.parquet", folder / f"{stem}__timings.parquet"

    def _discard_checkpoints(self, model: str, key: str) -> None:
        folder = self.checkpoint_root / model / key
        if folder.exists():
            shutil.rmtree(folder)

    def checkpoint_files(self, model: str) -> list:
        folder = self.checkpoint_root / model
        return sorted(folder.rglob("*.parquet")) if folder.exists() else []

    def _save_checkpoint(self, model, key, fold, use_covariates, metrics, timings) -> None:
        m_path, t_path = self._checkpoint_paths(model, key, fold, use_covariates)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        # Metrics first, timings last: the timings file is the "done" marker,
        # and each is written to a temp name and renamed so a kill mid-write
        # can never leave a half-file under the real name.
        if len(metrics):
            _atomic_parquet(metrics, m_path)
        _atomic_parquet(timings, t_path)

    def _load_checkpoint(self, model, key, fold, use_covariates=False):
        m_path, t_path = self._checkpoint_paths(model, key, fold, use_covariates)
        if not t_path.exists():
            return None
        try:
            timings = pd.read_parquet(t_path)
            metrics = pd.read_parquet(m_path) if m_path.exists() else pd.DataFrame()
        except Exception:
            # A corrupt checkpoint is treated as absent, not as a crash.
            for path in (m_path, t_path):
                path.unlink(missing_ok=True)
            return None

        if self.retry_failed and (timings["status"] == "failed").any():
            for path in (m_path, t_path):
                path.unlink(missing_ok=True)
            return None
        return {"metrics": metrics, "timings": timings.assign(resumed=True)}

    # -- output -----------------------------------------------------------

    def _metadata(self, panel, splitter, models, tuned_params=None) -> dict:
        tuned_params = tuned_params or {}
        return {
            "progress": self.progress(models, tuned_params),
            "run_name": self.cfg.run.name,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config_hash": self.cfg.hash,
            "config": self.cfg.to_dict(),
            "seed": self.cfg.run.seed,
            "n_repeats": self.cfg.run.n_repeats,
            "execution": self.cfg.run.execution,
            "tuning_trials": self.cfg.tuning.budget_trials,
            "tsbench_version": __version__,
            "git_commit": self._git_commit,
            "packages": _package_versions(),
            "hardware": device_info(self.device),
            "python": platform.python_version(),
            "models": models,
            "covariate_ablation": self.cfg.models.covariate_ablation,
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

    def progress(self, models, tuned_params=None) -> dict:
        """Per model: how many (fold, arm) checkpoints exist, are clean, and
        which build produced them.

        result_key covers the config, not the code, so a checkpoint survives a
        rewrite of its own adapter. That is what makes resume usable, and it is
        also how two builds' timings end up in one table - so the commit each
        checkpoint carries is reported alongside the counts.
        """
        tuned_params = tuned_params or {}
        folds = self.cfg.protocol.folds
        out = {}
        for name in models:
            key = self.result_key(name, tuned_params.get(name, {}))
            arms = self._covariate_arms(name) if name in self.registry.names() else [False]
            complete = failed = 0
            commits = set()
            for arm in arms:
                for fold in range(folds):
                    _, t_path = self._checkpoint_paths(name, key, fold, arm)
                    if not t_path.exists():
                        continue
                    try:
                        t = pd.read_parquet(t_path, columns=["status", "git_commit"])
                        commits.update(c for c in t["git_commit"].dropna().unique() if c)
                    except Exception:
                        # Checkpoints predate the provenance column: still work,
                        # just unattributable.
                        try:
                            t = pd.read_parquet(t_path, columns=["status"])
                        except Exception:
                            continue
                    if (t["status"] == "failed").any():
                        failed += 1
                    else:
                        complete += 1
            out[name] = {"expected": folds * len(arms), "complete": complete,
                         "failed": failed, "result_key": key,
                         "commits": sorted(commits),
                         # Unknown when git is unavailable, which is not stale.
                         "stale": bool(self._git_commit and commits
                                       and any(c != self._git_commit for c in commits))}
        return out

    def stale_models(self, progress: dict) -> dict:
        """Models whose checkpoints were measured by a different build."""
        return {name: p["commits"] for name, p in progress.items() if p["stale"]}

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
            experiment = self.cfg.run.name
            if os.environ.get("DATABRICKS_RUNTIME_VERSION") and not experiment.startswith("/"):
                experiment = f"/Shared/tsbench/{experiment}"
            mlflow.set_experiment(experiment)
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


def _progress(name, fold_id, n_folds, use_covariates, timings) -> None:
    """One line per completed (model, fold), flushed so long runs are not silent."""
    ok = timings[timings["status"] == "ok"] if len(timings) else timings
    failed = int((timings["status"] == "failed").sum()) if len(timings) else 0
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    arm = "+cov" if use_covariates else ""
    if len(ok):
        fit = ok.drop_duplicates("fit_key")["fit_seconds"].median()
        pred = ok["predict_seconds"].median()
        detail = f"fit {fit:7.1f}s  predict {pred:6.2f}s"
    else:
        detail = "no successful units"
    print(f"[{stamp}] {name:18s}{arm:5s} fold {fold_id + 1}/{n_folds}  {detail}"
          + (f"  ({failed} failed)" if failed else ""), flush=True)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def collect(cfg: Config, models=None, registry=None, tuned_params=None) -> RunResult:
    """Assemble results from whatever checkpoints exist, running nothing.

    For looking at a run in progress, or rebuilding the aggregate files after
    a crash without waiting for the remaining models.
    """
    runner = BenchmarkRunner(cfg, registry=registry, mlflow_enabled=False,
                             retry_failed=False)
    models = list(models or cfg.models.enabled)
    tuned_params = tuned_params or {}

    metrics, timings = [], []
    for name in models:
        key = runner.result_key(name, tuned_params.get(name, {}))
        arms = runner._covariate_arms(name) if name in runner.registry.names() else [False]
        for arm in arms:
            for fold in range(cfg.protocol.folds):
                done = runner._load_checkpoint(name, key, fold, arm)
                if done is not None:
                    metrics.append(done["metrics"])
                    timings.append(done["timings"])

    metrics, timings = _concat(metrics), _concat(timings)
    progress = runner.progress(models, tuned_params)
    # collect() is where measurements from different builds silently become one
    # table, so this is the last place to say so before the numbers are used.
    warn_if_stale(progress, runner._git_commit)
    metadata = {
        "run_name": cfg.run.name,
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_hash": cfg.hash,
        "config": cfg.to_dict(),
        "models": models,
        "progress": progress,
        "note": "assembled from checkpoints by collect(); no models were run",
    }
    paths = runner._write(metrics, timings, metadata)
    return RunResult(metrics=metrics, timings=timings, metadata=metadata, paths=paths)


def _empty_measurement(device: str, train: pd.DataFrame) -> dict:
    return {
        "fit_seconds": np.nan, "predict_seconds": np.nan, "peak_memory_mb": np.nan,
        "n_series": int(train["unique_id"].nunique()), "n_params": None,
        "peak_rss_mb": np.nan, "fit_reused": False, **device_info(device),
    }


def _package_versions() -> dict:
    out = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def warn_if_stale(progress: dict, head: str) -> list:
    """Name the models whose checkpoints were measured by a different build.

    Resuming them is still correct - the config that determines their results
    has not moved. What is not correct is reading their timings next to a
    freshly measured model's, so the warning says how to re-measure.
    """
    stale = {name: p["commits"] for name, p in progress.items() if p["stale"]}
    if not stale:
        return []

    print(f"WARNING: {len(stale)} model(s) carry checkpoints measured by a different "
          f"build than HEAD ({head[:8]}); their timings are not comparable with "
          f"freshly measured models.")
    for name, commits in stale.items():
        print(f"  {name:18s} built at {', '.join(c[:8] for c in commits)}")
    print(f"  re-measure with: --models {','.join(stale)} --force")
    return list(stale)


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
