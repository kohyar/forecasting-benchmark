"""Equal-budget hyperparameter search.

Every tunable model gets the same number of Optuna trials, taken from the
config and recorded on every output row. Tuning the neural models while
running AutoARIMA on defaults invalidates a comparison, and reviewers check.

The search runs entirely before the first forecast origin: an inner origin is
placed one long horizon earlier, so the window used to score a trial is itself
training data for fold 0 and no test observation is ever consulted.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tsbench.config import Config
from tsbench.data.prepare import impute_gaps
from tsbench.eval.metrics import evaluate, seasonal_denominators
from tsbench.eval.splitter import RollingOriginSplitter
from tsbench.models import registry as registry_module
from tsbench.models.base import validate_prediction
from tsbench.seeding import set_seeds

WEEK = pd.Timedelta(days=7)
OBJECTIVE = "mean_MASE"


@dataclass
class TuningResult:
    best: dict
    best_value: dict
    budget: dict
    trials: pd.DataFrame
    path: Path = None
    meta: dict = field(default_factory=dict)


def tuning_origin(panel: pd.DataFrame, cfg: Config) -> pd.Timestamp:
    """One long horizon before the first forecast origin."""
    first = RollingOriginSplitter(cfg).origins(panel)[0]
    return first - max(cfg.protocol.horizons) * WEEK


class _RecordingTrial:
    """Stands in for an Optuna trial so an adapter declares its search space
    without a study being run. Every suggestion returns the low end, which is
    only ever used to build the dict we then throw away.
    """

    def __init__(self):
        self.spaces = {}

    def _numeric(self, name, low, high, log):
        self.spaces[name] = f"{low}-{high}" + (" log" if log else "")

    def suggest_int(self, name, low, high, step=1, log=False):
        self._numeric(name, low, high, log)
        return low

    def suggest_float(self, name, low, high, step=None, log=False):
        self._numeric(name, low, high, log)
        return low

    def suggest_categorical(self, name, choices):
        self.spaces[name] = "{" + ", ".join(str(c) for c in choices) + "}"
        return choices[0]


def declared_space(adapter_cls, cfg) -> dict:
    """The ranges an adapter searches, as display strings.

    Empty means there is nothing to search, and that covers two different
    cases: the zero-shot models, which declare no space at all, and the
    statsforecast Auto* models, which select their own order internally. Both
    must end up with a budget of zero - spending trials on an empty space costs
    real time and searches nothing.
    """
    if not getattr(adapter_cls, "tunable", True):
        return {}
    trial = _RecordingTrial()
    try:
        adapter_cls(cfg).tuning_space(trial)
    except (AttributeError, NotImplementedError):
        return {}
    return trial.spaces


def load_best(cfg: Config) -> tuple:
    """The parameters an earlier --tune recorded, as (best, budget).

    result_key folds the parameters in, so a run that forgets them reads a
    different checkpoint than the tuned run wrote. Every entry point loads this
    rather than trusting the config alone.
    """
    path = Path(cfg.run.output_dir) / cfg.run.name / "tuning_best.json"
    if not path.exists():
        return {}, {}
    saved = json.loads(path.read_text())
    return saved.get("best", {}), saved.get("budget", {})


def tune_all(panel: pd.DataFrame, cfg: Config, models=None, registry=None) -> TuningResult:
    registry = registry or registry_module.default()
    models = list(models or cfg.models.enabled)

    origin = tuning_origin(panel, cfg)
    train = panel[panel["ds"] <= origin]
    horizon = max(cfg.protocol.horizons)
    lo, hi = origin + WEEK, origin + horizon * WEEK
    actual = panel[panel["ds"].between(lo, hi)][["unique_id", "ds", "y"]]

    fit_frame, _ = impute_gaps(train)
    denominators = seasonal_denominators(train, cfg.data.season_length)

    best, best_value, budget, rows = {}, {}, {}, []
    for name in models:
        adapter_cls = registry.get(name)
        trials = cfg.tuning.budget_trials if declared_space(adapter_cls, cfg) else 0
        budget[name] = trials

        if trials == 0:
            best[name], best_value[name] = {}, np.nan
            continue

        study_best, study_value, study_rows = _search(
            name, adapter_cls, cfg, fit_frame, actual, denominators, horizon, trials)
        best[name], best_value[name] = study_best, study_value
        rows.extend(study_rows)

    frame = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["model", "trial", "value", "params", "status", "error",
                 "config_hash", "seed", "objective"])
    path = _write(frame, cfg, best, budget, origin)
    return TuningResult(best=best, best_value=best_value, budget=budget,
                        trials=frame, path=path,
                        meta={"origin": str(origin.date()), "objective": OBJECTIVE})


def _search(name, adapter_cls, cfg, train, actual, denominators, horizon, trials):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rows = []

    def objective(trial):
        set_seeds(cfg.run.seed + trial.number)
        probe = adapter_cls(cfg)
        params = probe.tuning_space(trial)
        record = {"model": name, "trial": trial.number,
                  "params": json.dumps(params, default=str),
                  "config_hash": cfg.hash, "seed": cfg.run.seed + trial.number,
                  "objective": OBJECTIVE}
        try:
            model = adapter_cls(cfg, params=params)
            model.fit(train)
            pred = model.predict(horizon)

            expected_ds = pd.date_range(train["ds"].max() + WEEK, periods=horizon, freq="7D")
            validate_prediction(pred, sorted(train["unique_id"].unique()), expected_ds)

            scored = evaluate(actual, pred, denominators, model=name, fold=-1,
                              horizon=horizon)
            value = float(scored.loc[scored["metric"] == "MASE", "value"].mean())
            if not np.isfinite(value):
                raise ValueError(f"objective was {value}")
        except Exception as exc:
            rows.append({**record, "value": np.nan, "status": "failed",
                         "error": f"{type(exc).__name__}: {exc}"})
            # A trial that dies must not end the search - report it and move on.
            raise optuna.TrialPruned() from exc

        rows.append({**record, "value": value, "status": "ok", "error": ""})
        return value

    sampler = optuna.samplers.TPESampler(seed=cfg.run.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=trials, catch=())

    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        return {}, np.nan, rows
    winner = min(ok, key=lambda r: r["value"])
    return json.loads(winner["params"]), winner["value"], rows


def _write(frame: pd.DataFrame, cfg: Config, best: dict, budget: dict, origin) -> Path:
    out_dir = Path(cfg.run.output_dir) / cfg.run.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # The tiers tune separately, so a plain write would drop whichever tier
    # went first. Keep every other model's record and replace only this one's.
    path = out_dir / "tuning_trials.parquet"
    if path.exists():
        prior = pd.read_parquet(path)
        parts = [p for p in (prior[~prior["model"].isin(list(budget))], frame)
                 if not p.empty]
        frame = pd.concat(parts, ignore_index=True) if parts else frame.iloc[:0]
    frame.to_parquet(path, index=False)

    best_path = out_dir / "tuning_best.json"
    merged_best, merged_budget = {}, {}
    if best_path.exists():
        prior = json.loads(best_path.read_text())
        merged_best.update(prior.get("best", {}))
        merged_budget.update(prior.get("budget", {}))
    merged_best.update(best)
    merged_budget.update(budget)

    best_path.write_text(json.dumps({
        "config_hash": cfg.hash,
        "seed": cfg.run.seed,
        "budget_trials": cfg.tuning.budget_trials,
        "objective": OBJECTIVE,
        "tuning_origin": str(origin.date()),
        "budget": merged_budget,
        "best": merged_best,
    }, indent=2, default=str))
    return path
