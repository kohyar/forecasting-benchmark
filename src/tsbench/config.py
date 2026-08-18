"""Run configuration. A run must be reproducible from config + seed alone."""
import dataclasses
import hashlib
import json
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int
    n_repeats: int
    output_dir: str
    # Recorded on every timing row: parallelism changes wall clock, so it must
    # be identical across models and visible in the results.
    n_jobs: int = 1
    device: str = "auto"
    # Each (model, fold) in its own process: some adapter libraries cannot
    # share one (lightgbm and torch both bundle OpenMP), and a segfault or an
    # OOM kill can only be survived by watching a child exit.
    execution: str = "subprocess"
    # Scratch for worker hand-off. Defaults to the system temp dir; keep it on
    # local disk even when output_dir is a network volume.
    work_dir: str = None


@dataclass(frozen=True)
class ScopeConfig:
    geography_level: list
    channels: list


@dataclass(frozen=True)
class DataConfig:
    path: str
    target: str
    season_length: int
    scope: ScopeConfig
    # Weeks the export omits entirely: evidence says these are weeks with no
    # sales, not unmeasured weeks. "zero" treats them as observed zeros;
    # "missing" keeps them NaN for a sensitivity check.
    absent_rows: str = "zero"


@dataclass(frozen=True)
class SamplingConfig:
    n_series: int
    volume_deciles: int
    zero_share_bins: list
    min_observed_weeks: int
    sample_path: str


@dataclass(frozen=True)
class ProtocolConfig:
    folds: int
    step: int
    horizons: list
    min_train_weeks: int


@dataclass(frozen=True)
class CostConfig:
    """Set once the runner hardware is fixed; turns wall clock into the
    $/1k series figure the paper reports alongside accuracy."""
    usd_per_hour: float = None
    instance_type: str = None
    runtime_version: str = None


@dataclass(frozen=True)
class TuningConfig:
    budget_trials: int


@dataclass(frozen=True)
class MetricsConfig:
    quantile_levels: list
    coverage_levels: list
    mase_denominator_window: str


@dataclass(frozen=True)
class ModelsConfig:
    enabled: list
    params: dict = field(default_factory=dict)
    # Run covariate-capable models twice, with and without. Doubles their cost
    # and is a core result: the model is fixed, covariates are the only change.
    covariate_ablation: bool = False


@dataclass(frozen=True)
class Config:
    run: RunConfig
    data: DataConfig
    sampling: SamplingConfig
    protocol: ProtocolConfig
    tuning: TuningConfig
    metrics: MetricsConfig
    models: ModelsConfig
    cost: CostConfig = field(default_factory=CostConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        return _build(cls, data, path="")

    @classmethod
    def from_yaml(cls, path) -> "Config":
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def hash(self) -> str:
        """Stable identity of the whole run configuration, recorded on every
        output row for provenance."""
        return _digest(self.to_dict())

    def result_key(self, model: str, params: dict | None = None) -> str:
        """What determines one model's results.

        Excludes file locations, the run name, cost metadata, the enabled list
        and every *other* model's parameters - so fixing one model's batch size,
        or moving the run to a cluster, does not invalidate everyone else's
        completed work. Includes device and n_jobs, because they change the
        timings that sit next to the accuracy numbers.
        """
        d = self.to_dict()
        d["run"] = {k: v for k, v in d["run"].items()
                    if k not in ("name", "output_dir", "work_dir")}
        d["data"] = {k: v for k, v in d["data"].items() if k != "path"}
        d["sampling"] = {k: v for k, v in d["sampling"].items() if k != "sample_path"}
        d.pop("cost", None)
        d["models"] = {"model": model, "params": params or {}}
        return _digest(d)

    @property
    def sample_key(self) -> str:
        """What determines the frozen sample: the data's identity, the sampling
        scheme, and the protocol (eligibility depends on the origins)."""
        d = self.to_dict()
        return _digest({
            "seed": d["run"]["seed"],
            "data": {k: v for k, v in d["data"].items() if k != "path"},
            "sampling": {k: v for k, v in d["sampling"].items() if k != "sample_path"},
            "protocol": d["protocol"],
            "mase_denominator_window": d["metrics"]["mase_denominator_window"],
        })

    def dump_yaml(self, path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=True))


def _digest(payload) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _build(cls, data: Any, path: str):
    if not isinstance(data, dict):
        raise ConfigError(f"expected a mapping at {path or 'root'}, got {type(data).__name__}")

    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"unknown config key(s) at {path or 'root'}: {', '.join(unknown)}")

    kwargs = {}
    for f in fields(cls):
        here = f"{path}{f.name}"
        if f.name not in data:
            if f.default is MISSING and f.default_factory is MISSING:
                raise ConfigError(f"missing required config key: {here}")
            continue
        value = data[f.name]
        kwargs[f.name] = _build(f.type, value, path=f"{here}.") if is_dataclass(f.type) else value
    return cls(**kwargs)
