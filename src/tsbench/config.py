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


@dataclass(frozen=True)
class Config:
    run: RunConfig
    data: DataConfig
    sampling: SamplingConfig
    protocol: ProtocolConfig
    tuning: TuningConfig
    metrics: MetricsConfig
    models: ModelsConfig

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
        """Stable identity of the run configuration, recorded on every output row."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def dump_yaml(self, path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=True))


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
