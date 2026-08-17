"""Every registered adapter, end to end through the real execution path.

Each adapter runs in its own worker process, exactly as it does in a real run.
That is not only realistic, it is required: fitting lightgbm and a torch model
in one process segfaults, so a conformance suite that fitted them all inline
would crash rather than report.
"""
import pandas as pd
import pytest

from tsbench.config import Config
from tsbench.data.loader import normalize_panel
from tsbench.models import registry as registry_module
from tsbench.runner import BenchmarkRunner

pytestmark = pytest.mark.slow

AVAILABLE = registry_module.default().available_names()

# keep neural fits to a few steps; this checks the contract, not accuracy.
# Only the neural adapters accept these - the rest reject unknown parameters,
# which is the behaviour that catches a typo in a config.
FAST_PARAMS = {"max_steps": 3, "input_size": 8}
NEURAL = {"nhits", "patchtst", "dlinear", "tft"}


@pytest.fixture
def cfg(tmp_path):
    return Config.from_dict({
        "run": {"name": "conformance", "seed": 42, "n_repeats": 1,
                "output_dir": str(tmp_path), "n_jobs": 1, "device": "cpu",
                "execution": "subprocess"},
        "data": {"path": "unused.csv", "target": "Dollars", "season_length": 4,
                 "absent_rows": "zero",
                 "scope": {"geography_level": ["MARKET"],
                           "channels": ["CONVENTIONAL|MULTI OUTLET"]}},
        "sampling": {"n_series": 3, "volume_deciles": 2, "zero_share_bins": [0.0, 1.01],
                     "min_observed_weeks": 8, "sample_path": "data/s.csv"},
        "protocol": {"folds": 1, "step": 2, "horizons": [3], "min_train_weeks": 8},
        "tuning": {"budget_trials": 20},
        "metrics": {"quantile_levels": [0.1, 0.5, 0.9], "coverage_levels": [0.8],
                    "mase_denominator_window": "earliest_origin"},
        "models": {"enabled": []},
    })


@pytest.fixture
def panel(raw_frame, cfg):
    return normalize_panel(raw_frame, cfg)


@pytest.fixture
def run_adapter(cfg, panel):
    def _run(name):
        params = {name: dict(FAST_PARAMS)} if name in NEURAL else {}
        return BenchmarkRunner(cfg).run(panel, models=[name], tuned_params=params)
    return _run


@pytest.mark.parametrize("name", AVAILABLE)
def test_adapter_completes_a_fold_and_is_scored(name, run_adapter):
    result = run_adapter(name)
    failed = result.timings[result.timings["status"] == "failed"]

    assert failed.empty, failed.iloc[0]["error"] if len(failed) else ""
    assert len(result.metrics) > 0
    assert set(result.metrics["metric"]) >= {"MASE", "RMSSE", "MAE", "RMSE"}


@pytest.mark.parametrize("name", AVAILABLE)
def test_timings_are_recorded_for_every_adapter(name, run_adapter):
    t = run_adapter(name).timings

    assert (t["predict_seconds"] > 0).all()
    assert t["fit_seconds"].notna().all()
    assert t["device"].eq("cpu").all()


@pytest.mark.parametrize("name", AVAILABLE)
def test_quantile_claims_match_the_scored_metrics(name, run_adapter, cfg):
    adapter = registry_module.default().get(name)
    result = run_adapter(name)
    scored = set(result.metrics["metric"])

    assert ("CRPS" in scored) == adapter(cfg).supports_quantiles()


def test_every_adapter_declares_a_known_family():
    reg = registry_module.default()

    for name in reg.names():
        assert reg.get(name).family in ("baseline", "local", "global", "foundation")


def test_zero_shot_models_report_a_zero_tuning_budget():
    reg = registry_module.default()

    for name in reg.names():
        adapter = reg.get(name)
        if adapter.family == "foundation":
            assert adapter.tunable is False, f"{name} is zero-shot, budget must be 0"
