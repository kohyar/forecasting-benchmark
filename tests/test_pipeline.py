"""Tying loading, eligibility and sampling together for the CLIs."""
import pytest

from tsbench.config import Config
from tsbench.data.loader import normalize_panel
from tsbench.pipeline import eligible_profile, ensure_sample


@pytest.fixture
def cfg(config_dict, tmp_path):
    config_dict["sampling"]["n_series"] = 2
    config_dict["sampling"]["sample_path"] = str(tmp_path / "sample.csv")
    return Config.from_dict(config_dict)


@pytest.fixture
def panel(raw_frame, cfg):
    return normalize_panel(raw_frame, cfg)


def test_eligible_profile_keeps_only_protocol_ready_series(panel, cfg):
    prof, report = eligible_profile(panel, cfg)

    assert set(prof["unique_id"]) <= set(panel["unique_id"])
    assert report["eligible"] == len(prof)
    assert report["min_train_weeks_found_eligible"] >= cfg.protocol.min_train_weeks


def test_builds_and_freezes_a_sample_on_first_call(panel, cfg):
    sample, meta = ensure_sample(panel, cfg)

    assert len(sample) == 2
    assert meta["config_hash"] == cfg.hash
    assert meta["built"] is True


def test_reuses_the_frozen_sample_on_the_next_call(panel, cfg):
    first, _ = ensure_sample(panel, cfg)
    second, meta = ensure_sample(panel, cfg)

    assert second["unique_id"].tolist() == first["unique_id"].tolist()
    assert meta["built"] is False


def test_a_changed_config_forces_a_rebuild(panel, config_dict, tmp_path):
    config_dict["sampling"]["n_series"] = 2
    config_dict["sampling"]["sample_path"] = str(tmp_path / "sample.csv")
    cfg = Config.from_dict(config_dict)
    ensure_sample(panel, cfg)

    config_dict["run"]["seed"] = 99
    changed = Config.from_dict(config_dict)
    _, meta = ensure_sample(panel, changed)

    assert meta["built"] is True
    assert meta["config_hash"] == changed.hash


def test_moving_the_data_or_output_path_keeps_the_frozen_sample(panel, config_dict, tmp_path):
    """The sample depends on data content, sampling and protocol - not on where
    files live. Otherwise moving to a cluster silently rebuilds it."""
    config_dict["sampling"]["n_series"] = 2
    config_dict["sampling"]["sample_path"] = str(tmp_path / "sample.csv")
    ensure_sample(panel, Config.from_dict(config_dict))

    config_dict["data"]["path"] = "/Volumes/c/s/v/export.csv"
    config_dict["run"]["output_dir"] = "/Volumes/c/s/v/results"
    config_dict["run"]["name"] = "on-the-cluster"
    config_dict["models"]["enabled"] = ["naive", "autoets"]
    _, meta = ensure_sample(panel, Config.from_dict(config_dict))

    assert meta["built"] is False


def test_changing_the_protocol_rebuilds_the_sample(panel, config_dict, tmp_path):
    """Eligibility depends on the origins, so a protocol change can change
    which series are even allowed in."""
    config_dict["sampling"]["n_series"] = 2
    config_dict["sampling"]["sample_path"] = str(tmp_path / "sample.csv")
    ensure_sample(panel, Config.from_dict(config_dict))

    config_dict["protocol"]["folds"] = 1
    _, meta = ensure_sample(panel, Config.from_dict(config_dict))

    assert meta["built"] is True
