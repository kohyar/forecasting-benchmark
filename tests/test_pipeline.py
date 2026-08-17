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
