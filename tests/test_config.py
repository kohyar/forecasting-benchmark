"""The config is the reproducibility contract: everything in it, nothing hardcoded."""
import pytest

from tsbench.config import Config, ConfigError


def test_loads_nested_values(config_dict):
    cfg = Config.from_dict(config_dict)

    assert cfg.data.target == "Dollars"
    assert cfg.data.scope.channels == ["CONVENTIONAL|MULTI OUTLET"]
    assert cfg.protocol.horizons == [1, 3]
    assert cfg.tuning.budget_trials == 20


def test_unknown_key_is_rejected(config_dict):
    config_dict["data"]["targt"] = "Units"

    with pytest.raises(ConfigError, match="targt"):
        Config.from_dict(config_dict)


def test_hash_is_stable_across_key_order(config_dict):
    reordered = {k: config_dict[k] for k in reversed(list(config_dict))}

    assert Config.from_dict(config_dict).hash == Config.from_dict(reordered).hash


def test_hash_changes_when_any_value_changes(config_dict):
    before = Config.from_dict(config_dict).hash
    config_dict["tuning"]["budget_trials"] = 21

    assert Config.from_dict(config_dict).hash != before


def test_seed_and_budget_are_required(config_dict):
    del config_dict["tuning"]["budget_trials"]

    with pytest.raises(ConfigError, match="budget_trials"):
        Config.from_dict(config_dict)


def test_roundtrips_through_yaml(tmp_path, config_dict):
    import yaml

    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(config_dict))

    assert Config.from_yaml(p).hash == Config.from_dict(config_dict).hash


def test_shipped_default_config_holds_the_non_negotiables():
    """season_length=52, the protocol defaults, and an equal tuning budget."""
    cfg = Config.from_yaml("configs/default.yaml")

    assert cfg.data.season_length == 52
    assert cfg.protocol.folds == 5
    assert cfg.protocol.step == 4
    assert cfg.protocol.horizons == [4, 13]
    assert cfg.protocol.min_train_weeks == 104
    assert cfg.tuning.budget_trials == 20
    assert cfg.run.n_repeats == 3
    assert cfg.sampling.n_series == 1000
