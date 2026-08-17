"""Stratified sampling, frozen to disk.

Random sampling leaves stratum coverage to chance; stratification makes it
exact and reproducible. The sample file is the anchor for every later run.
"""
import numpy as np
import pandas as pd
import pytest

from tsbench.config import Config
from tsbench.data.sampling import (
    label_strata,
    load_sample,
    save_sample,
    series_profile,
    stratified_sample,
)
from tsbench.data.loader import normalize_panel


@pytest.fixture
def cfg(config_dict):
    return Config.from_dict(config_dict)


@pytest.fixture
def population():
    """1,200 synthetic series with a long-tailed volume distribution and a
    minority of intermittent ones - the shape of real retail data."""
    rng = np.random.default_rng(7)
    n = 1200
    volume = np.exp(rng.normal(8, 2.5, n))
    zero_share = np.where(rng.random(n) < 0.15, rng.uniform(0.05, 0.6, n), 0.0)
    return pd.DataFrame({
        "unique_id": [f"S{i:04d}" for i in range(n)],
        "n_obs": rng.integers(120, 233, n),
        "total_volume": volume,
        "zero_share": zero_share,
        "first_ds": pd.Timestamp("2022-01-09"),
        "last_ds": pd.Timestamp("2026-06-14"),
    })


def test_profile_reports_volume_zero_share_and_span(raw_frame, cfg):
    panel = normalize_panel(raw_frame, cfg)
    prof = series_profile(panel)

    assert set(prof.columns) >= {"unique_id", "n_obs", "n_missing", "total_volume",
                                 "zero_share", "first_ds", "last_ds"}
    b = prof.set_index("unique_id").loc["SUB_B@@ALBANY, NY - MULO"]
    assert b["n_obs"] == 60, "omitted weeks are zero-sales weeks, so they count"
    assert b["n_missing"] == 0
    assert b["zero_share"] == pytest.approx(5 / 60), "three reported plus two omitted"


def test_returns_exactly_the_requested_size(population, config_dict):
    config_dict["sampling"]["n_series"] = 100
    config_dict["sampling"]["volume_deciles"] = 10
    cfg = Config.from_dict(config_dict)

    assert len(stratified_sample(population, cfg)) == 100


def test_same_seed_gives_the_identical_sample(population, config_dict):
    config_dict["sampling"]["n_series"] = 100
    cfg = Config.from_dict(config_dict)

    first = stratified_sample(population, cfg)
    second = stratified_sample(population, cfg)

    pd.testing.assert_frame_equal(first, second)


def test_different_seed_gives_a_different_sample(population, config_dict):
    config_dict["sampling"]["n_series"] = 100
    a = stratified_sample(population, Config.from_dict(config_dict))
    config_dict["run"]["seed"] = 43
    b = stratified_sample(population, Config.from_dict(config_dict))

    assert set(a["unique_id"]) != set(b["unique_id"])


def test_every_stratum_is_represented_within_rounding(population, config_dict):
    """The point of stratifying: coverage is exact, not left to sampling noise."""
    config_dict["sampling"]["n_series"] = 200
    config_dict["sampling"]["volume_deciles"] = 10
    cfg = Config.from_dict(config_dict)

    sample = stratified_sample(population, cfg)
    labelled = label_strata(population, cfg)

    want = labelled["stratum"].value_counts(normalize=True) * 200
    got = sample["stratum"].value_counts().reindex(want.index).fillna(0)

    assert (got - want).abs().max() <= 1.0


def test_volume_deciles_are_evenly_covered(population, config_dict):
    config_dict["sampling"]["n_series"] = 200
    config_dict["sampling"]["volume_deciles"] = 10
    sample = stratified_sample(population, Config.from_dict(config_dict))

    per_decile = sample["volume_decile"].value_counts()

    assert len(per_decile) == 10
    assert per_decile.min() >= 15, "no decile is crowded out by the long tail"


def test_intermittent_series_are_represented_deterministically(population, config_dict):
    config_dict["sampling"]["n_series"] = 200
    sample = stratified_sample(population, Config.from_dict(config_dict))

    share_in_pop = (population["zero_share"] > 0).mean()
    share_in_sample = (sample["zero_share"] > 0).mean()

    assert share_in_sample == pytest.approx(share_in_pop, abs=0.03)


def test_ineligible_series_are_excluded_before_stratifying(population, config_dict):
    population.loc[:99, "n_obs"] = 5
    config_dict["sampling"]["n_series"] = 100
    config_dict["sampling"]["min_observed_weeks"] = 120
    sample = stratified_sample(population, Config.from_dict(config_dict))

    assert not set(sample["unique_id"]) & set(population.loc[:99, "unique_id"])


def test_degenerate_zero_share_bin_does_not_break_sampling(population, config_dict):
    """Real SPINS produce data has almost no zeros - the bin collapses."""
    population["zero_share"] = 0.0
    config_dict["sampling"]["n_series"] = 50
    sample = stratified_sample(population, Config.from_dict(config_dict))

    assert len(sample) == 50


def test_asking_for_more_than_available_raises(population, config_dict):
    config_dict["sampling"]["n_series"] = 5000
    with pytest.raises(ValueError, match="eligible"):
        stratified_sample(population, Config.from_dict(config_dict))


def test_sample_file_roundtrips_with_its_provenance(tmp_path, population, config_dict):
    config_dict["sampling"]["n_series"] = 50
    cfg = Config.from_dict(config_dict)
    sample = stratified_sample(population, cfg)
    path = tmp_path / "sample.csv"

    save_sample(sample, path, cfg)
    loaded, meta = load_sample(path)

    pd.testing.assert_frame_equal(loaded, sample)
    assert meta["seed"] == 42
    assert meta["config_hash"] == cfg.hash


def test_loading_a_sample_built_from_another_config_is_flagged(tmp_path, population, config_dict):
    cfg = Config.from_dict(config_dict)
    sample = stratified_sample(population, cfg)
    path = tmp_path / "sample.csv"
    save_sample(sample, path, cfg)

    config_dict["sampling"]["n_series"] = 7
    other = Config.from_dict(config_dict)

    with pytest.raises(ValueError, match="config_hash"):
        load_sample(path, expect=other)
