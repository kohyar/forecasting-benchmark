"""Integration checks against the actual SPINS export.

Marked slow: reads the 945MB source. Run with `pytest -m slow`.
"""
import pandas as pd
import pytest

from tsbench.config import Config
from tsbench.data.loader import load_panel
from tsbench.data.sampling import series_profile, stratified_sample
from tsbench.eval.metrics import denominators_for_protocol
from tsbench.eval.splitter import RollingOriginSplitter

pytestmark = pytest.mark.slow

WEEK = pd.Timedelta(days=7)


@pytest.fixture(scope="module")
def cfg():
    return Config.from_yaml("configs/default.yaml")


@pytest.fixture(scope="module")
def panel(cfg):
    return load_panel(cfg)


def test_scope_selects_market_mulo_only(panel):
    assert panel["geography_level"].unique().tolist() == ["MARKET"]
    assert panel["channel"].unique().tolist() == ["CONVENTIONAL|MULTI OUTLET"]
    assert panel["unique_id"].nunique() == 7473


def test_panel_is_on_a_complete_weekly_grid(panel):
    gaps = panel.groupby("unique_id")["ds"].diff().dropna().unique()

    assert list(gaps) == [WEEK]


def test_target_is_dollars(cfg, panel):
    assert cfg.data.target == "Dollars"
    assert panel["y"].notna().sum() > 1_000_000


def test_protocol_origins_are_weekly_and_correctly_spaced(cfg, panel):
    origins = RollingOriginSplitter(cfg).origins(panel)

    assert len(origins) == 5
    assert all(b - a == 4 * WEEK for a, b in zip(origins, origins[1:]))
    assert origins[-1] == panel["ds"].max() - 13 * WEEK


def test_eligible_series_clear_two_seasonal_cycles(cfg, panel):
    from tsbench.pipeline import eligible_profile

    profile, summary = eligible_profile(panel, cfg)
    report = RollingOriginSplitter(cfg).eligibility(panel)
    report = report[report["unique_id"].isin(profile["unique_id"])]
    eligible = report

    assert len(eligible) > 1000, "enough eligible series to draw the sample from"
    assert eligible["train_weeks_at_earliest_origin"].min() >= 104


def test_sample_is_exactly_one_thousand_and_reproducible(cfg, panel):
    report = RollingOriginSplitter(cfg).eligibility(panel)
    profile = series_profile(panel).merge(
        report[["unique_id", "eligible"]], on="unique_id")
    profile = profile[profile["eligible"]].drop(columns="eligible")

    first = stratified_sample(profile, cfg)
    second = stratified_sample(profile, cfg)

    assert len(first) == 1000
    assert first["unique_id"].tolist() == second["unique_id"].tolist()
    assert first["volume_decile"].nunique() == 10


def test_frozen_sample_passes_the_protocol_assertion(cfg, panel):
    """The assert-before-you-run gate, on the sample the paper will use."""
    from tsbench.data.sampling import load_sample

    sample, meta = load_sample(cfg.sampling.sample_path, expect=cfg)
    sub = panel[panel["unique_id"].isin(sample["unique_id"])]

    summary = RollingOriginSplitter(cfg).assert_protocol_feasible(sub)

    assert summary["n_series"] == 1000
    assert summary["min_train_weeks_found"] >= 104
    assert meta["seed"] == cfg.run.seed


def test_no_fold_leaks_on_the_real_panel(cfg, panel):
    from tsbench.data.sampling import load_sample

    sample, _ = load_sample(cfg.sampling.sample_path, expect=cfg)
    sub = panel[panel["unique_id"].isin(sample["unique_id"])]

    for fold in RollingOriginSplitter(cfg).split(sub):
        assert fold.train["ds"].max() <= fold.origin
        for h in cfg.protocol.horizons:
            test = fold.test(h)
            assert test["ds"].min() == fold.origin + WEEK
            assert test["ds"].max() == fold.origin + h * WEEK
            assert not set(test["ds"]) & set(fold.train["ds"])
            assert "y" not in fold.future_covariates(h).columns


def test_denominators_are_finite_for_the_sampled_series(cfg, panel):
    from tsbench.pipeline import eligible_profile

    splitter = RollingOriginSplitter(cfg)
    den = denominators_for_protocol(panel, cfg, splitter).set_index("unique_id")
    profile, _ = eligible_profile(panel, cfg)

    sub = den.loc[profile["unique_id"]]

    assert sub["n_pairs"].min() > 0
    assert not sub["zero_denominator"].any(), "flat series are filtered out upstream"
    assert sub["mase_denom"].notna().all()
