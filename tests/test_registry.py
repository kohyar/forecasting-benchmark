"""The registry: adding a model means writing an adapter and registering it,
never editing the runner."""
import pandas as pd
import pytest

from tsbench.config import Config
from tsbench.models import base, registry
from tsbench.models.base import AdapterError, ModelAdapter, validate_prediction


@pytest.fixture
def cfg(config_dict):
    return Config.from_dict(config_dict)


@pytest.fixture
def scratch():
    """Register into an isolated registry so tests do not leak into each other."""
    return registry.Registry()


class Dummy(ModelAdapter):
    name = "dummy"
    family = "baseline"

    def fit(self, train_df):
        self._ids = sorted(train_df["unique_id"].unique())
        self._last = train_df["ds"].max()

    def predict(self, horizon):
        ds = pd.date_range(self._last + pd.Timedelta(days=7), periods=horizon, freq="7D")
        return pd.DataFrame({
            "unique_id": [u for u in self._ids for _ in range(horizon)],
            "ds": list(ds) * len(self._ids),
            "yhat": 1.0,
        })


def test_registering_then_retrieving_returns_the_adapter(scratch):
    scratch.register(Dummy)

    assert scratch.get("dummy") is Dummy


def test_duplicate_registration_is_refused(scratch):
    scratch.register(Dummy)

    with pytest.raises(ValueError, match="already registered"):
        scratch.register(Dummy)


def test_unknown_model_lists_what_is_available(scratch):
    scratch.register(Dummy)

    with pytest.raises(KeyError, match="dummy"):
        scratch.get("nope")


def test_family_must_be_one_of_the_four(scratch):
    class Weird(Dummy):
        name = "weird"
        family = "hybrid"

    with pytest.raises(ValueError, match="family"):
        scratch.register(Weird)


def test_disabled_adapter_is_listed_but_not_available(scratch):
    class Gated(Dummy):
        name = "gated"

    scratch.register(Gated, enabled=False, disabled_reason="licence pending")

    assert "gated" in scratch.names()
    assert "gated" not in scratch.available_names()
    assert scratch.status("gated")["disabled_reason"] == "licence pending"


def test_create_passes_config_and_params(scratch, cfg):
    scratch.register(Dummy)

    model = scratch.create("dummy", cfg, params={"alpha": 3})

    assert isinstance(model, Dummy)
    assert model.params["alpha"] == 3
    assert model.season_length == cfg.data.season_length


def test_adapters_declare_their_capabilities(scratch, cfg):
    scratch.register(Dummy)
    model = scratch.create("dummy", cfg)

    assert model.supports_quantiles() is False
    assert model.supports_covariates() is False


def test_default_registry_holds_the_roster():
    """Importing the package registers every adapter without the runner
    knowing any of their names."""
    names = registry.default().names()

    assert {"naive", "seasonal_naive"} <= set(names)


def test_tirex_is_registered_but_gated_on_its_licence():
    status = registry.default().status("tirex")

    assert status["enabled"] is False
    assert "licence" in status["disabled_reason"].lower()


def test_unavailable_library_is_reported_not_raised(scratch, cfg):
    class NeedsMissingLib(Dummy):
        name = "needs_lib"

        @classmethod
        def is_available(cls):
            return False, "nosuchlib is not installed"

    scratch.register(NeedsMissingLib)

    assert "needs_lib" not in scratch.available_names()
    assert "nosuchlib" in scratch.status("needs_lib")["disabled_reason"]


# --- the output contract every adapter must satisfy ------------------------

@pytest.fixture
def expected():
    ds = pd.date_range("2022-03-06", periods=3, freq="7D")
    return {"ids": ["A", "B"], "ds": ds}


def good_frame(expected, **extra):
    rows = pd.DataFrame({
        "unique_id": [u for u in expected["ids"] for _ in expected["ds"]],
        "ds": list(expected["ds"]) * len(expected["ids"]),
        "yhat": 1.0,
    })
    for k, v in extra.items():
        rows[k] = v
    return rows


def test_a_well_formed_prediction_passes(expected):
    validate_prediction(good_frame(expected), expected["ids"], expected["ds"])


def test_missing_yhat_is_an_error(expected):
    bad = good_frame(expected).drop(columns=["yhat"])

    with pytest.raises(AdapterError, match="yhat"):
        validate_prediction(bad, expected["ids"], expected["ds"])


def test_a_dropped_series_is_an_error(expected):
    bad = good_frame(expected).query("unique_id == 'A'")

    with pytest.raises(AdapterError, match="series"):
        validate_prediction(bad, expected["ids"], expected["ds"])


def test_wrong_timestamps_are_an_error(expected):
    bad = good_frame(expected)
    bad.loc[0, "ds"] = pd.Timestamp("2030-01-06")

    with pytest.raises(AdapterError, match="timestamp"):
        validate_prediction(bad, expected["ids"], expected["ds"])


def test_duplicate_rows_are_an_error(expected):
    bad = pd.concat([good_frame(expected), good_frame(expected).head(1)])

    with pytest.raises(AdapterError, match="duplicate"):
        validate_prediction(bad, expected["ids"], expected["ds"])


def test_crossed_quantiles_are_an_error(expected):
    bad = good_frame(expected, yhat_q10=5.0, yhat_q90=1.0)

    with pytest.raises(AdapterError, match="quantile"):
        validate_prediction(bad, expected["ids"], expected["ds"])


def test_ordered_quantiles_pass(expected):
    ok = good_frame(expected, yhat_q10=0.5, yhat_q90=1.5)

    validate_prediction(ok, expected["ids"], expected["ds"])


def test_all_nan_forecasts_are_an_error(expected):
    bad = good_frame(expected)
    bad["yhat"] = float("nan")

    with pytest.raises(AdapterError, match="NaN"):
        validate_prediction(bad, expected["ids"], expected["ds"])


def test_unavailable_models_are_partitioned_out_not_fatal(scratch, cfg):
    """A missing library must not stop the rest of a long run."""
    from tsbench.models.registry import partition_available

    class Missing(Dummy):
        name = "missing"

        @classmethod
        def is_available(cls):
            return False, "nosuchlib is not installed"

    scratch.register(Dummy)
    scratch.register(Missing)

    runnable, skipped = partition_available(scratch, ["dummy", "missing"])

    assert runnable == ["dummy"]
    assert skipped == [("missing", "nosuchlib is not installed")]


def test_unknown_names_are_reported_separately(scratch):
    from tsbench.models.registry import partition_available

    scratch.register(Dummy)

    with pytest.raises(KeyError, match="nope"):
        partition_available(scratch, ["dummy", "nope"])
