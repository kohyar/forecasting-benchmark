"""Resume semantics for long runs on rented hardware.

Completed (model, fold) work must survive: a crash, a config edit that does
not affect that model, a code fix to a different model, and a path change
between machines. Failed work must be retried once the code is fixed, and
partial results must be on disk before the run ends.
"""
import numpy as np
import pandas as pd
import pytest

from tsbench.config import Config
from tsbench.data.loader import normalize_panel
from tsbench.models import registry
from tsbench.models.base import ModelAdapter
from tsbench.runner import BenchmarkRunner, collect, warn_if_stale


class Constant(ModelAdapter):
    name = "constant"
    family = "baseline"
    fits = []

    def fit(self, train_df):
        Constant.fits.append(self.name)
        self._ids = sorted(train_df["unique_id"].unique())
        self._last = train_df["ds"].max()

    def predict(self, horizon):
        ds = pd.date_range(self._last + pd.Timedelta(days=7), periods=horizon, freq="7D")
        return pd.DataFrame({
            "unique_id": np.repeat(self._ids, horizon),
            "ds": np.tile(ds.to_numpy(), len(self._ids)),
            "yhat": float(self.params.get("level", 5.0)),
        })


class Other(Constant):
    name = "other"


class Flaky(Constant):
    """Fails until someone flips the switch - a stand-in for a code fix."""

    name = "flaky"
    fixed = False

    def fit(self, train_df):
        if not Flaky.fixed:
            raise RuntimeError("not implemented yet")
        super().fit(train_df)


class Interrupting(Constant):
    """Simulates the operator killing the run mid-way."""

    name = "interrupting"

    def fit(self, train_df):
        raise KeyboardInterrupt


@pytest.fixture(autouse=True)
def reset():
    Constant.fits = []
    Flaky.fixed = False


@pytest.fixture
def base(config_dict, tmp_path):
    config_dict["run"]["output_dir"] = str(tmp_path)
    config_dict["run"]["n_repeats"] = 1
    config_dict["run"]["execution"] = "inprocess"
    return config_dict


@pytest.fixture
def cfg(base):
    return Config.from_dict(base)


@pytest.fixture
def panel(raw_frame, cfg):
    return normalize_panel(raw_frame, cfg)


@pytest.fixture
def reg():
    r = registry.Registry()
    for adapter in (Constant, Other, Flaky, Interrupting):
        r.register(adapter)
    return r


def _fits(model):
    return sum(1 for f in Constant.fits if f == model)


# --- what invalidates a checkpoint, and what must not ------------------------

def test_changing_another_models_params_keeps_this_models_checkpoint(base, panel, reg):
    BenchmarkRunner(Config.from_dict(base), registry=reg).run(panel, models=["constant", "other"])
    Constant.fits = []

    base["models"]["params"] = {"other": {"level": 9.0}}
    result = BenchmarkRunner(Config.from_dict(base), registry=reg).run(
        panel, models=["constant", "other"], tuned_params={"other": {"level": 9.0}})

    assert _fits("constant") == 0, "constant's checkpoint survived"
    assert _fits("other") > 0, "other was rerun with its new params"
    assert result.timings[result.timings["model"] == "constant"]["resumed"].all()


def test_changing_a_models_own_params_reruns_only_that_model(base, panel, reg):
    BenchmarkRunner(Config.from_dict(base), registry=reg).run(
        panel, models=["constant"], tuned_params={"constant": {"level": 1.0}})
    Constant.fits = []

    BenchmarkRunner(Config.from_dict(base), registry=reg).run(
        panel, models=["constant"], tuned_params={"constant": {"level": 2.0}})

    assert _fits("constant") > 0


def test_changing_the_enabled_list_keeps_checkpoints(base, panel, reg):
    base["models"]["enabled"] = ["constant"]
    BenchmarkRunner(Config.from_dict(base), registry=reg).run(panel)
    Constant.fits = []

    base["models"]["enabled"] = ["constant", "other"]
    BenchmarkRunner(Config.from_dict(base), registry=reg).run(panel)

    assert _fits("constant") == 0
    assert _fits("other") > 0


def test_paths_and_names_do_not_change_the_result_key(base, reg):
    a = BenchmarkRunner(Config.from_dict(base), registry=reg).result_key("constant", {})

    base["data"]["path"] = "/Volumes/catalog/schema/vol/export.csv"
    base["run"]["name"] = "renamed"
    base["sampling"]["sample_path"] = "elsewhere/sample.csv"
    base["run"]["work_dir"] = "/local_disk0/tmp"
    base["cost"] = {"usd_per_hour": 9.9, "instance_type": "x", "runtime_version": "y"}
    b = BenchmarkRunner(Config.from_dict(base), registry=reg).result_key("constant", {})

    assert a == b


def test_the_laptop_and_cluster_configs_share_result_and_sample_keys():
    """Checkpoints and the frozen sample must be interchangeable between
    machines running the same protocol."""
    import yaml

    laptop = Config.from_yaml("configs/default.yaml")
    cluster = Config.from_dict(yaml.safe_load(open("configs/databricks-t4.yaml")))

    assert laptop.sample_key == cluster.sample_key
    assert laptop.result_key("naive", {}) == cluster.result_key("naive", {})
    assert laptop.hash != cluster.hash, "full provenance hash still tells them apart"


def test_protocol_and_seed_do_change_the_result_key(base, reg):
    a = BenchmarkRunner(Config.from_dict(base), registry=reg).result_key("constant", {})

    base["protocol"]["horizons"] = [1, 2]
    b = BenchmarkRunner(Config.from_dict(base), registry=reg).result_key("constant", {})
    base["protocol"]["horizons"] = [1, 3]
    base["run"]["seed"] = 7
    c = BenchmarkRunner(Config.from_dict(base), registry=reg).result_key("constant", {})

    assert len({a, b, c}) == 3


def test_every_row_carries_its_result_key_and_commit(cfg, panel, reg):
    result = BenchmarkRunner(cfg, registry=reg).run(panel, models=["constant"])

    for df in (result.metrics, result.timings):
        assert df["result_key"].nunique() == 1
        assert "git_commit" in df.columns


# --- failures are retried once fixed ----------------------------------------

def test_a_failed_checkpoint_is_retried_by_default(cfg, panel, reg):
    first = BenchmarkRunner(cfg, registry=reg).run(panel, models=["flaky"])
    assert (first.timings["status"] == "failed").all()

    Flaky.fixed = True
    second = BenchmarkRunner(cfg, registry=reg).run(panel, models=["flaky"])

    assert (second.timings["status"] == "ok").all()
    assert not second.timings["resumed"].any()
    assert "flaky" in set(second.metrics["model"])


def test_failed_checkpoints_can_be_kept_when_asked(cfg, panel, reg):
    BenchmarkRunner(cfg, registry=reg).run(panel, models=["flaky"])
    Flaky.fixed = True

    kept = BenchmarkRunner(cfg, registry=reg, retry_failed=False).run(panel, models=["flaky"])

    assert (kept.timings["status"] == "failed").all()
    assert kept.timings["resumed"].all()


def test_completed_work_is_not_redone_alongside_a_retry(cfg, panel, reg):
    BenchmarkRunner(cfg, registry=reg).run(panel, models=["constant", "flaky"])
    Constant.fits = []
    Flaky.fixed = True

    BenchmarkRunner(cfg, registry=reg).run(panel, models=["constant", "flaky"])

    assert _fits("constant") == 0
    assert _fits("flaky") > 0


# --- robustness of the files themselves -------------------------------------

def test_a_corrupt_checkpoint_is_rerun_not_fatal(cfg, panel, reg):
    runner = BenchmarkRunner(cfg, registry=reg)
    runner.run(panel, models=["constant"])
    for path in runner.checkpoint_files("constant"):
        path.write_bytes(b"not a parquet file")
    Constant.fits = []

    result = BenchmarkRunner(cfg, registry=reg).run(panel, models=["constant"])

    assert _fits("constant") > 0
    assert (result.timings["status"] == "ok").all()


def test_checkpoints_are_written_atomically(cfg, panel, reg):
    """No half-written parquet is ever left under the checkpoint name."""
    runner = BenchmarkRunner(cfg, registry=reg)
    runner.run(panel, models=["constant"])

    leftovers = [p for p in runner.checkpoint_root.rglob("*") if p.suffix == ".tmp"]
    assert not leftovers


def test_scratch_files_do_not_live_under_the_results_dir(cfg, reg):
    runner = BenchmarkRunner(cfg, registry=reg)

    assert runner.out_dir not in runner.work_root.parents
    assert runner.work_root != runner.out_dir


# --- results are on disk before the run ends -------------------------------

def test_partial_results_are_written_after_each_model(cfg, panel, reg):
    with pytest.raises(KeyboardInterrupt):
        BenchmarkRunner(cfg, registry=reg).run(panel, models=["constant", "interrupting"])

    metrics = pd.read_parquet(cfg_out(cfg) / "metrics.parquet")
    timings = pd.read_parquet(cfg_out(cfg) / "timings.parquet")

    assert set(metrics["model"]) == {"constant"}
    assert set(timings["model"]) == {"constant"}


def test_collect_assembles_results_without_running_anything(cfg, panel, reg):
    BenchmarkRunner(cfg, registry=reg).run(panel, models=["constant"])
    Constant.fits = []

    result = collect(cfg, models=["constant", "other"], registry=reg)

    assert Constant.fits == [], "collect never fits a model"
    assert set(result.metrics["model"]) == {"constant"}, "other has no checkpoint yet"
    assert result.timings["resumed"].all()
    assert (cfg_out(cfg) / "run_metadata.json").exists()


def test_collect_reports_what_is_missing(cfg, panel, reg):
    BenchmarkRunner(cfg, registry=reg).run(panel, models=["constant"])

    result = collect(cfg, models=["constant", "other"], registry=reg)

    assert result.metadata["progress"]["constant"]["complete"] == 2
    assert result.metadata["progress"]["other"]["complete"] == 0
    assert result.metadata["progress"]["other"]["expected"] == 2


def cfg_out(cfg):
    from pathlib import Path
    return Path(cfg.run.output_dir) / cfg.run.name


def test_force_discards_completed_checkpoints_for_the_named_models_only(cfg, panel, reg):
    """After an adapter change, a model that *succeeded* under the old code
    must be re-runnable without touching anyone else's checkpoints."""
    BenchmarkRunner(cfg, registry=reg).run(panel, models=["constant", "other"])
    Constant.fits = []

    BenchmarkRunner(cfg, registry=reg, force=["other"]).run(panel, models=["constant", "other"])

    assert _fits("other") > 0, "forced model reran"
    assert _fits("constant") == 0, "unforced model resumed"


# --- provenance: a checkpoint outlives the code that produced it -------------
#
# result_key covers the config, not the code, so an adapter rewrite silently
# leaves its old checkpoints valid and collect() merges measurements from two
# builds into one table. The hazard is a table that *mixes* builds - not one
# that merely predates HEAD, since most commits never touch measurement.

OLD = "a" * 40
NEW = "b" * 40


def _run_at(commit, cfg, panel, reg, models):
    runner = BenchmarkRunner(cfg, registry=reg)
    runner._git_commit = commit
    runner.run(panel, models=models)
    return runner


def test_progress_names_the_commit_each_checkpoint_was_built_at(cfg, panel, reg):
    _run_at(OLD, cfg, panel, reg, ["constant"])

    progress = BenchmarkRunner(cfg, registry=reg).progress(["constant"])

    assert progress["constant"]["complete"] == 2, "the checkpoint still resumes"
    assert progress["constant"]["commits"] == [OLD]


def test_one_build_behind_head_is_not_a_warning(cfg, panel, reg, capsys):
    """A reporting-only commit must not invalidate every measurement in the
    repository - the table is internally comparable, which is what matters."""
    _run_at(OLD, cfg, panel, reg, ["constant", "other"])
    progress = BenchmarkRunner(cfg, registry=reg).progress(["constant", "other"])
    capsys.readouterr()

    stale = warn_if_stale(progress, head=NEW)

    assert stale == []
    assert capsys.readouterr().out == ""


def test_a_table_that_mixes_builds_warns(cfg, panel, reg, capsys, monkeypatch):
    monkeypatch.setattr("tsbench.runner._is_ancestor", lambda a, b: a == OLD and b == NEW)
    _run_at(OLD, cfg, panel, reg, ["constant"])
    _run_at(NEW, cfg, panel, reg, ["other"])
    progress = BenchmarkRunner(cfg, registry=reg).progress(["constant", "other"])
    capsys.readouterr()

    stale = warn_if_stale(progress, head=NEW)
    out = capsys.readouterr().out

    assert stale == ["constant"], "only the model on the older build"
    assert "other" not in out.split("re-measure")[1], "the newest build is not re-run"
    assert "--models constant --force" in out


def test_a_partial_re_measure_is_not_suggested_when_every_build_trails_head(
        cfg, panel, reg, capsys, monkeypatch):
    """--force re-measures at HEAD. If no group is already at HEAD, re-running
    the older ones lands on a third build instead of converging on one."""
    monkeypatch.setattr("tsbench.runner._is_ancestor", lambda a, b: a == OLD and b == NEW)
    _run_at(OLD, cfg, panel, reg, ["constant"])
    _run_at(NEW, cfg, panel, reg, ["other"])
    progress = BenchmarkRunner(cfg, registry=reg).progress(["constant", "other"])
    capsys.readouterr()

    warn_if_stale(progress, head="c" * 40)
    out = capsys.readouterr().out

    assert "--models" not in out, "a partial re-measure would add a third build"
    assert "re-measure all with: --force" in out


def test_mixed_builds_of_unknown_order_name_both_groups(cfg, panel, reg, capsys, monkeypatch):
    """Two commits with no ancestry between them (a rebase, another machine):
    say the table is mixed without guessing which side is authoritative."""
    monkeypatch.setattr("tsbench.runner._is_ancestor", lambda a, b: False)
    _run_at(OLD, cfg, panel, reg, ["constant"])
    _run_at(NEW, cfg, panel, reg, ["other"])
    progress = BenchmarkRunner(cfg, registry=reg).progress(["constant", "other"])
    capsys.readouterr()

    warn_if_stale(progress, head=NEW)
    out = capsys.readouterr().out

    assert OLD[:8] in out and NEW[:8] in out
    assert "constant" in out and "other" in out
    assert "--models" not in out, "no guess about which side to discard"


def test_collect_warns_when_a_table_spans_builds(cfg, panel, reg, capsys, monkeypatch):
    monkeypatch.setattr("tsbench.runner._is_ancestor", lambda a, b: a == OLD and b == NEW)
    _run_at(OLD, cfg, panel, reg, ["constant"])
    _run_at(NEW, cfg, panel, reg, ["other"])
    capsys.readouterr()

    result = collect(cfg, models=["constant", "other"], registry=reg)

    assert "constant" in capsys.readouterr().out
    assert result.metadata["progress"]["constant"]["commits"] == [OLD]
    assert result.metadata["code_versions"] == {OLD: ["constant"], NEW: ["other"]}


def test_a_checkpoint_without_a_commit_column_still_counts_as_complete(cfg, panel, reg):
    """Checkpoints predate the provenance column; reading it must not make
    older work look unfinished."""
    import pandas as pd

    runner = _run_at(OLD, cfg, panel, reg, ["constant"])
    for path in runner.checkpoint_files("constant"):
        if path.name.endswith("__timings.parquet"):
            t = pd.read_parquet(path).drop(columns=["git_commit"])
            t.to_parquet(path, index=False)

    progress = BenchmarkRunner(cfg, registry=reg).progress(["constant"])

    assert progress["constant"]["complete"] == 2
    assert progress["constant"]["commits"] == []
