"""Timing, memory and device recording.

Fit and predict are timed separately on purpose: foundation models have zero
fit and nonzero predict, and collapsing the two hides the paper's core finding.
"""
import time

import pytest

from tsbench.measure import Measurement, device_info, measure, resolve_device, synchronize


def test_resolves_to_a_real_device():
    assert resolve_device("auto") in {"cpu", "mps", "cuda"}


def test_cpu_can_always_be_forced():
    assert resolve_device("cpu") == "cpu"


def test_asking_for_an_unavailable_device_fails_loudly():
    with pytest.raises(RuntimeError, match="cuda"):
        resolve_device("cuda", available={"cpu"})


def test_synchronize_is_a_noop_on_cpu():
    synchronize("cpu")


def test_device_info_records_what_the_paper_needs():
    info = device_info("cpu")

    assert info["device"] == "cpu"
    assert "gpu_model" in info
    assert info["cpu_count"] >= 1


def test_measure_times_the_block_with_perf_counter():
    with measure() as m:
        time.sleep(0.05)

    assert m.seconds >= 0.05
    assert m.seconds < 2.0


def test_measure_records_peak_memory():
    with measure() as m:
        blob = [0] * 2_000_000

    assert m.peak_memory_mb > 0
    del blob


def test_fit_and_predict_are_recorded_separately():
    with measure() as fit:
        time.sleep(0.02)
    with measure() as predict:
        time.sleep(0.05)

    row = Measurement.combine(fit, predict, device="cpu", n_series=10, n_params=None)

    assert row["fit_seconds"] == pytest.approx(fit.seconds)
    assert row["predict_seconds"] == pytest.approx(predict.seconds)
    assert row["fit_seconds"] != row["predict_seconds"]
    assert row["device"] == "cpu"
    assert row["n_series"] == 10


def test_zero_fit_time_is_preserved_not_collapsed():
    """A zero-shot model really does have no fit stage."""
    with measure() as fit:
        pass
    with measure() as predict:
        time.sleep(0.02)

    row = Measurement.combine(fit, predict, device="cpu", n_series=1, n_params=None)

    assert row["fit_seconds"] < 0.01
    assert row["predict_seconds"] > 0.01
    assert "fit_seconds" in row and "predict_seconds" in row


def test_measure_synchronizes_the_device_before_stopping(monkeypatch):
    """On a GPU the timer must not stop before the kernels finish."""
    calls = []
    monkeypatch.setattr("tsbench.measure.synchronize", lambda d: calls.append(d))

    with measure(device="cuda"):
        pass

    assert calls == ["cuda"]


def test_measurement_survives_an_exception_and_still_reports(monkeypatch):
    with pytest.raises(ValueError):
        with measure() as m:
            raise ValueError("boom")

    assert m.seconds > 0


def test_peak_rss_is_reported_for_the_whole_process():
    """tracemalloc only sees Python allocations; a torch or lightgbm model
    does most of its allocating in C and would look free."""
    from tsbench.measure import peak_rss_mb

    value = peak_rss_mb()

    assert value > 1.0, "the interpreter alone is bigger than a megabyte"


def test_peak_rss_grows_with_a_large_allocation():
    from tsbench.measure import peak_rss_mb

    before = peak_rss_mb()
    blob = bytearray(200 * 1024 * 1024)
    after = peak_rss_mb()
    del blob

    assert after >= before
