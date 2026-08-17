"""Adapter modules must not drag in each other's libraries.

lightgbm and torch each bundle an OpenMP runtime; exercising both in one
process segfaults on macOS arm64 (exit 139), which no try/except can catch.
The harness runs each model in its own process, and that only helps if
importing one adapter module does not load the other's library.
"""
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.slow

PROBE = textwrap.dedent("""
    import importlib, sys
    importlib.import_module("{module}")
    top = {{m for m in sys.modules if "." not in m}}
    print(",".join(sorted(top & {{"torch", "lightgbm", "neuralforecast", "mlforecast"}})))
""")


def _loaded(module: str) -> set:
    proc = subprocess.run([sys.executable, "-c", PROBE.format(module=module)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return {m for m in proc.stdout.strip().split(",") if m}


def test_a_statsforecast_worker_loads_neither_torch_nor_lightgbm():
    assert _loaded("tsbench.models.statsforecast_adapters") == set()


def test_a_lightgbm_worker_does_not_load_torch():
    loaded = _loaded("tsbench.models.lightgbm_adapters")

    assert "torch" not in loaded


def test_a_neural_worker_does_not_load_lightgbm():
    loaded = _loaded("tsbench.models.neural_adapters")

    assert "lightgbm" not in loaded


def test_the_registry_still_sees_every_adapter():
    from tsbench.models import registry

    names = registry.default().names()

    assert {"naive", "lightgbm_local", "lightgbm_global", "tft", "prophet"} <= set(names)
