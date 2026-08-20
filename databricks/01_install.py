# Databricks notebook source
# MAGIC %md # Install - run once per cluster start
# MAGIC `%pip` is notebook-scoped on Databricks: packages installed that way are
# MAGIC invisible to every other notebook. So this installs into a **shared
# MAGIC directory** on the cluster's local disk instead; `00_common` puts it on
# MAGIC `PYTHONPATH`, which both the runner notebooks' subprocesses and any
# MAGIC direct imports resolve first. No notebook environment is modified, so
# MAGIC there is no restart and this is safe to re-run.
# MAGIC
# MAGIC **torch is pinned to 2.10.0 deliberately.** The runtime ships 2.7.0, but
# MAGIC granite-tsfm 0.3.8 needs torch>=2.10,<2.11 and neuralforecast>=3.2 needs
# MAGIC >=2.9.1. torchvision must move with torch (transformers imports it; a
# MAGIC mismatch dies with "operator torchvision::nms does not exist").

# COMMAND ----------

import os, subprocess, sys, time

LIBS = "/local_disk0/tsbench-libs"
os.makedirs("/local_disk0/tmp", exist_ok=True)
with open("/local_disk0/tmp/constraints.txt", "w") as fh:
    fh.write("torch==2.10.0\ntorchvision==0.25.0\n")

def pip(*args):
    cmd = [sys.executable, "-m", "pip", "install",
           "--target", LIBS, "--upgrade",
           "-c", "/local_disk0/tmp/constraints.txt", *args]
    print(">>>", " ".join(args)[:120])
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    print(f"    done in {time.perf_counter() - t0:.0f}s")

# One resolve for everything except Toto, so nothing can tug torch around.
# Toto's metadata is a lockfile (numpy==1.26.4, transformers==4.52.1, ...) that
# would downgrade the world - it goes in with --no-deps; its actual inference
# dependencies are already in the main set.
pip("torch==2.10.0", "torchvision==0.25.0",
    "statsforecast>=2.1", "mlforecast>=1.1", "neuralforecast>=3.2",
    "prophet>=1.4", "optuna>=4.0", "pandas>=2.2,<3",
    "pyarrow", "pyyaml", "scipy", "matplotlib",
    "chronos-forecasting>=1.5", "timesfm>=2.0", "tabpfn-time-series>=1.0",
    "granite-tsfm>=0.3.8", "accelerate>=1.6,<2",
    "einops", "jaxtyping", "rotary-embedding-torch", "gluonts[torch]", "lightning")
pip("--no-deps", "toto-ts>=0.2")
print("installed into", LIBS)

# COMMAND ----------

# MAGIC %md ## Environment check - as the workers will see it

# COMMAND ----------

import sys
LIBS = "/local_disk0/tsbench-libs"
if LIBS not in sys.path:
    sys.path.insert(0, LIBS)

import importlib, torch, torchvision
assert torch.__version__.startswith("2.10."), f"torch is {torch.__version__}"
torchvision.ops.nms  # raises if torchvision was built for a different torch
from transformers import PreTrainedModel  # noqa: F401
for m in ("statsforecast", "mlforecast", "neuralforecast", "prophet", "optuna",
          "chronos", "timesfm", "tabpfn_time_series", "tsfm_public", "toto"):
    importlib.import_module(m)
import numpy, pandas, transformers
print("torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| transformers", transformers.__version__, "| numpy", numpy.__version__,
      "| pandas", pandas.__version__)
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-",
      "| native bf16:", torch.cuda.is_available()
      and torch.cuda.is_bf16_supported(including_emulation=False))
print("environment OK")
