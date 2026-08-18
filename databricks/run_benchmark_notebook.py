# Databricks notebook source
# MAGIC %md
# MAGIC # SPINS weekly benchmark - Databricks runner
# MAGIC
# MAGIC Run cell by cell the first time. Every cell is safe to re-run: the
# MAGIC benchmark resumes from `checkpoints/` on the Volume, so a cluster
# MAGIC restart, a killed cell or a failed model costs only the unfinished
# MAGIC (model, fold) units. Keep the running cell open - a running command is
# MAGIC what stops the cluster's auto-termination.

# COMMAND ----------

# MAGIC %md ## 1. Install (once per cluster start)
# MAGIC Four cells. The first pins the torch this environment already has, so no
# MAGIC later install can downgrade it - pip was seen trying to pull torch 2.8
# MAGIC while backtracking. Toto is installed separately because it pins
# MAGIC `datasets==2.17.1` (for its evaluation code, not inference), which
# MAGIC `tabpfn-time-series` cannot accept.

# COMMAND ----------

import os, torch
os.makedirs("/local_disk0/tmp", exist_ok=True)
with open("/local_disk0/tmp/constraints.txt", "w") as fh:
    fh.write(f"torch=={torch.__version__.split('+')[0]}\n")
print("pinned", open("/local_disk0/tmp/constraints.txt").read().strip(),
      "| cuda:", torch.cuda.is_available())

# COMMAND ----------

# MAGIC %pip install -c /local_disk0/tmp/constraints.txt --no-deps -e /Workspace/Repos/iman.kohyarnejad@vancereaviejunction.onmicrosoft.com/forecasting-benchmark
# MAGIC %pip install -c /local_disk0/tmp/constraints.txt "statsforecast>=2.1" "mlforecast>=1.1" "neuralforecast>=3.2" "prophet>=1.4" "optuna>=4.0" "pandas>=2.2,<3" pyarrow pyyaml scipy matplotlib
# MAGIC %pip install -c /local_disk0/tmp/constraints.txt "chronos-forecasting>=1.5" "timesfm>=2.0" "tabpfn-time-series>=1.0" "granite-tsfm>=0.2"

# COMMAND ----------

# Toto without its declared deps, then its runtime deps minus the datasets pin.
import importlib.metadata as md, re, subprocess, sys
print("installing into:", sys.executable)
pip = [sys.executable, "-m", "pip", "install", "-c", "/local_disk0/tmp/constraints.txt"]
subprocess.run([*pip, "--no-deps", "toto-ts>=0.1"], check=True)
runtime_deps = [
    r for r in (md.requires("toto-ts") or [])
    if ";" not in r                                                    # no extras / markers
    and re.split(r"[<>=!~\[ ]", r)[0].lower() not in ("datasets", "torch")
]
print("toto runtime deps:", runtime_deps)
subprocess.run([*pip, *runtime_deps], check=True)

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, subprocess, sys

VOLUME = "/Volumes/forecaster_develop/bronze/benchmark"
REPO = "/Workspace/Repos/iman.kohyarnejad@vancereaviejunction.onmicrosoft.com/forecasting-benchmark"
CONFIG = f"{REPO}/configs/databricks-t4.yaml"

os.chdir(REPO)
os.makedirs(f"{VOLUME}/results", exist_ok=True)
os.makedirs("/local_disk0/tmp", exist_ok=True)

# Worker processes must see exactly what this notebook sees: the repo's src
# and the notebook-scoped site-packages that %pip just installed.
extra = [p for p in sys.path if p and ("pythonEnv" in p or p.endswith("site-packages"))]
os.environ["PYTHONPATH"] = os.pathsep.join(
    [f"{REPO}/src", *extra, os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)


def sh(*args):
    """Run a repo script with the notebook's interpreter, streaming output."""
    proc = subprocess.run([sys.executable, *args], text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}: {' '.join(args)}")

# COMMAND ----------

# MAGIC %md ## 2. Sanity: GPU, adapters, data on the Volume

# COMMAND ----------

import torch
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-",
      "| bf16:", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
print("data:", os.path.exists(f"{VOLUME}/New_Query_2026_06_01_10_17_28.csv"))
sh("scripts/run_benchmark.py", "--config", CONFIG, "--list-models")

# COMMAND ----------

# MAGIC %md ## 3. Smoke test - 50 series, two baselines. Proves the pipeline here and gives the per-hour projection.

# COMMAND ----------

sh("scripts/run_benchmark.py", "--config", CONFIG, "--n-series", "50",
   "--models", "naive,seasonal_naive", "--run-name", "smoke-50", "--no-mlflow")

# COMMAND ----------

# MAGIC %md ## 4. Foundation adapters on 50 series
# MAGIC These five are unverified against real weights. A failure is recorded,
# MAGIC not fatal; fix the adapter, `git pull` in Repos, re-run this cell -
# MAGIC only the failed ones rerun.

# COMMAND ----------

sh("scripts/run_benchmark.py", "--config", CONFIG, "--n-series", "50",
   "--models", "chronos2,timesfm,toto,tabpfn_ts,ttm", "--run-name", "smoke-50", "--no-mlflow")

# COMMAND ----------

# MAGIC %md ## 5. Everything on 50 series - the full roster, one pass, before spending on 1,000

# COMMAND ----------

sh("scripts/run_benchmark.py", "--config", CONFIG, "--n-series", "50",
   "--run-name", "smoke-50", "--no-mlflow")

# COMMAND ----------

# MAGIC %md ## 6. Full run. Re-run this cell as many times as needed - it resumes.

# COMMAND ----------

sh("scripts/run_benchmark.py", "--config", CONFIG)

# COMMAND ----------

# MAGIC %md ## Progress / partial results at any time (runs nothing)

# COMMAND ----------

sh("scripts/run_benchmark.py", "--config", CONFIG, "--status")
sh("scripts/run_benchmark.py", "--config", CONFIG, "--collect-only")

# COMMAND ----------

# MAGIC %md ## 7. Statistics and tables, once the run is complete

# COMMAND ----------

sh("scripts/run_stats.py", "--run", f"{VOLUME}/results/spins-weekly-v1")
