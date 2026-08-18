# Databricks notebook source
# MAGIC %md
# MAGIC # SPINS weekly benchmark - Databricks runner
# MAGIC
# MAGIC Run cell by cell the first time. Every cell is safe to re-run: the
# MAGIC benchmark resumes from `checkpoints/` on the Volume, so a cluster
# MAGIC restart, a killed cell or a failed model costs only the unfinished
# MAGIC (model, fold) units. Keep this notebook cell running while the
# MAGIC benchmark runs - a running command is what stops auto-termination.

# COMMAND ----------

# MAGIC %md ## 0. Settings - edit these three

# COMMAND ----------

VOLUME = "/Volumes/CATALOG/SCHEMA/VOLUME/tsbench"   # Unity Catalog volume root
REPO = "/Workspace/Repos/<you>/spins-forecasting-benchmark"  # Repos clone
CONFIG = f"{REPO}/configs/databricks-t4.yaml"

# COMMAND ----------

# MAGIC %md ## 1. Install (once per cluster start; runtime torch is kept)

# COMMAND ----------

# MAGIC %pip install --no-deps -e $REPO
# MAGIC %pip install "statsforecast>=2.1" "mlforecast>=1.1" "neuralforecast>=3.2" "prophet>=1.4" "optuna>=4.0" pyarrow pyyaml scipy matplotlib
# MAGIC %pip install "chronos-forecasting>=1.5" "timesfm>=2.0" "tabpfn-time-series>=1.0" "granite-tsfm>=0.2"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os, subprocess, sys
VOLUME = "/Volumes/CATALOG/SCHEMA/VOLUME/tsbench"
REPO = "/Workspace/Repos/<you>/spins-forecasting-benchmark"
CONFIG = f"{REPO}/configs/databricks-t4.yaml"
os.chdir(REPO)
os.makedirs(f"{VOLUME}/results", exist_ok=True)
os.makedirs("/local_disk0/tmp", exist_ok=True)

def sh(*args):
    """Run a script and stream its output; raise if it fails."""
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
sh("scripts/run_benchmark.py", "--config", CONFIG, "--list-models")
print(os.listdir(VOLUME))

# COMMAND ----------

# MAGIC %md ## 3. Smoke test - 50 series, two baselines. Twenty minutes; proves the pipeline here.

# COMMAND ----------

sh("scripts/run_benchmark.py", "--config", CONFIG, "--n-series", "50",
   "--models", "naive,seasonal_naive", "--run-name", "smoke-50")

# COMMAND ----------

# MAGIC %md ## 4. Foundation adapters, one at a time on 50 series
# MAGIC These five are unverified against real weights. Each failure is
# MAGIC recorded, not fatal; fix the adapter and re-run the cell - only the
# MAGIC failed one reruns.

# COMMAND ----------

sh("scripts/run_benchmark.py", "--config", CONFIG, "--n-series", "50",
   "--models", "chronos2,timesfm,toto,tabpfn_ts,ttm", "--run-name", "smoke-50")

# COMMAND ----------

# MAGIC %md ## 5. Full run. Re-run this cell as many times as needed - it resumes.

# COMMAND ----------

sh("scripts/run_benchmark.py", "--config", CONFIG)

# COMMAND ----------

# MAGIC %md ## Progress / partial results at any time (does not run anything)

# COMMAND ----------

sh("scripts/run_benchmark.py", "--config", CONFIG, "--status")
sh("scripts/run_benchmark.py", "--config", CONFIG, "--collect-only")

# COMMAND ----------

# MAGIC %md ## 6. Statistics and tables, once the run is complete

# COMMAND ----------

sh("scripts/run_stats.py", "--run", f"{VOLUME}/results/spins-weekly-v1")
