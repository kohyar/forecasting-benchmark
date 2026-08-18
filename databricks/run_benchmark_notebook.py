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
# MAGIC **Start from a clean environment**: if this notebook has already run
# MAGIC installs on this cluster session, detach & re-attach (or restart the
# MAGIC cluster) first, so nothing left over from an earlier attempt lingers.
# MAGIC
# MAGIC **torch is pinned to 2.10.0 on purpose.** DBR 17.3 LTS ML ships torch
# MAGIC 2.7.0, but the current TTM package (`granite-tsfm 0.3.8`) requires
# MAGIC `torch>=2.10,<2.11` and `neuralforecast>=3.2` requires `>=2.9.1`; the
# MAGIC older releases that accept 2.7 lose the TTM variant selection and hard-pin
# MAGIC transformers. So the runtime torch is replaced with the 2.10.0 CUDA-12.8
# MAGIC wheel, up front and under a constraint, so nothing later can move it.
# MAGIC `run_metadata.json` records the version that actually ran.

# COMMAND ----------

import os
TORCH = "2.10.0"
os.makedirs("/local_disk0/tmp", exist_ok=True)
with open("/local_disk0/tmp/constraints.txt", "w") as fh:
    fh.write(f"torch=={TORCH}\n")
print("constraint:", open("/local_disk0/tmp/constraints.txt").read().strip())

# COMMAND ----------

# MAGIC %pip install -c /local_disk0/tmp/constraints.txt "torch==2.10.0"
# MAGIC %pip install -c /local_disk0/tmp/constraints.txt --no-deps -e /Workspace/Repos/iman.kohyarnejad@vancereaviejunction.onmicrosoft.com/forecasting-benchmark
# MAGIC %pip install -c /local_disk0/tmp/constraints.txt "statsforecast>=2.1" "mlforecast>=1.1" "neuralforecast>=3.2" "prophet>=1.4" "optuna>=4.0" "pandas>=2.2,<3" pyarrow pyyaml scipy matplotlib
# MAGIC %pip install -c /local_disk0/tmp/constraints.txt "chronos-forecasting>=1.5" "timesfm>=2.0" "tabpfn-time-series>=1.0" "granite-tsfm>=0.3.8" "accelerate>=1.6,<2"

# COMMAND ----------

# MAGIC %md
# MAGIC Toto's package metadata is a lockfile - it pins `numpy==1.26.4`,
# MAGIC `pandas==2.2.3`, `transformers==4.52.1`, `datasets==2.17.1` and more
# MAGIC with `==`. Installing those would downgrade the whole environment under
# MAGIC a torch built for numpy 2, so Toto goes in with `--no-deps` and only the
# MAGIC libraries its inference path actually imports are added, unpinned.

# COMMAND ----------

# MAGIC %pip install -c /local_disk0/tmp/constraints.txt --no-deps "toto-ts>=0.2"
# MAGIC %pip install -c /local_disk0/tmp/constraints.txt einops jaxtyping rotary-embedding-torch "gluonts[torch]" lightning

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
      "| native bf16:", torch.cuda.is_available()
      and torch.cuda.is_bf16_supported(including_emulation=False))
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

# MAGIC %md ### If anything failed above: the recorded tracebacks (runs nothing)

# COMMAND ----------

sh("scripts/run_benchmark.py", "--config", CONFIG, "--n-series", "50",
   "--models", "chronos2,timesfm,toto,tabpfn_ts,ttm", "--run-name", "smoke-50", "--errors")

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
