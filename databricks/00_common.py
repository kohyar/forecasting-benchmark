# Databricks notebook source
# MAGIC %md # Common setup (no installs here)
# MAGIC Included by every runner notebook via `%run ./00_common`. Paths, the
# MAGIC worker environment, and the `sh()` helper. Installs live in
# MAGIC `01_install` because `%pip` + `restartPython()` cannot run inside a
# MAGIC `%run` include.

# COMMAND ----------

import os, subprocess, sys

VOLUME = "/Volumes/forecaster_develop/bronze/benchmark"
REPO = "/Workspace/Repos/iman.kohyarnejad@vancereaviejunction.onmicrosoft.com/forecasting-benchmark"
CONFIG = f"{REPO}/configs/databricks-t4.yaml"

os.chdir(REPO)
os.makedirs(f"{VOLUME}/results", exist_ok=True)
os.makedirs("/local_disk0/tmp", exist_ok=True)

# Optional: only needed to run TabPFN's gated v3 checkpoint instead of the
# default (ungated) v2 weights.
if os.environ.get("TABPFN_TOKEN"):
    print("TABPFN_TOKEN present (cluster environment)")
else:
    try:
        os.environ["TABPFN_TOKEN"] = dbutils.secrets.get(scope="benchmark", key="tabpfn_token")
        print("TABPFN_TOKEN set from secret benchmark/tabpfn_token")
    except Exception:
        print("TABPFN_TOKEN not set (fine: tabpfn_ts defaults to the ungated v2 weights)")

# Worker processes must see exactly what this notebook sees: the repo's src
# and the notebook-scoped site-packages that %pip installed.
extra = [p for p in sys.path if p and ("pythonEnv" in p or p.endswith("site-packages"))]
os.environ["PYTHONPATH"] = os.pathsep.join(
    [f"{REPO}/src", *extra, os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)


def sh(*args):
    """Run a repo script with the notebook's interpreter, streaming output
    line by line (-u: otherwise the child's stdout is block-buffered)."""
    proc = subprocess.run([sys.executable, "-u", *args], text=True,
                          env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}: {' '.join(args)}")

print("common setup done:", CONFIG)
