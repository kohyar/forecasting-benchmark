# Databricks notebook source
# MAGIC %md # Common setup (no installs here)
# MAGIC Included by every runner notebook via `%run ./00_common`. Wires the
# MAGIC shared library directory that `01_install` populated onto PYTHONPATH -
# MAGIC `%pip` would be notebook-scoped and invisible to the other notebooks.

# COMMAND ----------

import os, subprocess, sys

# Set both to your own workspace before the first run.
VOLUME = "/Volumes/forecaster_develop/bronze/benchmark"
REPO = "/Workspace/Repos/<your-databricks-user>/forecasting-benchmark"
CONFIG = f"{REPO}/configs/databricks-t4.yaml"
LIBS = "/local_disk0/tsbench-libs"

if not os.path.isdir(os.path.join(LIBS, "statsforecast")):
    raise RuntimeError(
        f"{LIBS} is missing or incomplete - run the 01_install notebook once "
        "on this cluster (it installs after every cluster start).")

os.chdir(REPO)
os.makedirs(f"{VOLUME}/results", exist_ok=True)
os.makedirs("/local_disk0/tmp", exist_ok=True)

# Workers and any notebook-side imports resolve the shared libs first, then
# the repo's src; the cluster runtime's own site-packages stay behind them.
os.environ["PYTHONPATH"] = os.pathsep.join(
    [f"{REPO}/src", LIBS, os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)
for path in (LIBS, f"{REPO}/src"):
    if path not in sys.path:
        sys.path.insert(0, path)

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


def sh(*args):
    """Run a repo script with this interpreter, streaming output line by line
    (-u: otherwise the child's stdout is block-buffered)."""
    proc = subprocess.run([sys.executable, "-u", *args], text=True,
                          env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}: {' '.join(args)}")

print("common setup done:", CONFIG)
