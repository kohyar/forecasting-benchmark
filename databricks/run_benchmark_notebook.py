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
# torchvision must move with torch: the runtime's build is for 2.7, and
# transformers imports it, so a mismatch surfaces as
# "operator torchvision::nms does not exist" inside every transformers model.
with open("/local_disk0/tmp/constraints.txt", "w") as fh:
    fh.write(f"torch=={TORCH}\ntorchvision==0.25.0\n")
print("constraint:", open("/local_disk0/tmp/constraints.txt").read().strip())

# COMMAND ----------

# MAGIC %pip install -c /local_disk0/tmp/constraints.txt "torch==2.10.0" "torchvision==0.25.0"
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

# MAGIC %md ### Environment check - fails here rather than an hour into a run

# COMMAND ----------

import importlib, torch, torchvision
assert torch.__version__.startswith("2.10."), f"torch is {torch.__version__}, expected 2.10.x - re-run cell 1 from a clean env"
torchvision.ops.nms  # raises if torchvision was built for a different torch
from transformers import PreTrainedModel  # noqa: F401  (fails on the torchvision mismatch)
for m in ("statsforecast", "mlforecast", "neuralforecast", "prophet", "optuna",
          "chronos", "timesfm", "tabpfn_time_series", "tsfm_public", "toto"):
    importlib.import_module(m)
import huggingface_hub, transformers, numpy, pandas
print("torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| transformers", transformers.__version__, "| huggingface_hub", huggingface_hub.__version__,
      "| numpy", numpy.__version__, "| pandas", pandas.__version__)
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("environment OK")

# COMMAND ----------

import os, subprocess, sys

VOLUME = "/Volumes/forecaster_develop/bronze/benchmark"
REPO = "/Workspace/Repos/iman.kohyarnejad@vancereaviejunction.onmicrosoft.com/forecasting-benchmark"
CONFIG = f"{REPO}/configs/databricks-t4.yaml"

os.chdir(REPO)
os.makedirs(f"{VOLUME}/results", exist_ok=True)
os.makedirs("/local_disk0/tmp", exist_ok=True)

# TabPFN needs a one-time licence acceptance and an API key for local
# inference (register at https://ux.priorlabs.ai, accept the licence, copy the
# key from /account). Either set TABPFN_TOKEN as a cluster environment
# variable (Compute > Advanced > Environment variables; takes effect after a
# cluster restart) or store it as the secret benchmark/tabpfn_token. Worker
# processes inherit the environment, so setting it once is enough.
if os.environ.get("TABPFN_TOKEN"):
    print("TABPFN_TOKEN present (cluster environment)")
else:
    try:
        os.environ["TABPFN_TOKEN"] = dbutils.secrets.get(scope="benchmark", key="tabpfn_token")
        print("TABPFN_TOKEN set from secret benchmark/tabpfn_token")
    except Exception:
        print("TABPFN_TOKEN not set - tabpfn_ts will fail with a licence error until it is")

# Worker processes must see exactly what this notebook sees: the repo's src
# and the notebook-scoped site-packages that %pip just installed.
extra = [p for p in sys.path if p and ("pythonEnv" in p or p.endswith("site-packages"))]
os.environ["PYTHONPATH"] = os.pathsep.join(
    [f"{REPO}/src", *extra, os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)


def sh(*args):
    """Run a repo script with the notebook's interpreter, streaming output
    line by line (-u: otherwise the child's stdout is block-buffered and only
    appears when it exits)."""
    proc = subprocess.run([sys.executable, "-u", *args], text=True,
                          env={**os.environ, "PYTHONUNBUFFERED": "1"})
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

# MAGIC %md ## 3b. Warm the weight cache
# MAGIC Five series, foundation models only. Its purpose is to pull every
# MAGIC checkpoint into the local cache so no timed fold includes a download.

# COMMAND ----------

sh("scripts/run_benchmark.py", "--config", CONFIG, "--n-series", "5",
   "--models", "chronos2,timesfm,toto,tabpfn_ts,ttm", "--run-name", "warmup", "--no-mlflow")

# COMMAND ----------

# MAGIC %md ## 4. Foundation adapters on 50 series
# MAGIC A failure is recorded, not fatal; fix the adapter, pull in Repos,
# MAGIC re-run this cell - only the failed ones rerun. After an adapter change
# MAGIC that should be re-validated even where it previously *succeeded*, add
# MAGIC `"--force"` to the call once (it discards only these five models'
# MAGIC checkpoints for this run), then remove it again.

# COMMAND ----------

sh("scripts/run_benchmark.py", "--config", CONFIG, "--n-series", "50",
   "--models", "chronos2,timesfm,toto,tabpfn_ts,ttm", "--run-name", "smoke-50", "--no-mlflow")

# COMMAND ----------

# MAGIC %md ### If anything failed above: the recorded tracebacks (runs nothing)

# COMMAND ----------

sh("scripts/run_benchmark.py", "--config", CONFIG, "--n-series", "50",
   "--models", "chronos2,timesfm,toto,tabpfn_ts,ttm", "--run-name", "smoke-50", "--errors")

# COMMAND ----------

# MAGIC %md ### Diagnostics (run only when something above is puzzling)
# MAGIC **TabPFN licence** - the env var alone is not enough: the key must
# MAGIC verify, and the licence for *this model version* (TabPFN v3, repo
# MAGIC `Prior-Labs/tabpfn_3`) must be accepted on the Licenses tab at
# MAGIC ux.priorlabs.ai. This prints which of the three is missing.

# COMMAND ----------

import os
from tabpfn.settings import settings
from tabpfn.browser_auth import (_get_license_name, check_license_accepted,
                                 get_cached_token, verify_token)
tok = get_cached_token()
print("TABPFN_TOKEN present:", bool(tok), "| length:", len(tok or ""))
api = settings.tabpfn.auth_api_url
print("key verifies:", verify_token(tok, api) if tok else "-")
try:
    lic = _get_license_name("tabpfn_3")
    print("licence required for v3:", lic)
    print("accepted:", check_license_accepted(tok, api, lic) if tok else "-")
except Exception as exc:
    print("could not resolve the v3 licence name (HF access?):", exc)

# COMMAND ----------

# MAGIC %md **Toto load time** - 50 s per load is more than 600 MB of weights
# MAGIC should take. This splits it into imports / snapshot lookup / load.

# COMMAND ----------

import os, time, torch
t0 = time.perf_counter()
import toto.model.toto, toto.inference.forecaster, toto.data.util.dataset  # noqa: E401
print(f"imports        {time.perf_counter() - t0:6.1f}s")
from huggingface_hub import snapshot_download
t0 = time.perf_counter()
d = snapshot_download("Datadog/Toto-Open-Base-1.0", allow_patterns=["*.json", "*.safetensors"])
print(f"snapshot       {time.perf_counter() - t0:6.1f}s  -> {d}")
t0 = time.perf_counter()
m = toto.model.toto.Toto.load_from_checkpoint(d, map_location="cuda", strict=False)
print(f"load_from_ckpt {time.perf_counter() - t0:6.1f}s")
print("HF cache:", os.environ.get("HF_HOME"), os.environ.get("HF_HUB_CACHE"))

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
