# Databricks notebook source
# MAGIC %md # Install - run once per cluster start
# MAGIC **Start from a clean environment**: if installs already ran in this
# MAGIC cluster session, detach & re-attach first.
# MAGIC
# MAGIC **torch is pinned to 2.10.0 deliberately.** DBR 17.3 LTS ML ships 2.7.0,
# MAGIC but granite-tsfm 0.3.8 needs torch>=2.10,<2.11 and neuralforecast>=3.2
# MAGIC needs >=2.9.1. torchvision must move with torch (transformers imports it;
# MAGIC a mismatch dies with "operator torchvision::nms does not exist").

# COMMAND ----------

import os
TORCH = "2.10.0"
os.makedirs("/local_disk0/tmp", exist_ok=True)
with open("/local_disk0/tmp/constraints.txt", "w") as fh:
    fh.write(f"torch=={TORCH}\ntorchvision==0.25.0\n")
print("constraints:", open("/local_disk0/tmp/constraints.txt").read().strip())

# COMMAND ----------

# MAGIC %pip install -c /local_disk0/tmp/constraints.txt "torch==2.10.0" "torchvision==0.25.0"
# MAGIC %pip install -c /local_disk0/tmp/constraints.txt --no-deps -e /Workspace/Repos/iman.kohyarnejad@vancereaviejunction.onmicrosoft.com/forecasting-benchmark
# MAGIC %pip install -c /local_disk0/tmp/constraints.txt "statsforecast>=2.1" "mlforecast>=1.1" "neuralforecast>=3.2" "prophet>=1.4" "optuna>=4.0" "pandas>=2.2,<3" pyarrow pyyaml scipy matplotlib
# MAGIC %pip install -c /local_disk0/tmp/constraints.txt "chronos-forecasting>=1.5" "timesfm>=2.0" "tabpfn-time-series>=1.0" "granite-tsfm>=0.3.8" "accelerate>=1.6,<2"

# COMMAND ----------

# MAGIC %md Toto's package metadata is a lockfile (`numpy==1.26.4`,
# MAGIC `transformers==4.52.1`, `datasets==2.17.1`, ...) that would downgrade the
# MAGIC environment, so it installs with `--no-deps` plus its actual inference
# MAGIC dependencies, unpinned.

# COMMAND ----------

# MAGIC %pip install -c /local_disk0/tmp/constraints.txt --no-deps "toto-ts>=0.2"
# MAGIC %pip install -c /local_disk0/tmp/constraints.txt einops jaxtyping rotary-embedding-torch "gluonts[torch]" lightning

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Environment check - fails here rather than an hour into a run

# COMMAND ----------

import importlib, torch, torchvision
assert torch.__version__.startswith("2.10."), f"torch is {torch.__version__} - re-run from a clean env"
torchvision.ops.nms  # raises if torchvision was built for a different torch
from transformers import PreTrainedModel  # noqa: F401
for m in ("statsforecast", "mlforecast", "neuralforecast", "prophet", "optuna",
          "chronos", "timesfm", "tabpfn_time_series", "tsfm_public", "toto"):
    importlib.import_module(m)
import huggingface_hub, transformers, numpy, pandas
print("torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| transformers", transformers.__version__, "| numpy", numpy.__version__,
      "| pandas", pandas.__version__)
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-",
      "| native bf16:", torch.cuda.is_available()
      and torch.cuda.is_bf16_supported(including_emulation=False))
print("environment OK")
