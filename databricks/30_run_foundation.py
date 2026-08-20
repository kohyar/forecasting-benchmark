# Databricks notebook source
# MAGIC %md # Tier: foundation models (zero-shot)
# MAGIC chronos2, timesfm, toto, tabpfn_ts, ttm. fit = checkpoint loading,
# MAGIC predict = pure inference. TabPFN-TS runs ~1.85 s/series and dominates
# MAGIC this tier's wall clock (~15 h at 1,000 series); to overlap, run this
# MAGIC notebook on a second identical cluster while 10/20 run on the first.

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

# MAGIC %md ## Warm the weight cache (5 series - no timed fold pays a download)

# COMMAND ----------

sh("scripts/run_foundation.py", "--config", CONFIG, "--n-series", "5",
   "--run-name", "warmup", "--no-mlflow")

# COMMAND ----------

# MAGIC %md ## Smoke: 50 series

# COMMAND ----------

sh("scripts/run_foundation.py", "--config", CONFIG, "--n-series", "50",
   "--run-name", "smoke-50", "--no-mlflow")

# COMMAND ----------

# MAGIC %md ## Real run: 1,000 series

# COMMAND ----------

sh("scripts/run_foundation.py", "--config", CONFIG)

# COMMAND ----------

# MAGIC %md ## Status / recorded failures for this tier (runs nothing)

# COMMAND ----------

sh("scripts/run_foundation.py", "--config", CONFIG, "--status")

# COMMAND ----------

sh("scripts/run_foundation.py", "--config", CONFIG, "--errors")
