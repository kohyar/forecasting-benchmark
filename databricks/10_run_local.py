# Databricks notebook source
# MAGIC %md # Tier: baselines + local models
# MAGIC naive, seasonal_naive, autoets, autoarima, autotheta, prophet,
# MAGIC lightgbm_local. CPU-bound; AutoARIMA at season_length=52 is the long
# MAGIC pole. Every cell resumes from checkpoints - re-run freely.

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

# MAGIC %md ## Smoke: 50 series

# COMMAND ----------

sh("scripts/run_local.py", "--config", CONFIG, "--n-series", "50",
   "--run-name", "smoke-50", "--no-mlflow")

# COMMAND ----------

# MAGIC %md ## Real run: 1,000 series

# COMMAND ----------

sh("scripts/run_local.py", "--config", CONFIG)

# COMMAND ----------

# MAGIC %md ## Status / recorded failures for this tier (runs nothing)

# COMMAND ----------

sh("scripts/run_local.py", "--config", CONFIG, "--status")

# COMMAND ----------

sh("scripts/run_local.py", "--config", CONFIG, "--errors")
