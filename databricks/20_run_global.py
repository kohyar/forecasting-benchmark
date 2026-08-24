# Databricks notebook source
# MAGIC %md # Tier: global models
# MAGIC lightgbm_global, tft, patchtst, nhits, dlinear. The neural trainers
# MAGIC refit per horizon (the horizon is baked into the architecture), so
# MAGIC expect fit to dominate. Every cell resumes - re-run freely.

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

# MAGIC %md ## Smoke: 50 series

# COMMAND ----------

sh("scripts/run_global.py", "--config", CONFIG, "--n-series", "50",
   "--run-name", "smoke-50", "--no-mlflow")

# COMMAND ----------

# MAGIC %md ## Real run: 1,000 series

# COMMAND ----------

sh("scripts/run_global.py", "--config", CONFIG, "--tune")

# COMMAND ----------

# MAGIC %md ## Status / recorded failures for this tier (runs nothing)

# COMMAND ----------

sh("scripts/run_global.py", "--config", CONFIG, "--status")

# COMMAND ----------

sh("scripts/run_global.py", "--config", CONFIG, "--errors")
