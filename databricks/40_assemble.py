# Databricks notebook source
# MAGIC %md # Assemble + statistics
# MAGIC Combines every tier's checkpoints into the final tables (running no
# MAGIC models), then the significance tests and critical-difference diagrams.
# MAGIC Safe to run at any time for a partial view.

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

# MAGIC %md ## Overall progress across all tiers

# COMMAND ----------

sh("scripts/run_benchmark.py", "--config", CONFIG, "--status")

# COMMAND ----------

# MAGIC %md ## Assemble the combined results (runs nothing)

# COMMAND ----------

sh("scripts/run_benchmark.py", "--config", CONFIG, "--collect-only")

# COMMAND ----------

# MAGIC %md ## Statistics, tables and diagrams (needs >=3 models complete)

# COMMAND ----------

sh("scripts/run_stats.py", "--run", f"{VOLUME}/results/spins-weekly-v1")

# COMMAND ----------

# MAGIC %md ## The paper's tables and figures
# MAGIC Writes `results/spins-weekly-v1/paper/`: T3, T4, T5, T7 and T8 as booktabs
# MAGIC LaTeX plus CSV, and F1, F8 and the coverage figure as vector PDF and 300dpi
# MAGIC PNG. F7 comes from the statistics cell above, in `analysis/`.

# COMMAND ----------

sh("scripts/build_paper_artifacts.py", "--run", f"{VOLUME}/results/spins-weekly-v1")

# COMMAND ----------

# MAGIC %md ## Same, for the 50-series smoke run

# COMMAND ----------

sh("scripts/run_benchmark.py", "--config", CONFIG, "--n-series", "50",
   "--run-name", "smoke-50", "--collect-only")
sh("scripts/run_stats.py", "--run", f"{VOLUME}/results/smoke-50")
