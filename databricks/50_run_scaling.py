# Databricks notebook source
# MAGIC %md # Scaling: the cardinality ladder
# MAGIC Four example models - prophet, nhits, dlinear, chronos2 - over nested
# MAGIC samples of 500, 1,000, 2,000 and 4,000 series drawn from the eligible
# MAGIC pool. Each rung is its own run (`spins-weekly-scaling-n<N>`), so the
# MAGIC paper's `spins-weekly-v1` checkpoints are never touched; the tuned
# MAGIC parameters of the N=1,000 search are copied into every rung so all four
# MAGIC resolve the same result keys. The N=1,000 rung re-measures the paper's
# MAGIC sample on this build, which keeps the curve one build end to end.
# MAGIC Every cell resumes from checkpoints - re-run freely.

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

import json
import shutil

SCALING = f"{VOLUME}/scaling"
PAPER_RUN = f"{VOLUME}/results/spins-weekly-v1"
RUNGS = (500, 1000, 2000, 4000)
MODELS = "prophet,nhits,dlinear,chronos2"


def rung_run(n: int) -> str:
    return f"spins-weekly-scaling-n{n}"


def rung_config(n: int) -> str:
    # The N=1,000 rung is the paper's own frozen sample, so it runs the base config.
    return CONFIG if n == 1000 else f"{SCALING}/configs/n{n}.yaml"


os.makedirs(f"{SCALING}/configs", exist_ok=True)

# COMMAND ----------

# MAGIC %md ## Eligible pool
# MAGIC Profiles the whole panel under the protocol and freezes the stratum-labelled
# MAGIC pool the ladder draws from. Never writes the frozen sample.

# COMMAND ----------

sh("scripts/build_scaling_pool.py", "--config", CONFIG,
   "--out", f"{SCALING}/eligible_pool.csv")

# COMMAND ----------

# MAGIC %md ## Nested samples and per-rung configs
# MAGIC Writes `sample_series_<N>.csv` for every rung but the anchor, each a superset
# MAGIC of the rung below, plus one config per rung that differs from the base in
# MAGIC `n_series` and `sample_path` only. The subset assertions run inside.

# COMMAND ----------

sh("scripts/build_scaling_ladder.py", "--config", CONFIG,
   "--pool", f"{SCALING}/eligible_pool.csv",
   "--sizes", ",".join(str(n) for n in RUNGS),
   "--out-dir", SCALING, "--config-dir", f"{SCALING}/configs")

# The ladder was first drawn on a laptop from the same export and its series
# sets are recorded in the committed manifest; a mismatch means the two draws
# diverged and the local numbers no longer describe these samples.
here = json.load(open(f"{SCALING}/configs/manifest.json"))["rungs"]
committed = json.load(open(f"{REPO}/configs/scaling/manifest.json"))["rungs"]
print("\nseries sets against the committed manifest:")
for n in RUNGS:
    a = here.get(str(n), {}).get("ids_digest")
    b = committed.get(str(n), {}).get("ids_digest")
    print(f"  N={n:<6,} cluster {a}  committed {b}  {'ok' if a == b else 'DIFFERENT'}")

# COMMAND ----------

# MAGIC %md ## Tuned parameters: one search, every rung
# MAGIC `load_best` reads `results/<run.name>/tuning_best.json`, so each rung gets a
# MAGIC copy of the N=1,000 search. Tuning once and applying everywhere is what keeps
# MAGIC the result keys - and the comparison - the same across the ladder.

# COMMAND ----------

source = f"{PAPER_RUN}/tuning_best.json"
if not os.path.exists(source):
    raise FileNotFoundError(f"{source} is missing: the tuned N=1,000 run is the anchor "
                            "of the ladder and has to exist first")
for n in RUNGS:
    target_dir = f"{VOLUME}/results/{rung_run(n)}"
    os.makedirs(target_dir, exist_ok=True)
    target = f"{target_dir}/tuning_best.json"
    if os.path.exists(target):
        print(f"present  {target}")
    else:
        shutil.copyfile(source, target)
        print(f"copied   {target}")

# COMMAND ----------

# MAGIC %md ## Run the ladder
# MAGIC Smallest rung first: at 500 series it doubles as the smoke test for the
# MAGIC whole path before the longer rungs start. Roughly 18 minutes of measured
# MAGIC compute over the four rungs, plus panel loading per rung.

# COMMAND ----------

for n in RUNGS:
    print(f"\n{'=' * 78}\nN = {n:,}   run {rung_run(n)}\n{'=' * 78}")
    sh("scripts/run_benchmark.py", "--config", rung_config(n),
       "--run-name", rung_run(n), "--models", MODELS)

# COMMAND ----------

# MAGIC %md ## Status per rung (runs nothing)

# COMMAND ----------

for n in RUNGS:
    print(f"\nN = {n:,}")
    sh("scripts/run_benchmark.py", "--config", rung_config(n),
       "--run-name", rung_run(n), "--models", MODELS, "--status")

# COMMAND ----------

# MAGIC %md ## Recorded failures (runs nothing)

# COMMAND ----------

for n in RUNGS:
    print(f"\nN = {n:,}")
    sh("scripts/run_benchmark.py", "--config", rung_config(n),
       "--run-name", rung_run(n), "--models", MODELS, "--errors")

# COMMAND ----------

# MAGIC %md ## T9, F9 and the crossover summary
# MAGIC Reads the finished rungs only, so it is safe on a partial ladder. Writes
# MAGIC `results/scaling/`: the table as booktabs LaTeX plus CSV, the figure as
# MAGIC vector PDF and 300dpi PNG, and `scaling_summary.json`.

# COMMAND ----------

sh("scripts/build_scaling_report.py", "--results-dir", f"{VOLUME}/results",
   "--reference", PAPER_RUN, "--out", f"{VOLUME}/results/scaling")
