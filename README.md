# tsbench — SPINS weekly forecasting benchmark

Reproducible harness for the NCA benchmark paper. Every number traces back to a
config hash, a seed, and a run.

## Status

All ten build steps are implemented and tested.

| Step | Module | Verified on real data |
|---|---|---|
| 1. Loading, sampling, schema, frequency | `data/` | yes |
| 2. Rolling-origin splitter + leakage tests | `eval/splitter.py` | yes |
| 3. Metrics + shared MASE denominator | `eval/metrics.py` | yes |
| 4. Adapter interface + registry | `models/base.py`, `models/registry.py` | yes |
| 5. Baselines and local models | `models/statsforecast_adapters.py`, `prophet_adapter.py`, `lightgbm_adapters.py` | yes |
| 6. Global models | `models/neural_adapters.py`, `lightgbm_adapters.py` | yes |
| 7. Foundation models | `models/foundation_adapters.py` | **no — needs the GPU runner** |
| 8. Tuning harness | `tuning.py` | yes |
| 9. Statistical testing | `stats/tests.py` | yes |
| 10. Results aggregation | `stats/aggregate.py` | yes |

The five foundation adapters share a tested base class — context construction,
the no-op fit, quantile assembly and shape validation all have tests — and each
`_forecast()` was written against the source of the installed library version
(chronos-forecasting 2.3.1, timesfm 2.0.2, toto-ts 0.2.0, tabpfn-time-series
1.2.0, granite-tsfm 0.3.8) and exercised against fakes that return those
libraries' documented shapes (`test_foundation_plumbing.py`). What remains
unverified is the run against real weights on the GPU box.

## Setup

```bash
uv venv && uv pip install -e ".[dev,models]"
brew install libomp                     # lightgbm's OpenMP runtime (macOS)

.venv/bin/python -m pytest -m "not slow"   # fast suite
.venv/bin/python -m pytest                 # everything, incl. real-data checks

.venv/bin/python scripts/build_sample.py                     # freeze the sample
.venv/bin/python scripts/run_benchmark.py --list-models
.venv/bin/python scripts/run_benchmark.py --n-series 50      # a smoke run
.venv/bin/python scripts/run_stats.py --run results/<name>   # tests and tables
.venv/bin/python scripts/profile_models.py --n-series 5      # cost probe
```

The SPINS export is licensed and is not in this repo. Point `data.path` at your
own copy; `build_sample.py` regenerates `data/` and `results/` locally, both
gitignored. Same export + config + seed reproduces the frozen sample exactly.

## Data decisions

Source: 3,288,786 rows, 14,942 series, 232 weekly periods (2022-01-09 …
2026-06-14, all Sundays, every gap a whole number of weeks).

**Scope: MARKET × CONVENTIONAL|MULTI OUTLET (7,473 series).** The export mixes
aggregation levels — CENSUS REGION geographies aggregate MARKET ones, and
within a market CONVENTIONAL|FOOD is a subset of MULO. A wider scope would put
parents and children in one sample and correlate series that Friedman and
Diebold-Mariano assume are independent.

**Target: `Dollars`.** `ARP` is realised average price and satisfies
`Dollars = Units × ARP`, so it is a component of the target — disclose the
circularity in the covariates subsection.

### Two kinds of missing week

This one changes results, so it is worth stating precisely.

A week the export **omits entirely** is a week with **no sales**, not an
unmeasured week. Evidence: sales in the week before a gap run at a median of
3.5% of that series' median (77% below 25%); gaps cluster in ISO weeks 49–52
and 16–17, matching out-of-season produce; and the export carries almost no
explicit zeros (331 rows across 20 series) — it drops the row instead. Those
weeks become observed zeros (`data.absent_rows: zero`), and they are scored.

A week **present with a null measure** is genuinely unknown. It stays NaN in
the panel, is never scored, and is filled only in the training frame.

`row_present` keeps the two distinguishable. Setting `absent_rows: missing`
restores the naive reading for a sensitivity check.

**Consequence:** the panel *is* intermittent. 700 of the eligible series have
zero weeks; under the naive reading only 5 did. On the frozen sample the
Syntetos-Boylan classification is 83.0% smooth, 14.7% erratic, 2.2% lumpy and
0.1% intermittent, with 100 series holding at least one zero week, 23 above the
Croston-relevant ADI ≥ 1.32, 30 above 20% zero weeks and 13 above 50%. That
answers the roster's open question with evidence: Croston/SBA/TSB stay
excluded, but say so on these numbers rather than by assumption.

### Training-frame preparation

Applied once, identically for every model, so the benchmark compares models and
not imputations:

| Case | Treatment | Reported as |
|---|---|---|
| Omitted week | observed zero (in the panel, before anything else) | — |
| Interior null | linear interpolation | `n_imputed` |
| Trailing null | last value carried forward | `n_carried_forward` |
| Leading null | left missing | `n_still_missing` |

Trailing nulls must be filled: libraries anchor a forecast on each series' last
observation, so a series stopping short of the origin silently forecasts the
wrong weeks. `assert_frame_reaches_origin` makes that loud.

### Covariate groups

| Group | Columns | Why |
|---|---|---|
| `static` | Department, Category, Subcategory, region, channel, geography_level | fixed per series |
| `past_observed` | Units, ARP, ARP_EQ_Units | realised only after the week closes |
| `known_future` | Dollars_Yago, Units_Yago, ARP_Yago, ARP_EQ_Units_Yago, ARP_planned, week_of_year | see below |

The `_Yago` columns are exact 52-week calendar lags (verified: 99.9995% match),
so they are known at any origin for h ≤ 52. They are **reconstructed from the
training window**, never read from the export's future rows — `test_leakage.py`
poisons those raw columns and asserts the reconstruction ignores them.

`ARP_planned` is the last price observed at the origin, carried forward. Future
ARP is unknowable and is a factor of the target, but *today's price, assumed to
hold* is exactly what a planner has, so this is the honest price signal for the
with/without-covariates ablation.

There is no promotion field and no posted price plan. A holiday calendar is the
obvious addition and has a hook in `Fold.future_covariates`.

### The ablation

`models.covariate_ablation: true` runs every covariate-capable model twice —
once with, once without — and stamps a `covariates` column on every metric and
timing row. Models that cannot use covariates run once. The two arms are
checkpointed apart, so a resumed run does not merge them.

`test_covariates.py` asserts the arms actually diverge: a model that declared
support but ignored its inputs would produce two identical arms and an ablation
that proves nothing. That test caught exactly that bug in the LightGBM adapters.
It needs more than a year of fixture history, because the `_Yago` covariates are
52-week lags and a shorter panel hands the model a column of zeros.

## Protocol

Rolling origin, expanding window, 5 folds, step 4 weeks, horizons 4 and 13.
Origins: 2025-11-23, 2025-12-21, 2026-01-18, 2026-02-15, 2026-03-15.

Step 4 < horizon 13, so test windows overlap and fold errors are correlated.
`Fold.metadata` carries `step_weeks` and `overlapping_test_windows` into the
output so the Harvey-Leybourne-Newbold correction can be justified in §4.

Eligibility (7,473 series):

| Outcome | Count |
|---|---|
| below `min_train_weeks` = 104 at the earliest origin | 60 |
| no observation covering the evaluation span | 305 |
| zero seasonal denominator (unscoreable) | 55 |
| **eligible** | **7,139** |

Minimum training window among eligible series: **111 weeks**. The zero-
denominator filter is not a curiosity — runs of zero-sales weeks make a flat
52-week difference reachable, and such a series cannot carry MASE or RMSSE at
all.

## Sample

1,000 series, stratified by volume decile × zero-share bin, frozen to
`data/sample_series.csv` with a provenance header (seed, config hash, scope,
bin edges). Proportional allocation with largest-remainder rounding, so every
stratum is represented to within one series rather than by chance. Exactly ~100
per volume decile; 100 intermittent series; volumes span 1,207 to 1.06e9.

## Measurement

Per (model, fold, horizon, repeat): `fit_seconds`, `predict_seconds`,
`peak_memory_mb`, `n_series`, `n_params`, `device`, `gpu_model`, `cpu_count`,
`n_jobs`, `tuning_trials`, `seed`, `config_hash`.

Fit and predict are never summed. A fit shared across horizons is recorded once
under a `fit_key` and flagged `fit_reused`, so cost totals reflect what the run
actually cost.

For zero-shot models `fit_seconds` is **checkpoint loading**, not training — the
adapters load weights in `fit()` (TabPFN additionally runs a one-series warm-up,
because it loads lazily) so that `predict_seconds` is pure inference. The
asymmetry the paper reports is therefore "seconds of setup vs minutes-to-hours
of training", and the inference column is never inflated by disk I/O. The
notebook runs a 5-series warm-up first so no timed fold includes a download. Neural models set `horizon_is_fit_time`, because the horizon is
baked into the architecture, and are refitted per horizon.

`measure()` calls `synchronize(device)` before stopping the clock —
`torch.cuda.synchronize` on CUDA, `torch.mps.synchronize` on MPS, nothing on
CPU — so a GPU timing measures compute rather than async launch.

## Execution isolation

Each (model, fold) runs in its own process (`run.execution: subprocess`). Two
reasons, and the first is not optional:

1. **lightgbm and torch cannot share a process here.** Both bundle an OpenMP
   runtime; on macOS arm64, whichever loads second corrupts the first, and the
   process segfaults (exit 139) inside whichever library runs. Confirmed both
   ways. Adapter modules therefore register lazily — `tsbench/models/__init__.py`
   is deliberately empty — so a lightgbm worker never imports torch. Even the
   device resolution and seeding avoid importing torch for this reason.
2. **A segfault or an OOM kill cannot be caught by `try/except`.** Watching a
   child exit can. `test_worker.py` kills a worker with `os._exit(139)` and
   asserts the run records the failure and continues.

Set `execution: inprocess` for debugging.

## Tuning

Equal budget: `tuning.budget_trials` Optuna trials for every tunable model,
0 for zero-shot models, recorded on every output row either way.

The search runs entirely before the first forecast origin. `tuning_origin()`
places an inner origin one long horizon earlier, so the window used to score a
trial is itself training data for fold 0 and no test observation is ever
consulted. A trial that raises is recorded as failed and the search continues.
Trials are written to `tuning_trials.parquet`, the winner to `tuning_best.json`.

## Metrics

Point: MASE and RMSSE (primary), MAE and RMSE (secondary), sMAPE and MAPE for
comparability only. Probabilistic: pinball, CRPS, empirical coverage.

MASE and RMSSE denominators come from the in-sample seasonal naive error at
`season_length = 52`, computed **once per series** from data strictly before
the earliest origin — in-sample for every fold and identical across folds and
models. `evaluate()` takes them as an argument and never recomputes them;
`test_mase_uses_the_denominator_it_is_given` halves them and asserts MASE
doubles.

CRPS is `2 × mean pinball` over a symmetric quantile grid, which collapses to
MAE for a degenerate forecast. Quantile crossing is repaired by sorting, in
every adapter that emits quantiles.

sMAPE and MAPE report `n_undefined` per series instead of dropping points
silently.

## Statistical testing

Separate module, run over a finished results file:

- Diebold-Mariano with the Harvey-Leybourne-Newbold small-sample correction,
  autocovariances to h−1, pairwise
- Friedman omnibus with Nemenyi post-hoc and a critical-difference diagram
- Wilcoxon signed-rank with Holm adjustment as the second post-hoc
- Percentile bootstrap CIs on skill scores

Both post-hocs always run. The Nemenyi mean-rank procedure is unstable when
models are added or removed (Benavoli et al. 2016), so `posthoc_comparison`
reports an explicit `agree` flag and `run_stats.py` prints every disagreement
rather than picking the friendlier answer.

## Output

`results/<run>/`:

| File | Contents |
|---|---|
| `metrics.parquet` | one row per (model, fold, horizon, series, metric) |
| `timings.parquet` | one row per (model, fold, horizon, repeat) |
| `run_metadata.json` | git commit, package versions, hardware, timestamp, config hash, protocol, imputation counts |
| `tuning_trials.parquet`, `tuning_best.json` | every trial and the winner |
| `analysis/` | accuracy and cost tables, post-hoc results, CD diagrams |
| `checkpoints/<config-hash>/` | per (model, fold), so a crash resumes |

MLflow is logged when available; Parquet is the system of record.

## Resuming

Built for long runs on rented hardware: a crash, a killed cell, a cluster
restart or a broken model costs only the unfinished (model, fold) units.

Layout: `checkpoints/<model>/<result_key>/fold<N>[__cov]__{metrics,timings}.parquet`.

**`result_key` covers only what determines that model's results** — data
identity, sampling, protocol, metrics, seed, repeats, tuning budget, device,
`n_jobs`, and *that model's own* parameters. It excludes file paths, the run
name, cost metadata, the enabled list and every other model's parameters. So:

| You change… | What reruns |
|---|---|
| TFT's batch size | TFT only |
| the enabled list | only models with no checkpoint |
| `data.path`, `output_dir`, `run.name` (moving to a cluster) | nothing — same keys |
| protocol, seed, season length | everything, by design |

`configs/default.yaml` and `configs/databricks-t4.yaml` produce identical
result keys and sample keys — checkpoints and the frozen sample are
interchangeable between machines. The full `config_hash` still differs and is
recorded on every row for provenance; `result_key` and `git_commit` are too.

**Failures are retried by default.** A checkpoint containing any failed row is
discarded on the next run and that (model, fold) runs again — so after fixing
an adapter you just re-run; nothing is deleted by hand. `--keep-failed` turns
that off for a model that fails for a reason you accept. If you *also* change
the model's parameters, it gets a new key and the old failure stays on disk as
a record.

**Writes are atomic** (temp file + rename), metrics before timings, so a kill
mid-write cannot leave a half-file under the real name; a corrupt checkpoint is
treated as absent, not as a crash. **Aggregate results are refreshed after
every model**, so `metrics.parquet` on disk is never more than one model behind.

```bash
python scripts/run_local.py      --config configs/databricks-t4.yaml   # baselines + local tier
python scripts/run_global.py     --config configs/databricks-t4.yaml   # global tier
python scripts/run_foundation.py --config configs/databricks-t4.yaml   # foundation tier
python scripts/run_benchmark.py  --config ... --status                 # progress, runs nothing
python scripts/run_benchmark.py  --config ... --collect-only           # assemble all tiers
```

The three tier runners are aliases for `run_benchmark.py --family ...` and
accept all its flags. Checkpoints are shared, so tiers can be completed in any
order — or on different clusters in parallel — and `--collect-only` assembles
the combined results.

The frozen sample is validated by `sample_key` (data identity + sampling +
protocol + seed), so it survives a move between machines and rebuilds only when
something that could change eligibility changes.

## Databricks

`configs/databricks-t4.yaml` plus the `databricks/` notebooks are the
cluster-side setup for a single-node `Standard_NC8as_T4_v3` on 17.3 LTS ML (GPU):

| Notebook | Purpose |
|---|---|
| `01_install` | installs everything into a **shared** `/local_disk0/tsbench-libs` (pip `--target`), environment check — once per cluster start. `%pip` is notebook-scoped on Databricks and would be invisible to the other notebooks, so installs must not use it |
| `00_common` | puts the shared libs + repo src on PYTHONPATH, paths, `sh()` — included by the others via `%run`; fails fast with "run 01_install" if the libs are missing |
| `10_run_local` | baselines + local tier (smoke, real run, status, errors) |
| `20_run_global` | global tier |
| `30_run_foundation` | foundation tier (incl. weight-cache warm-up) |
| `40_assemble` | combine all tiers' checkpoints, statistics, CD diagrams |

Tiers share checkpoints, so notebooks run in any order, independently, or on
two clusters in parallel (e.g. `30` on a second cluster while `10`/`20` run on
the first).
Fill in the Volume path and the `cost.usd_per_hour` (VM rate + 1.5 DBU/h ×
your DBU rate). Three rules that the notebook encodes:

- **`results/` and the sample live on a Volume**, never on cluster-local disk —
  the cluster auto-terminates and local disk goes with it. Scratch
  (`run.work_dir`) stays on `/local_disk0`, because a Volume is a network mount.
- **Keep the notebook cell running** — a running command is what stops
  auto-termination; a background process from the web terminal is not.
- **torch is pinned to 2.10.0, deliberately.** DBR 17.3 LTS ML ships 2.7.0,
  but `granite-tsfm 0.3.8` (TTM) requires `torch>=2.10,<2.11` and
  `neuralforecast>=3.2` requires `>=2.9.1`; the older releases that accept 2.7
  lose TTM's variant selection and hard-pin transformers. The notebook installs
  the 2.10.0 CUDA-12.8 wheel first, under a constraints file passed to every
  later install, so nothing can move it — pip has been observed backtracking
  toward an older torch while resolving the foundation packages. The
  reproducibility statement should say "torch 2.10.0 (pip) on 17.3 LTS ML",
  which is what `run_metadata.json` records.
- **torchvision moves with torch.** transformers imports torchvision when it
  is present; the runtime's torchvision is built for torch 2.7, so after the
  torch upgrade every transformers model dies with `operator torchvision::nms
  does not exist`, wrapped in a misleading "Could not import PreTrainedModel".
  The notebook pins `torchvision==0.25.0` next to `torch==2.10.0` and an
  environment-check cell touches `torchvision.ops.nms` before any run.
- **Toto is loaded without its hub mixin.** Its `_from_pretrained` targets
  huggingface_hub < 1.0; the adapter fetches the snapshot and calls Toto's own
  `load_from_checkpoint(directory)`.
- **TabPFN-TS runs on the TabPFN-v2 regressor weights** (`tabpfn-v2-regressor.ckpt`,
  public and ungated on HuggingFace) — the backbone the cited TabPFN-TS paper
  used. The package's own default is a v3 time-series checkpoint behind a
  licence gate that needs a verified PriorLabs account and `TABPFN_TOKEN`; pass
  `--param tabpfn_ts:checkpoint=null` to use it once that works. The checkpoint
  actually used is recorded on every timing row.
- **Toto is installed without its declared dependencies.** `toto-ts` pins
  `datasets==2.17.1` for its evaluation code, which `tabpfn-time-series`
  cannot accept; inference does not need it, so the notebook installs Toto
  with `--no-deps` and adds its runtime dependencies by hand.

The T4 has no bf16; `preferred_dtype()` falls Chronos-2 back to float32 there.
Run order in the notebook: smoke test on two baselines, then the five
foundation adapters one at a time (each failure is recorded and retried after
the fix), then the full run — re-run that cell as many times as it takes.

## Environment notes

- `numpy` is pinned `<2.5`: pandas 2.3 relies on a timedelta unit numpy 2.5
  deprecates, and every `Timedelta` raises otherwise.
- `lightgbm` needs `libomp`, and must not share a process with torch (above).
- This machine has no CUDA. Accuracy is device-independent, but the cost and
  scaling numbers in §5–6 must come from one pinned GPU instance, recorded in
  `cost.usd_per_hour` / `cost.instance_type` / `cost.runtime_version`.

## Validation run

A 50-series smoke run over ten models (`results/roster-smoke`), neural models
capped at 40 steps so it finishes on a laptop — accuracy here is not a result,
the point is that the pipeline is sound end to end.

| model | MASE h=4 | MASE h=13 | fit s | predict s |
|---|---:|---:|---:|---:|
| autotheta | 0.632 | 0.762 | 64.4 | 61.1 |
| autoets | 0.653 | 0.906 | 72.6 | 59.0 |
| nhits | 0.666 | 0.766 | 343.0 | 36.4 |
| naive | 0.689 | 0.885 | 63.8 | 56.4 |
| lightgbm_local | 0.717 | 0.837 | 209.8 | 39.7 |
| lightgbm_global | 0.801 | 1.009 | 51.5 | 0.8 |
| prophet | 0.864 | 0.878 | 54.9 | 22.7 |
| seasonal_naive | 0.920 | 0.953 | 61.3 | 62.6 |
| dlinear | 1.001 | 0.983 | 390.7 | 25.7 |
| patchtst | 1.262 | 1.425 | 352.1 | 118.1 |

Three things worth noting, all of which the harness is built to surface:

- **SeasonalNaive lands at MASE ≈ 0.92–0.95.** It should sit near 1 by
  construction, since the denominator is its own in-sample error. That it does
  is the strongest single check that splitter, denominator and metrics agree.
- **LightGBM-local (0.72) beats LightGBM-global (0.80)** at 50 series, while
  global is 4× cheaper to fit and 50× cheaper to predict. Same algorithm, same
  features, same budget — scope is the only variable. Finding where that
  crossover flips as cardinality grows is the paper's first claim.
- **The two post-hocs disagree** on 5 pairs at h=4 and 2 at h=13. Exactly the
  Nemenyi instability Benavoli et al. describe, and the reason both are run.

## Adding a model

Write an adapter, register it, done — the runner never learns its name.

```python
@register
class MyAdapter(ModelAdapter):
    name = "my_model"
    family = "global"          # baseline | local | global | foundation

    def fit(self, train_df): ...
    def predict(self, horizon): ...   # unique_id, ds, yhat [, yhat_q10..q90]
```

Then add the module to `registry.ADAPTER_MODULES`. TiRex is registered but
gated on its licence: flip `enabled=True` in `models/gated.py` and nothing else
changes.
