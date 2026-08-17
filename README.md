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
the no-op fit, quantile assembly and shape validation all have tests. What is
unverified is one `_forecast()` method per model, which calls a library whose
weights are not available here. Expect to fix those five methods on first
contact with the GPU box; nothing else should need to move.

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

**Consequence:** the panel *is* intermittent. 700 of the eligible series (9.8%)
have zero weeks; under the naive reading only 5 did. On the frozen sample the
Syntetos-Boylan classification is 83.4% smooth, 14.1% erratic, 2.4% lumpy,
0.1% intermittent, with 25 series above the Croston-relevant ADI ≥ 1.32 and 14
above 50% zero weeks. That answers the roster's open question with evidence:
Croston/SBA/TSB stay excluded, but say so on these numbers rather than by
assumption.

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
actually cost. Neural models set `horizon_is_fit_time`, because the horizon is
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

Checkpoints are keyed by config hash: a protocol change cannot inherit the
previous run's completed work. MLflow is logged when available; Parquet is the
system of record.

## Environment notes

- `numpy` is pinned `<2.5`: pandas 2.3 relies on a timedelta unit numpy 2.5
  deprecates, and every `Timedelta` raises otherwise.
- `lightgbm` needs `libomp`, and must not share a process with torch (above).
- This machine has no CUDA. Accuracy is device-independent, but the cost and
  scaling numbers in §5–6 must come from one pinned GPU instance, recorded in
  `cost.usd_per_hour` / `cost.instance_type` / `cost.runtime_version`.

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
