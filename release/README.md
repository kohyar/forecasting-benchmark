# Released results

Everything the paper's tables, figures and tests are computed from, so that
the analysis can be re-run without access to the SPINS panel. The panel
itself is licensed and is not included; what is released is the identity of
the sampled series, the forecasts' scores, and the timings.

Layout mirrors `results/`, so the report scripts read it directly:

```bash
.venv/bin/python scripts/run_stats.py --run release/runs/spins-weekly-v1
.venv/bin/python scripts/build_paper_artifacts.py --run release/runs/spins-weekly-v1 \
    --sample release/samples/sample_series.csv
.venv/bin/python scripts/build_scaling_report.py --results-dir release/runs \
    --reference release/runs/spins-weekly-v1 --out release/scaling
.venv/bin/python scripts/build_appendix_tables.py --run release/runs/spins-weekly-v1 \
    --scaling release/scaling
```

## samples/

- `sample_series.csv` — the frozen 1,000-series sample of the main run: series
  identifier (`SUBCATEGORY@@MARKET`), stratum (volume decile, zero-share bin)
  and the profile columns the stratification used.
- `eligible_pool.csv` — the 7,125 series eligible under the protocol, from
  which the ladder draws.
- `scaling/sample_series_{500,2000,4000}.csv` — the nested ladder samples. The
  1,000-series rung is `sample_series.csv` itself.

## runs/

One directory per run, in the layout the harness writes:

- `metrics.parquet` — one row per (model, fold, horizon, series, metric,
  repeat): MASE, RMSSE, sMAPE, MAPE, MAE, RMSE, CRPS, pinball and coverage_80,
  with the count of undefined points beside each value.
- `timings.parquet` — one row per (model, fold, horizon, repeat): fit and
  predict seconds, whether the fit was reused from the other horizon, peak
  memory, parameter count, device and the code revision that produced it.
- `run_metadata.json` — the full configuration, package versions and hardware
  the run recorded.
- `tuning_best.json` — the hyperparameters the 20-trial search selected and
  the budget each model received.

`spins-weekly-v1` is the main run (17 models, 1,000 series). It also carries
`analysis/` (the clustered pairwise tests, the Friedman post-hocs and the
bootstrap skill intervals written by `run_stats.py`) and `paper/` (the CSV
form of every table in the paper). `spins-weekly-scaling-n{500,1000,2000,4000}`
are the four ladder rungs (Prophet, N-HiTS, DLinear, Chronos-2).

## scaling/

The ladder report: the fixed-evaluation-set table and its full-sample
counterpart, the crossover search, the paired tests at every rung, and the
summary JSON.

## Units and conventions

Scaled errors use a per-series seasonal (m = 52) denominator fixed before the
earliest origin. MAE, RMSE, CRPS and pinball are in the panel's own units,
weekly dollar sales. Timings are seconds on one Standard_NC8as_T4_v3 node; the
paper's compute-seconds per 1,000 series are the median fit plus median
predict, with the neural models charged their own fit at each horizon.
