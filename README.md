# tsbench — SPINS weekly forecasting benchmark

Reproducible harness for the NCA benchmark paper. Every number is traceable to
a config hash, a seed, and a run.

## Status

Steps 1–3 of the build order are complete and tested.

| Step | Module | State |
|---|---|---|
| 1. Loading, sampling, schema, frequency | `data/schema.py`, `data/loader.py`, `data/sampling.py` | done |
| 2. Rolling-origin splitter + leakage test | `eval/splitter.py` | done |
| 3. Metrics + MASE denominator caching | `eval/metrics.py` | done |
| 4–10. Adapters, models, tuning, stats, aggregation | — | not started |

## Setup

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest              # 87 unit tests
.venv/bin/python -m pytest -m slow      # 9 integration tests against the real export
.venv/bin/python scripts/build_sample.py
```

The SPINS export is licensed and is not in this repo. Point `data.path` in the
config at your own copy; `build_sample.py` then regenerates `data/` and
`results/` locally, both of which are gitignored. Given the same export, config
and seed, the frozen sample is reproduced exactly — that is what the provenance
header on the sample file is for.

## Data decisions

Source: `../dataset/New_Query_2026_06_01_10_17_28.csv` — 3,288,786 rows,
14,942 series, 232 weekly periods (2022-01-09 … 2026-06-14, all Sundays,
every gap a whole number of weeks).

**Scope: MARKET × CONVENTIONAL|MULTI OUTLET (7,473 series).** The export mixes
aggregation levels: CENSUS REGION geographies aggregate MARKET ones, and within
a market CONVENTIONAL|FOOD is a subset of MULO. Sampling across all of them
would put parents and children in the same benchmark and correlate series that
Friedman/Nemenyi and Diebold-Mariano assume are independent.

**Target: `Dollars`.** `ARP` is realised average price and satisfies
`Dollars = Units × ARP`, so it is a component of the target — disclose the
circularity in the covariates subsection.

**Covariate groups** (`data/schema.py::covariate_groups`):

| Group | Columns | Why |
|---|---|---|
| `static` | Department, Category, Subcategory, region, channel, geography_level | fixed per series |
| `past_observed` | Units, ARP, ARP_EQ_Units | realised only after the week closes |
| `known_future` | Dollars_Yago, Units_Yago, ARP_Yago, ARP_EQ_Units_Yago, week_of_year | exact 52-week calendar lags (verified: 99.9995% match), known at any origin for h ≤ 52 |

There is no promotion field and no posted price plan, so `known_future` is
limited to calendar lags and calendar features. A holiday calendar is the
obvious addition and has a hook in `Fold.future_covariates`.

The `_Yago` columns are **reconstructed from the training window**, never read
from the export's future rows — `test_leakage.py` poisons those raw columns and
asserts the reconstruction ignores them.

## Protocol

Rolling origin, expanding window, 5 folds, step 4 weeks, horizons 4 and 13.
Origins: 2025-11-23, 2025-12-21, 2026-01-18, 2026-02-15, 2026-03-15.

Step 4 < horizon 13, so test windows overlap and fold errors are correlated.
`Fold.metadata["step_weeks"]` and `["overlapping_test_windows"]` carry that into
the output so the Harvey-Leybourne-Newbold correction can be justified in §4.

Eligibility (7,473 series):

| Outcome | Count |
|---|---|
| below `min_train_weeks` = 104 at the earliest origin | 327 |
| no observation covering the evaluation span | 305 |
| **eligible** | **7,020** |

Minimum training window found among eligible series: **104 weeks** (exactly two
seasonal cycles); across all series, 0 — some start after the earliest origin.

## Sample

1,000 series, stratified by volume decile × zero-share bin, frozen to
`data/sample_series.csv` with a provenance header (seed, config hash, scope,
bin edges). Allocation is proportional with largest-remainder rounding, so
every stratum is represented to within one series rather than by chance.
Exactly 100 per volume decile; volumes span 11,348 to 1.03e9.

**Reportable finding:** this panel is not intermittent. Only 5 of the 7,020
eligible series have any zero week, so the zero-share stratum is vestigial and
one such series lands in the sample. That answers the roster's open question —
Croston/SBA/TSB stay excluded, on evidence rather than assumption.

## Season length

`season_length: 52` is set once in the config and threaded everywhere. Library
defaults assume daily or monthly data and would silently produce wrong results.
The `_Yago` reconstruction uses a fixed 52-week lag because that is what the
column means in the export, independent of the configured season.

## MASE / RMSSE denominator

Computed **once per series** from data strictly before the earliest origin, so
it is in-sample for every fold and identical across folds and models.
`evaluate()` takes denominators as an argument and never recomputes them;
`test_mase_uses_the_denominator_it_is_given` halves them and asserts MASE
doubles. Series with fewer than one seasonal cycle get `n_pairs = 0` and a NaN
denominator; flat series are flagged `zero_denominator` and scored NaN rather
than infinity. On the frozen sample, denominators rest on 42–151 seasonal pairs
and none are zero.

Metrics undefined near zero (sMAPE, MAPE) report `n_undefined` per series
instead of dropping points silently. CRPS is `2 × mean pinball` over a
symmetric quantile grid, which collapses to MAE for a degenerate forecast —
asserted in `test_crps_of_a_degenerate_forecast_equals_mae`.

## Next

Step 4: the `ModelAdapter` interface and registry, then Naive + SeasonalNaive
end to end on 50 series before any further adapters.
