"""Run the benchmark.

    python scripts/run_benchmark.py --models naive,seasonal_naive --n-series 50

Every override lands in the config before the hash is taken, so results stay
reproducible from config + seed alone.
"""
import argparse
import sys
from pathlib import Path

import yaml

from tsbench.config import Config
from tsbench.data.loader import load_panel
from tsbench.models import registry as registry_module
from tsbench.models.registry import partition_available
from tsbench.pipeline import ensure_sample
from tsbench.runner import BenchmarkRunner, collect
from tsbench.tuning import tune_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--models", help="comma-separated; defaults to models.enabled")
    ap.add_argument("--n-series", type=int, help="override sampling.n_series")
    ap.add_argument("--run-name", help="override run.name")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--tune", action="store_true",
                    help="search hyperparameters first, on data before fold 0")
    ap.add_argument("--keep-failed", action="store_true",
                    help="do not retry (model, fold) checkpoints that failed last time")
    ap.add_argument("--collect-only", action="store_true",
                    help="assemble results from existing checkpoints; run nothing")
    ap.add_argument("--status", action="store_true",
                    help="print checkpoint progress per model and exit")
    ap.add_argument("--strict", action="store_true",
                    help="exit if any requested model is unavailable instead of skipping it")
    ap.add_argument("--errors", action="store_true",
                    help="print the recorded traceback tail for each failed model and exit")
    ap.add_argument("--force", action="store_true",
                    help="discard existing checkpoints of the requested models and rerun them")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--param", action="append", default=[],
                    metavar="MODEL:KEY=VALUE",
                    help="override a model parameter, e.g. nhits:max_steps=20")
    args = ap.parse_args()

    with open(args.config) as fh:
        raw = yaml.safe_load(fh)

    if args.n_series:
        raw["sampling"]["n_series"] = args.n_series
        raw["sampling"]["sample_path"] = str(
            Path(raw["sampling"]["sample_path"]).with_name(f"sample_series_{args.n_series}.csv"))
        raw["run"]["name"] = args.run_name or f"{raw['run']['name']}-n{args.n_series}"
    elif args.run_name:
        raw["run"]["name"] = args.run_name
    if args.models:
        raw["models"]["enabled"] = [m.strip() for m in args.models.split(",")]
    for override in args.param:
        model, _, assignment = override.partition(":")
        key, _, value = assignment.partition("=")
        raw["models"].setdefault("params", {}).setdefault(model, {})[key] = yaml.safe_load(value)

    cfg = Config.from_dict(raw)
    registry = registry_module.default()

    if args.list_models:
        for status in registry.report():
            mark = "ok " if status["available"] else "-- "
            print(f"{mark}{status['name']:20s} {status['family']:11s} "
                  f"{status['disabled_reason']}")
        return

    try:
        runnable, skipped = partition_available(registry, cfg.models.enabled)
    except KeyError as exc:
        sys.exit(str(exc))
    for name, reason in skipped:
        print(f"SKIPPING {name}: {reason}")
    if skipped and args.strict:
        sys.exit(f"{len(skipped)} model(s) unavailable and --strict was given")
    if not runnable:
        sys.exit("no runnable models")
    raw["models"]["enabled"] = runnable
    cfg = Config.from_dict(raw)

    if args.status:
        _print_progress(BenchmarkRunner(cfg, registry=registry, mlflow_enabled=False),
                        cfg.models.enabled, cfg.models.params)
        return

    if args.errors:
        _print_errors(BenchmarkRunner(cfg, registry=registry, mlflow_enabled=False,
                                      retry_failed=False),
                      cfg.models.enabled, cfg.models.params)
        return

    if args.collect_only:
        result = collect(cfg, models=cfg.models.enabled, registry=registry,
                         tuned_params=cfg.models.params)
        print("assembled from checkpoints (nothing was run)")
        _summarise(result)
        return

    print(f"config {args.config}  hash={cfg.hash}  seed={cfg.run.seed}  "
          f"repeats={cfg.run.n_repeats}")
    panel = load_panel(cfg)
    sample, meta = ensure_sample(panel, cfg)
    panel = panel[panel["unique_id"].isin(sample["unique_id"])].reset_index(drop=True)
    print(f"sample: {len(sample):,} series ({'built' if meta['built'] else 'frozen'}) "
          f"-> {cfg.sampling.sample_path}")
    print(f"panel:  {len(panel):,} rows  {panel['ds'].min().date()} .. {panel['ds'].max().date()}")

    runner = BenchmarkRunner(cfg, registry=registry, mlflow_enabled=not args.no_mlflow,
                             retry_failed=not args.keep_failed,
                             force=cfg.models.enabled if args.force else ())
    print(f"device: {runner.device}  n_jobs={cfg.run.n_jobs}")
    print(f"models: {', '.join(cfg.models.enabled)}")
    _print_progress(runner, cfg.models.enabled, cfg.models.params)
    print()

    params = dict(cfg.models.params)
    if args.tune:
        print(f"tuning: {cfg.tuning.budget_trials} trials per tunable model, "
              f"0 for zero-shot")
        tuned = tune_all(panel, cfg, cfg.models.enabled, registry=registry)
        for name in cfg.models.enabled:
            print(f"  {name:18s} budget={tuned.budget[name]:3d}  "
                  f"best={tuned.best[name] or '-'}")
        params = {name: {**params.get(name, {}), **tuned.best.get(name, {})}
                  for name in cfg.models.enabled}
        print(f"  trials -> {tuned.path}\n")

    result = runner.run(panel, tuned_params=params)
    _summarise(result)
    for name, reason in skipped:
        print(f"NOTE: {name} was skipped ({reason})")


def _print_progress(runner, models, params) -> None:
    progress = runner.progress(models, params)
    done = sum(p["complete"] for p in progress.values())
    total = sum(p["expected"] for p in progress.values())
    print(f"checkpoints: {done}/{total} (model, fold) units complete "
          f"under {runner.checkpoint_root}")
    for name, p in progress.items():
        state = ("done" if p["complete"] == p["expected"]
                 else f"{p['complete']}/{p['expected']}"
                      + (f", {p['failed']} failed" if p["failed"] else ""))
        print(f"  {name:18s} {state}")


def _print_errors(runner, models, params, tail_lines: int = 25) -> None:
    """One recorded traceback per failed model, from the checkpoints on disk."""
    import pandas as pd

    shown = 0
    for name in models:
        for path in runner.checkpoint_files(name):
            if not path.name.endswith("__timings.parquet"):
                continue
            try:
                t = pd.read_parquet(path)
            except Exception:
                continue
            failed = t[t["status"] == "failed"]
            if failed.empty:
                continue
            row = failed.iloc[0]
            print(f"{'=' * 78}\n{name}  fold {row['fold']}  stage {row['stage']}\n{row['error']}\n")
            tb = str(row.get("traceback") or "")
            print("\n".join(tb.strip().splitlines()[-tail_lines:]))
            shown += 1
            break
    print(f"{'=' * 78}\n{shown} model(s) with recorded failures" if shown
          else "no recorded failures")


def _summarise(result) -> None:
    t = result.timings
    ok, failed = t[t["status"] == "ok"], t[t["status"] == "failed"]

    if len(failed):
        print(f"\n{len(failed)} failed (model, fold, horizon) combination(s):")
        for _, row in failed.drop_duplicates(["model", "stage"]).iterrows():
            print(f"  {row['model']:20s} {row['stage']:8s} {row['error'][:200]}")
        print("  (run with --errors for the recorded tracebacks)")

    if len(ok):
        fits = ok[~ok["fit_reused"]].groupby("model")["fit_seconds"].median()
        preds = ok.groupby(["model", "horizon"])["predict_seconds"].median().unstack()
        print("\nmedian fit seconds (per fold):")
        print(fits.round(3).to_string())
        print("\nmedian predict seconds by horizon:")
        print(preds.round(3).to_string())

    m = result.metrics
    if len(m):
        for metric in ("MASE", "RMSSE", "CRPS", "coverage_80"):
            sub = m[m["metric"] == metric]
            if not len(sub):
                continue
            table = sub.groupby(["model", "horizon"])["value"].median().unstack()
            print(f"\nmedian {metric} by horizon:")
            print(table.round(4).to_string())

        undefined = m[m["metric"] == "sMAPE"]["n_undefined"].sum()
        print(f"\nsMAPE undefined points: {int(undefined)}")

    print(f"\nwrote {result.paths['metrics']}")
    print(f"      {result.paths['timings']}")
    print(f"      {result.paths['metadata']}")


if __name__ == "__main__":
    main()
