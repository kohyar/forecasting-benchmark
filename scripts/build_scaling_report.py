"""T9 and F9 from the cardinality ladder's finished rungs.

    python scripts/build_scaling_report.py --results-dir results \
        --reference results/spins-weekly-v1

Each rung is its own run directory (results/<prefix><N>/). Nothing is
recomputed - the medians come from the metrics each rung already wrote - so a
partial ladder reports whatever has finished.

Two readings come out. The primary one scores every rung on the series all
rungs share (the smallest rung, with nesting): local and zero-shot models then
repeat themselves at every N, and a global model's movement is what more
training series bought on the same held-out series. The secondary one takes
each rung's full sample, where composition moves every model; the zero-shot
spread across rungs marks how much. The reference run supplies the level the
primary curve is read against: the best local model's median on those same
series.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from tsbench.report.scaling import (  # noqa: E402
    common_series,
    crossovers,
    ladder_warnings,
    load_ladder,
    noise_band,
    paired_tests,
    reference_levels,
    scaling_figure,
    scaling_table,
)
from tsbench.report.style import save_figure  # noqa: E402
from tsbench.report.tables import to_latex  # noqa: E402


def _emit_table(table: pd.DataFrame, out_dir: Path, stem: str, caption: str,
                label: str) -> None:
    table.to_csv(out_dir / f"{stem}.csv")
    shown = [c for c in table.columns if not c.endswith(("_lo", "_hi"))
             and not c.startswith("run_s") and c != "eval_series"]
    (out_dir / f"{stem}.tex").write_text(to_latex(
        table[shown].reset_index().set_index("model"), caption=caption, label=label,
        float_format="%.3f", fit_width=True))
    print(f"\n{stem} -> {out_dir / f'{stem}.tex'}  ({len(table)} rows)")
    print(table[shown].round(4).to_string())


def _print_crossovers(cross: pd.DataFrame, against: str) -> None:
    print(f"\ncrossover against {against}:")
    for (model, h), row in cross.iterrows():
        where = (f"first below at N={int(row['first_n_below']):,}"
                 if pd.notna(row["first_n_below"]) else
                 f"no crossover up to N={int(row['max_n']):,}")
        print(f"  {model:14s} h{h:<3} {where:32s} gap at max N {row['gap_at_max_pct']:+.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--prefix", default="spins-weekly-scaling-n",
                    help="rung directories are <results-dir>/<prefix><N>")
    ap.add_argument("--runs", nargs="*", help="explicit rung directories instead of the glob")
    ap.add_argument("--reference", help="the paper's run directory, for the reference level "
                                        "and the N=1,000 comparison")
    ap.add_argument("--reference-model", default="autoarima")
    ap.add_argument("--noise-model", default="chronos2",
                    help="zero-shot model whose spread across rungs marks sampling noise")
    ap.add_argument("--metric", default="MASE")
    ap.add_argument("--out", help="defaults to <results-dir>/scaling")
    ap.add_argument("--no-band", action="store_true", help="skip the bootstrap intervals")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    run_dirs = [Path(p) for p in args.runs] if args.runs else sorted(
        results_dir.glob(f"{args.prefix}*"))
    if not run_dirs:
        raise SystemExit(f"no rung directories under {results_dir} matching {args.prefix}*")
    out_dir = Path(args.out) if args.out else results_dir / "scaling"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("rungs:")
    rungs = load_ladder(run_dirs)
    if not rungs:
        raise SystemExit("no finished rung yet - every run directory lacks run_metadata.json")
    for n, rung in rungs.items():
        models = sorted(rung["metrics"]["model"].unique())
        commit = (rung["meta"].get("git_commit") or "")[:8]
        print(f"  N={n:<6,} {len(models)} models  build {commit or '?'}  <- {rung['path']}")
    warnings = ladder_warnings(rungs)
    for note in warnings:
        print(f"  WARNING: {note}")

    meta = next(iter(rungs.values()))["meta"]
    horizons = tuple(meta["config"]["protocol"]["horizons"])
    shared = common_series(rungs)
    print(f"\nfixed evaluation set: {len(shared):,} series scored by every rung")

    reference_metrics = None
    fixed_reference, full_reference = {}, {}
    if args.reference:
        reference_metrics = pd.read_parquet(Path(args.reference) / "metrics.parquet")
        fixed_reference = reference_levels(reference_metrics, args.reference_model,
                                           metric=args.metric, horizons=horizons,
                                           series=shared)
        full_reference = reference_levels(reference_metrics, args.reference_model,
                                          metric=args.metric, horizons=horizons)
        print(f"reference: {args.reference_model} median {args.metric} in {args.reference}")
        print("  on the fixed evaluation set: "
              + ", ".join(f"h{h} {v:.4f}" for h, v in fixed_reference.items()))
        print("  on its full sample:          "
              + ", ".join(f"h{h} {v:.4f}" for h, v in full_reference.items()))

    # --- primary: the same series at every N -------------------------------
    fixed = scaling_table(rungs, metrics=(args.metric, "RMSSE"), horizons=horizons,
                          band=not args.no_band, series=shared)
    _emit_table(fixed, out_dir, "T9_scaling",
                f"Median accuracy on the {len(shared):,} series every rung shares, "
                "against the number of series the models were run over, with the "
                "compute of one fit plus one predict per 1,000 series. Local and "
                "zero-shot models repeat themselves at every size; only the global "
                "models can move.", "tab:scaling")

    cross = pd.DataFrame()
    if fixed_reference:
        cross = crossovers(fixed, fixed_reference, metric=args.metric, horizons=horizons)
        cross.to_csv(out_dir / "crossover.csv")
        _print_crossovers(cross, f"{args.reference_model} on the same {len(shared):,} series")

    # Paired on the same series: the medians above can move by composition of
    # folds and repeats alone, so the per-series test is what supports "no change".
    paired = paired_tests(rungs, shared, metric=args.metric, horizons=horizons,
                          reference=reference_metrics, reference_model=args.reference_model)
    paired.to_csv(out_dir / "paired_tests.csv")
    print("\npaired Wilcoxon on the fixed evaluation set (negative diff favours the first side):")
    for (model, h, label), row in paired.iterrows():
        print(f"  {model:10s} h{h:<3} {label:24s} median {row['median_first']:.4f} vs "
              f"{row['median_second']:.4f}  paired diff {row['median_paired_diff']:+.4f}  "
              f"better on {row['share_first_better']:.0%} of series  p={row['p_value']:.3g}")

    fig = scaling_figure(fixed, horizons=horizons, metric=args.metric,
                         reference=fixed_reference,
                         reference_label=f"{args.reference_model} on the same series "
                                         "(paper run)",
                         noise_model=None, cost=True,
                         accuracy_label=f"median {args.metric}\n(same series at every N)")
    written = save_figure(fig, out_dir / "F9_scaling")
    print(f"\nF9 -> {', '.join(str(p) for p in written)}")

    # --- secondary: each rung's own sample -----------------------------------
    full = scaling_table(rungs, metrics=(args.metric, "RMSSE"), horizons=horizons,
                         band=not args.no_band)
    _emit_table(full, out_dir, "T9b_scaling_full_sample",
                "Median accuracy over each rung's full sample. Rungs are nested but "
                "not identical, so composition moves every model; the zero-shot "
                "model's spread across rungs is that effect alone.",
                "tab:scaling-full")
    bands = {h: noise_band(full, args.noise_model, args.metric, h) for h in horizons}
    print(f"\n{args.noise_model} spread across rungs on the full samples (sampling noise): "
          + ", ".join(f"h{h} {b[0]:.4f}..{b[1]:.4f} ({(b[1] / b[0] - 1) * 100:.1f}%)"
                      if b else f"h{h} n/a" for h, b in bands.items()))
    fig = scaling_figure(full, horizons=horizons, metric=args.metric, reference=None,
                         noise_model=args.noise_model, cost=False)
    written = save_figure(fig, out_dir / "F9b_scaling_full_sample")
    print(f"F9b -> {', '.join(str(p) for p in written)}")

    # The paper's N=1,000 numbers were measured earlier; the re-measured rung at
    # that size says whether the build moved accuracy or only timing.
    if reference_metrics is not None and 1000 in rungs:
        print("\nN=1,000 rung against the reference run's medians (full sample):")
        for model in sorted(rungs[1000]["metrics"]["model"].unique()):
            for h in horizons:
                here = full.loc[(model, 1000), f"{args.metric}_h{h}"]
                there = reference_levels(reference_metrics, model, metric=args.metric,
                                         horizons=(h,)).get(h)
                if there is None:
                    continue
                print(f"  {model:14s} h{h:<3} ladder {here:.4f}  paper {there:.4f}  "
                      f"{(here / there - 1) * 100:+.2f}%")

    (out_dir / "scaling_summary.json").write_text(json.dumps({
        "rungs": {n: {"path": str(r["path"]), "git_commit": r["meta"].get("git_commit"),
                      "models": sorted(r["metrics"]["model"].unique())}
                  for n, r in rungs.items()},
        "metric": args.metric,
        "fixed_evaluation_set": len(shared),
        "reference_model": args.reference_model if fixed_reference else None,
        "reference_fixed": fixed_reference,
        "reference_full": full_reference,
        "noise_model": args.noise_model,
        "noise_band_full_sample": bands,
        "crossover_fixed": json.loads(cross.reset_index().to_json(orient="records"))
        if len(cross) else [],
        "paired_tests_fixed": json.loads(paired.reset_index().to_json(orient="records")),
        "warnings": warnings,
    }, indent=2, default=str))
    print(f"summary -> {out_dir / 'scaling_summary.json'}")


if __name__ == "__main__":
    main()
