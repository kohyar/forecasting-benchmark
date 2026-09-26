"""T9 and F9 from the cardinality ladder's finished rungs.

    python scripts/build_scaling_report.py --results-dir results \
        --reference results/spins-weekly-v1

Each rung is its own run directory (results/<prefix><N>/). Nothing is
recomputed - the medians come from the metrics each rung already wrote - so a
partial ladder reports whatever has finished. The reference run supplies the
level the curve is read against: the best local model's median at the paper's
series count.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from tsbench.report.scaling import (  # noqa: E402
    crossovers,
    ladder_warnings,
    load_ladder,
    noise_band,
    reference_levels,
    scaling_figure,
    scaling_table,
)
from tsbench.report.style import save_figure  # noqa: E402
from tsbench.report.tables import to_latex  # noqa: E402


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
    for note in ladder_warnings(rungs):
        print(f"  WARNING: {note}")

    meta = next(iter(rungs.values()))["meta"]
    horizons = tuple(meta["config"]["protocol"]["horizons"])

    reference, reference_metrics = {}, None
    if args.reference:
        reference_metrics = pd.read_parquet(Path(args.reference) / "metrics.parquet")
        reference = reference_levels(reference_metrics, args.reference_model,
                                     metric=args.metric, horizons=horizons)
        print(f"\nreference: {args.reference_model} median {args.metric} in {args.reference}: "
              + ", ".join(f"h{h} {v:.4f}" for h, v in reference.items()))

    table = scaling_table(rungs, metrics=(args.metric, "RMSSE"), horizons=horizons,
                          band=not args.no_band)
    table.to_csv(out_dir / "T9_scaling.csv")
    shown = [c for c in table.columns if not c.endswith(("_lo", "_hi"))
             and not c.startswith("run_s")]
    (out_dir / "T9_scaling.tex").write_text(to_latex(
        table[shown].reset_index().set_index("model"),
        caption="Median accuracy and compute per 1,000 series against the number "
                "of series, on nested samples of the same panel.",
        label="tab:scaling", float_format="%.3f", fit_width=True))
    print(f"\nT9 -> {out_dir / 'T9_scaling.tex'}  ({len(table)} rows)")
    print(table[shown].round(4).to_string())

    cross = pd.DataFrame()
    if reference:
        cross = crossovers(table, reference, metric=args.metric, horizons=horizons)
        cross.to_csv(out_dir / "crossover.csv")
        print(f"\ncrossover against {args.reference_model}:")
        for (model, h), row in cross.iterrows():
            where = (f"first below at N={int(row['first_n_below']):,}"
                     if pd.notna(row["first_n_below"]) else
                     f"no crossover up to N={int(row['max_n']):,}")
            print(f"  {model:14s} h{h:<3} {where:32s} gap at max N {row['gap_at_max_pct']:+.1f}%")

    bands = {h: noise_band(table, args.noise_model, args.metric, h) for h in horizons}
    print(f"\n{args.noise_model} spread across rungs (sampling noise): "
          + ", ".join(f"h{h} {b[0]:.4f}..{b[1]:.4f}" if b else f"h{h} n/a"
                      for h, b in bands.items()))

    # The paper's N=1,000 numbers were measured by an earlier build; a re-measured
    # rung at that size says whether the build moved accuracy or only timing.
    if reference_metrics is not None and 1000 in rungs:
        print("\nN=1,000 rung against the reference run's medians:")
        for model in sorted(rungs[1000]["metrics"]["model"].unique()):
            for h in horizons:
                here = table.loc[(model, 1000), f"{args.metric}_h{h}"]
                there = reference_levels(reference_metrics, model, metric=args.metric,
                                         horizons=(h,)).get(h)
                if there is None:
                    continue
                print(f"  {model:14s} h{h:<3} ladder {here:.4f}  paper {there:.4f}  "
                      f"{(here / there - 1) * 100:+.2f}%")

    fig = scaling_figure(table, horizons=horizons, metric=args.metric, reference=reference,
                         reference_label=f"{args.reference_model} at N=1,000 (paper run)",
                         noise_model=args.noise_model)
    written = save_figure(fig, out_dir / "F9_scaling")
    print(f"\nF9 -> {', '.join(str(p) for p in written)}")

    (out_dir / "scaling_summary.json").write_text(json.dumps({
        "rungs": {n: {"path": str(r["path"]), "git_commit": r["meta"].get("git_commit"),
                      "models": sorted(r["metrics"]["model"].unique())}
                  for n, r in rungs.items()},
        "metric": args.metric,
        "reference_model": args.reference_model if reference else None,
        "reference": reference,
        "noise_model": args.noise_model,
        "noise_band": bands,
        "crossover": json.loads(cross.reset_index().to_json(orient="records"))
        if len(cross) else [],
        "warnings": ladder_warnings(rungs),
    }, indent=2, default=str))
    print(f"summary -> {out_dir / 'scaling_summary.json'}")


if __name__ == "__main__":
    main()
