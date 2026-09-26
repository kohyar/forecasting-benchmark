"""Build the nested sample ladder the cardinality study runs on.

    python scripts/build_scaling_ladder.py [--sizes 500,1000,2000,4000]

Emits one frozen sample per rung, each a superset of the rung below it and
each carrying its own rung's sample_key, so the runner reads it as frozen
instead of rebuilding one over the top. The rung at the anchor's size is the
existing frozen sample and is never rewritten.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from tsbench.config import Config
from tsbench.data.sampling import load_sample, nested_ladder, save_sample

_META_PREFIX = "#tsbench "


def _ids_digest(rung: pd.DataFrame) -> str:
    """Identity of the series set alone, so a ladder rebuilt elsewhere (the
    cluster draws from its own copy of the export) can be checked against
    this one without comparing files."""
    joined = "\n".join(sorted(rung["unique_id"].astype(str)))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def _pool(path: Path) -> pd.DataFrame:
    with open(path) as fh:
        head = fh.readline()
        if not head.startswith(_META_PREFIX):
            raise ValueError(f"{path} has no tsbench provenance header; "
                             "run scripts/build_scaling_pool.py first")
        return json.loads(head[len(_META_PREFIX):]), pd.read_csv(
            fh, parse_dates=["first_ds", "last_ds"])


def _rung_config(base: Config, n: int, path: str) -> Config:
    raw = base.to_dict()
    raw["sampling"]["n_series"] = n
    raw["sampling"]["sample_path"] = path
    return Config.from_dict(raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pool", default="data/eligible_pool.csv")
    ap.add_argument("--sizes", default="500,1000,2000,4000")
    ap.add_argument("--out-dir", default="data/scaling")
    ap.add_argument("--config-dir", default="configs/scaling")
    args = ap.parse_args()

    sizes = sorted(int(s) for s in args.sizes.split(","))
    cfg = Config.from_yaml(args.config)
    meta, pool = _pool(Path(args.pool))
    anchor, anchor_meta = load_sample(cfg.sampling.sample_path)
    print(f"pool   {len(pool):,} series, {pool['stratum'].nunique()} strata "
          f"(sample_key {meta['sample_key']})")
    print(f"anchor {len(anchor):,} series <- {cfg.sampling.sample_path}")

    rungs = nested_ladder(pool, anchor, sizes, seed=cfg.run.seed)

    # Nesting is the whole point of the ladder, so it is asserted, not assumed.
    print("\nnesting:")
    ordered = sorted(rungs)
    for lo, hi in zip(ordered, ordered[1:]):
        a, b = set(rungs[lo]["unique_id"]), set(rungs[hi]["unique_id"])
        if not a <= b:
            raise SystemExit(f"FAIL: S_{lo} is not a subset of S_{hi} "
                             f"({len(a - b)} series lost)")
        print(f"  S_{lo} ({len(a):,}) subset of S_{hi} ({len(b):,})  ok")

    anchor_n = len(anchor)
    if set(rungs[anchor_n]["unique_id"]) != set(anchor["unique_id"]):
        raise SystemExit(f"FAIL: the N={anchor_n} rung is not the frozen anchor")
    print(f"  S_{anchor_n} is exactly the frozen anchor  ok")

    out_dir, cfg_dir = Path(args.out_dir), Path(args.config_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}

    print("\nrungs:")
    for n in ordered:
        rung = rungs[n]
        if n == anchor_n:
            path = cfg.sampling.sample_path        # reuse; never rewritten
            rung_cfg = cfg
            built = "anchor (reused, not rewritten)"
        else:
            path = str(out_dir / f"sample_series_{n}.csv")
            rung_cfg = _rung_config(cfg, n, path)
            save_sample(rung, path, rung_cfg)
            built = "written"

        inter = int((rung["zero_share"] >= 0.05).sum())
        digest = _ids_digest(rung)
        print(f"  N={n:<5,} {built:<32} strata={rung['stratum'].nunique():<3} "
              f"intermittent={inter:<4} ({inter / n:.1%})  key={rung_cfg.sample_key}  "
              f"ids={digest}")
        manifest[n] = {
            "n_series": n,
            "sample_path": path,
            "sample_key": rung_cfg.sample_key,
            "config_hash": rung_cfg.hash,
            "ids_digest": digest,
            "strata": int(rung["stratum"].nunique()),
            "intermittent": inter,
        }

    # One config per rung: only n_series and sample_path move. sample_path is in
    # neither key, n_series is in both, so each rung gets its own checkpoint
    # namespace and the anchor rung resolves to work that is already done.
    raw = Path(args.config).read_text()
    for n in ordered:
        if n == anchor_n:
            continue
        text = raw.replace(f"n_series: {cfg.sampling.n_series}", f"n_series: {n}")
        text = text.replace(f"sample_path: {cfg.sampling.sample_path}",
                            f"sample_path: {manifest[n]['sample_path']}")
        (cfg_dir / f"n{n}.yaml").write_text(text)
        print(f"  config -> {cfg_dir / f'n{n}.yaml'}")

    (cfg_dir / "manifest.json").write_text(json.dumps({
        "seed": cfg.run.seed,
        "anchor": cfg.sampling.sample_path,
        "pool": args.pool,
        "pool_size": len(pool),
        "rungs": manifest,
    }, indent=2))
    print(f"\nmanifest -> {cfg_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
