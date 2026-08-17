"""Per-model cost probe: how long does one fold cost per series?

    python scripts/profile_models.py --n-series 5 --models autoets,prophet

Used to size the real run before committing cluster time.
"""
import argparse
import time
import warnings

import yaml

from tsbench.config import Config
from tsbench.data.loader import load_panel
from tsbench.data.prepare import impute_gaps
from tsbench.data.sampling import load_sample
from tsbench.eval.splitter import RollingOriginSplitter
from tsbench.models import registry as registry_module

DEFAULT = ("naive,seasonal_naive,autoets,autotheta,prophet,"
           "lightgbm_local,lightgbm_global,autoarima")


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-series", type=int, default=5)
    ap.add_argument("--models", default=DEFAULT)
    ap.add_argument("--horizon", type=int, default=13)
    args = ap.parse_args()

    cfg = Config.from_dict(yaml.safe_load(open(args.config)))
    panel = load_panel(cfg)
    sample, _ = load_sample(cfg.sampling.sample_path, expect=cfg)
    ids = sample["unique_id"].tolist()[:args.n_series]
    panel = panel[panel["unique_id"].isin(ids)]

    fold = next(iter(RollingOriginSplitter(cfg).split(panel)))
    train, _ = impute_gaps(fold.train)
    n = len(ids)
    print(f"{n} series, {len(train):,} training rows, origin {fold.origin.date()}, "
          f"h={args.horizon}, n_jobs={cfg.run.n_jobs}\n")

    reg = registry_module.default()
    for name in [m.strip() for m in args.models.split(",")]:
        model = reg.create(name, cfg)
        try:
            t0 = time.perf_counter()
            model.fit(train)
            fit = time.perf_counter() - t0

            t0 = time.perf_counter()
            pred = model.predict(args.horizon)
            predict = time.perf_counter() - t0

            q = "q" if any(c.startswith("yhat_q") for c in pred.columns) else "-"
            print(f"{name:18s} fit {fit:8.2f}s ({fit / n:7.3f}s/series)  "
                  f"predict {predict:7.2f}s  rows={len(pred):5d}  {q}")
        except Exception as exc:
            print(f"{name:18s} FAILED: {type(exc).__name__}: {str(exc)[:80]}")


if __name__ == "__main__":
    main()
