"""Fit and predict one model, in a process of its own.

    python -m tsbench.worker <workdir>

The parent hands over a spec, a training frame and the known-future frames;
the child writes forecasts and its own timings back. The child imports only
the adapter module it was asked for - loading every adapter would defeat the
purpose, since the isolation exists precisely because some of those libraries
cannot share a process.
"""
import importlib
import json
import sys
import traceback
from pathlib import Path

import pandas as pd

from tsbench.config import Config
from tsbench.measure import measure
from tsbench.seeding import set_seeds


def run(workdir: Path) -> int:
    spec = json.loads((workdir / "spec.json").read_text())
    cfg = Config.from_dict(spec["config"])
    device = spec["device"]  # already resolved by the parent

    # Import the adapter first, then seed: seeding only touches torch if the
    # adapter has loaded it, which keeps torch out of a lightgbm worker.
    module = importlib.import_module(spec["module"])
    adapter_cls = getattr(module, spec["class"])
    set_seeds(spec["seed"])

    train = pd.read_parquet(workdir / "train.parquet")
    model = adapter_cls(cfg, params=spec["params"], device=device)

    with measure(device) as fit_m:
        model.fit(train)

    result = {"horizons": {}, "fit_seconds": fit_m.seconds,
              "fit_peak_memory_mb": fit_m.peak_memory_mb}

    for horizon in spec["horizons"]:
        future_path = workdir / f"future_{horizon}.parquet"
        if future_path.exists():
            model.set_future_covariates(pd.read_parquet(future_path))

        with measure(device) as pred_m:
            pred = model.predict(horizon)

        pred.to_parquet(workdir / f"pred_{horizon}.parquet", index=False)
        result["horizons"][str(horizon)] = {
            "predict_seconds": pred_m.seconds,
            "predict_peak_memory_mb": pred_m.peak_memory_mb,
        }

    result["n_params"] = model.n_params
    result["n_series"] = int(train["unique_id"].nunique())
    result["device"] = device
    result["adapter_module"] = spec["module"]
    # Top-level imports the child actually pulled in - the evidence that a
    # lightgbm worker never loaded torch, and vice versa.
    result["modules"] = sorted(m for m in sys.modules
                               if "." not in m and not m.startswith("_"))
    (workdir / "result.json").write_text(json.dumps(result, default=str))
    return 0


def main() -> int:
    workdir = Path(sys.argv[1])
    try:
        return run(workdir)
    except Exception as exc:
        (workdir / "error.json").write_text(json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }))
        return 1


if __name__ == "__main__":
    sys.exit(main())
