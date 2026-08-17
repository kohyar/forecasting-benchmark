"""Loading, eligibility and sampling wired together, shared by the CLIs."""
from pathlib import Path

import pandas as pd

from tsbench.config import Config
from tsbench.data.sampling import load_sample, save_sample, series_profile, stratified_sample
from tsbench.eval.splitter import RollingOriginSplitter


def eligible_profile(panel: pd.DataFrame, cfg: Config) -> tuple:
    """Series the protocol can actually run on, plus the drop counts the
    paper's data section reports."""
    splitter = RollingOriginSplitter(cfg)
    verdict = splitter.eligibility(panel)
    eligible = verdict[verdict["eligible"]]

    profile = series_profile(panel)
    profile = profile[profile["unique_id"].isin(eligible["unique_id"])].reset_index(drop=True)

    report = {
        "series_total": len(verdict),
        "dropped_below_min_train": int((~verdict["meets_min_train"]).sum()),
        "dropped_no_evaluation_coverage": int((~verdict["covers_evaluation_span"]).sum()),
        "eligible": len(eligible),
        "min_train_weeks_found_all": int(verdict["train_weeks_at_earliest_origin"].min()),
        "min_train_weeks_found_eligible": int(
            eligible["train_weeks_at_earliest_origin"].min()) if len(eligible) else 0,
        "origins": [str(o.date()) for o in splitter.origins(panel)],
    }
    return profile, report


def ensure_sample(panel: pd.DataFrame, cfg: Config) -> tuple:
    """Load the frozen sample, or build and freeze it if the config moved."""
    path = Path(cfg.sampling.sample_path)
    if path.exists():
        try:
            sample, meta = load_sample(path, expect=cfg)
            return sample, {**meta, "built": False}
        except ValueError:
            pass  # config changed: the frozen sample no longer describes this run

    profile, _ = eligible_profile(panel, cfg)
    sample = stratified_sample(profile, cfg)
    save_sample(sample, path, cfg)
    _, meta = load_sample(path, expect=cfg)
    return sample, {**meta, "built": True}
