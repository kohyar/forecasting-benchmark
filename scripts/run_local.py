"""Run only the baselines + local models tier - a thin alias for
`run_benchmark.py --family baseline,local`. All other flags pass through
(--n-series, --force, --status, --errors, --tune, ...). Checkpoints are shared
with every other runner, so tiers can be completed one at a time and a full
`--collect-only` at the end assembles everything.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.execv(sys.executable, [sys.executable, "-u", os.path.join(HERE, "run_benchmark.py"),
                          "--family", "baseline,local", *sys.argv[1:]])
