"""One place to seed every source of randomness a run touches.

torch is seeded only when it is already loaded - importing it here would drag
it into workers that must not have it.
"""
import os
import random
import sys

import numpy as np


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch = sys.modules.get("torch")
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
