from __future__ import annotations

import importlib.util
import random


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    if importlib.util.find_spec("numpy"):
        import numpy

        numpy.random.seed(seed)
    if importlib.util.find_spec("torch"):
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
