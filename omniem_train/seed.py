"""Determinism — seed every RNG omniem-train touches.

From ``run.seed`` we seed:
  * Python ``random``
  * NumPy
  * Torch (CPU + CUDA)
  * MONAI ``set_determinism`` (data spine + augmentations)
The DataLoader receives a seeded ``torch.Generator`` and a ``worker_init_fn``
(see ``data.loader._worker_init_fn_factory``); their state is captured in the
trainer-state checkpoint so a resumed run reproduces the un-interrupted run's
stochastic aug.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> torch.Generator:
    """Seed every RNG; return a seeded ``torch.Generator`` for the DataLoader.

    Args:
        seed: the run seed (``cfg.run.seed``).

    Returns:
        A seeded ``torch.Generator`` to hand to the DataLoader.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MONAI determinism (spatial transforms, CacheDataset shuffle, etc.).
    try:
        from monai.utils import set_determinism

        set_determinism(seed)
    except Exception:  # pragma: no cover (monai always present in our env)
        pass
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
