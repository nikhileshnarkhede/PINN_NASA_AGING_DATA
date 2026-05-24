"""
utils/seed.py
=============
Reproducibility helper -- seeds every random number generator the project
touches (Python `random`, NumPy, and PyTorch CPU + CUDA) from a single call.

The trainer calls set_seed(config.SEED) once at startup so a run can be
reproduced exactly.
"""
import logging
import random

import numpy as np
import torch

log = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed all random number generators used in the project.

    Args:
        seed:          The integer seed applied to `random`, NumPy and PyTorch
                       (CPU and all CUDA devices).
        deterministic: If True, also force cuDNN into deterministic mode. This
                       makes GPU runs bit-for-bit reproducible at some cost to
                       speed; leave False for normal training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN deterministic mode + disable the autotuner.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Force deterministic algorithm selection across all ops. warn_only=True
        # keeps training alive if an op (e.g. some cuDNN RNN kernels) has no
        # deterministic implementation -- it warns instead of raising.
        torch.use_deterministic_algorithms(True, warn_only=True)

    log.info("seed set to %d (deterministic=%s)", seed, deterministic)
