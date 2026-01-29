import logging
import os
import random
from typing import Optional

import numpy as np
import torch

from .computational.ddp import get_rank

logger = logging.getLogger(__name__)

MAX_SEED_VALUE = np.iinfo(np.uint32).max
MIN_SEED_VALUE = np.iinfo(np.uint32).min


def pl_worker_init_function(worker_id: int) -> None:
    """
    To handle seeding with multiprocessing.

    From https://pytorch-lightning.readthedocs.io/en/1.7.7/_modules/pytorch_lightning/utilities/seed.html#pl_worker_init_function.
    """
    # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    process_seed = torch.initial_seed()
    # back out the base seed so we can use all the bits (https://docs.pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading)
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([base_seed, worker_id, get_rank()])
    # use 128 bits (4 x 32-bit words)
    np.random.seed(ss.generate_state(4))
    # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
    torch_ss, stdlib_ss = ss.spawn(2)
    # PyTorch 1.7 and above takes a 64-bit seed
    dtype = np.uint64 if torch.__version__ > "1.7.0" else np.uint32
    torch.manual_seed(torch_ss.generate_state(1, dtype=dtype)[0])
    # use 128 bits expressed as an integer
    stdlib_seed = (
        stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]
    ).sum()
    random.seed(stdlib_seed)


def seed_everything(seed: Optional[int] = None, deterministic: bool = False) -> None:
    """
    To control reproducibility.

    It will seed pseudo-random number generators in: PyTorch, Numpy, python.random. The seed
    can be accessed via the environment variable ``"CLINICADL_GLOBAL_SEED"``.

    Besides, if ``deterministic=True``, PyTorch's operations will be configured in deterministic mode,
    to the extent possible. In this case, an environment variable ``"CLINICADL_DETERMINISTIC"`` will
    also be created.

    .. important:: ``deterministic=True``
        - does not guarantee fully reproducible results; it only ensures determinism within PyTorch’s current limitations;
        - comes with a cost in computing performances. It is advised to use this parameter only for your final
          experiments.

    Parameters
    ----------
    seed : Optional[int], default=None
        The seed to use. If ``None``, a random seed will be generated.
    deterministic : bool, default=False
        Whether to configure PyTorch's operations in deterministic mode.
    """
    if seed is None:
        seed = random.randint(MIN_SEED_VALUE, MAX_SEED_VALUE)
    if not (MIN_SEED_VALUE <= seed <= MAX_SEED_VALUE):
        raise ValueError(
            f"Seed must be between {MIN_SEED_VALUE} and {MAX_SEED_VALUE}. Got {seed}"
        )

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["CLINICADL_GLOBAL_SEED"] = str(seed)
    logger.info("Global seed set to %d", seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        os.environ["CLINICADL_DETERMINISTIC"] = "true"
