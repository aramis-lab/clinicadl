import os
import random

import numpy as np
import torch


def _get_rank() -> int:
    """Returns 0 unless the environment specifies a rank."""
    rank_keys = ("RANK", "SLURM_PROCID", "LOCAL_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


global_rank = _get_rank()


def pl_worker_init_function(worker_id: int) -> None:  # pragma: no cover
    """
    The worker_init_fn that Lightning automatically adds to your dataloader if you previously set
    set the seed with ``seed_everything(seed, workers=True)``.
    See also the PyTorch documentation on
    `randomness in DataLoaders <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_.
    """
    # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    process_seed = torch.initial_seed()
    # back out the base seed so we can use all the bits
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
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


def get_seed(seed: int = None) -> int:
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if seed is None:
        seed = random.randint(min_seed_value, max_seed_value)

    return seed


def seed_everything(seed, deterministic=False, compensation="memory") -> None:
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random

    Adapted from pytorch-lightning
    https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/utilities/seed.html#seed_everything

    Args:
        seed (int): Value of the seed for all pseudo-random number generators
        deterministic (bool): If set to True will raise an error if non-deterministic behaviour is encountered
        compensation (str): Chooses which computational aspect is affected when deterministic is set to True.
            Must be chosen between time and memory.

    Raises:
        ClinicaDLConfigurationError: if compensation is not in {"time", "memory"}.
        RuntimeError: if a non-deterministic behaviour was encountered.

    """
    from clinicadl.utils.exceptions import ClinicaDLConfigurationError

    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if not (min_seed_value <= seed <= max_seed_value):
        seed = random.randint(min_seed_value, max_seed_value)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        if compensation == "memory":
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        elif compensation == "time":
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        else:
            raise ClinicaDLConfigurationError(
                f"The compensation for a deterministic CUDA setting "
                f"must be chosen between 'time' and 'memory'."
            )
        torch.use_deterministic_algorithms(True)
