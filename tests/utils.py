import os
from functools import wraps

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup_ddp(rank: int, world_size: int, port: int) -> None:
    """
    Sets up DDP. Expects of course GPUs.
    """
    assert torch.cuda.device_count() >= world_size
    torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    backend = "nccl"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def ddp_cleanup() -> None:
    """
    Stops DDP.
    """
    dist.destroy_process_group()


def ddp_wrapper(func: callable) -> callable:
    """
    Turns a function into a function compatible
    with DDP multiprocessing.
    """

    @wraps(func)
    def wrapped(rank, world_size, port, *args, **kwargs):
        try:
            setup_ddp(rank, world_size, port)
            func(rank, *args, **kwargs)
        finally:
            ddp_cleanup()

    return wrapped


def ddp_test(func: callable, world_size: int) -> None:
    """
    Selects and random port and launch parallelism.
    """
    port = np.random.randint(10000, 20000)
    mp.spawn(func, args=(world_size, port), nprocs=world_size, join=True)
