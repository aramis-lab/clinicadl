from logging import Logger
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from . import cluster


class DDP(DistributedDataParallel):
    def _forward(self, *args, **kwargs):
        return self.module._forward(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.module.predict(*args, **kwargs)

    def transfer_weights(self, *args, **kwargs):
        return self.module.transfer_weights(*args, **kwargs)

    def state_dict(self):
        return self.module.state_dict()


def get_backend(gpu: bool = False):
    if gpu and dist.is_nccl_available():
        return "nccl"
    if dist.is_gloo_available():
        return "gloo"
    if dist.is_mpi_available():
        return "mpi"
    raise NotImplementedError("No good backend found")


def init_ddp(gpu: bool = True, logger: Optional[Logger] = None):
    dist.init_process_group(
        backend=get_backend(gpu=gpu),
        init_method="env://",
        world_size=cluster.world_size,
        rank=cluster.rank
    )
    if gpu:
        torch.cuda.set_device(cluster.local_rank)
    if logger is not None:
        logger.addFilter(cluster.Rank0Filter(rank=cluster.rank))

    assert dist.is_initialized(), "Something went wrong with the distribution initialization!"
