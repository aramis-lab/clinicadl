import os
import random
import socket
from copy import deepcopy
from functools import wraps
from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchio as tio
from monai.metrics import ConfusionMatrixMetric
from torch.nn.parallel import DistributedDataParallel as DDP

from clinicadl.data.structures import DataPoint
from clinicadl.metrics import Metric

DATAPOINT = DataPoint(
    image=tio.ScalarImage(tensor=torch.randn(1, 2, 2, 2)),
    label=None,
    participant="abc",
    session="abc",
)

GTS = [[0, 1], [1, 0], [1, 0], [1, 0], [0, 1], [1, 0]]
PREDS = [[0, 1], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1]]

BATCH = [
    DataPoint(
        image=tio.ScalarImage(tensor=torch.randn(1, 2, 2, 2)),
        label=torch.tensor(gt),
        output=torch.tensor(pred),
        participant="abc",
        session="abc",
    )
    for pred, gt in zip(PREDS, GTS)
]


class CustomTestMetric(Metric):
    _optimum = "max"

    def _accumulate(self, batch):
        pred = torch.stack([datapoint["output"] for datapoint in batch])
        gt = torch.stack([datapoint["label"] for datapoint in batch])
        return (pred == gt).all(1)

    def _aggregate(self, data):
        return data.float().mean()


def test_metric():
    metric = CustomTestMetric()

    batches = [BATCH[:2], BATCH[2:4], BATCH[4:]]
    results = (
        torch.tensor([1.0, 1.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.0, 0.0]),
    )
    for batch, result in zip(batches, results):
        assert (metric(batch) == result).all()

    assert metric.aggregate() == 0.5

    assert metric.optimum == "max"


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # let the OS pick a free port
        return s.getsockname()[1]


def setup_ddp(rank: int, world_size: int) -> None:
    """
    Expects of course GPUs.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_port())
    backend = "nccl"
    assert torch.cuda.device_count() >= world_size
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup() -> None:
    dist.destroy_process_group()


def ddp_test(world_size: int) -> Callable[[Callable], Callable]:
    def ddp_test_builder(func):
        @wraps(func)
        def wrapped(rank, *args, **kwargs):
            try:
                setup_ddp(rank, world_size)
                func(rank, *args, **kwargs)
            finally:
                cleanup()

        return wrapped

    return ddp_test_builder


WORLD_SIZE = 2


@ddp_test(world_size=WORLD_SIZE)
def ddp_worker(rank):
    # batches = [BATCH[:2], BATCH[2:4], BATCH[4:]]
    # metric = TestMetric()
    if rank == 0:
        assert rank == 0
        # metric(batches[0])
        # metric(batches[2])
    elif rank == 1:
        assert rank == 1
        # metric(batches[1])

    # if rank == 0:
    #     assert metric.aggregate() == 0.5


@pytest.mark.multi_gpu
def test_metric_dpp():
    mp.spawn(ddp_worker, nprocs=WORLD_SIZE, join=True)
