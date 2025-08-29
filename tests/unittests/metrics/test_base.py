import os
import random
import socket
from copy import deepcopy
from functools import wraps
from typing import Callable

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchio as tio
from monai.metrics import AveragePrecisionMetric, ConfusionMatrixMetric
from monai.transforms import Activations, AsDiscrete
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


def setup_ddp(rank: int, world_size: int, port: int) -> None:
    """
    Expects of course GPUs.
    """
    print("Setting up...")
    assert torch.cuda.device_count() >= world_size
    torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    backend = "nccl"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup() -> None:
    dist.destroy_process_group()


def ddp_test(func):
    @wraps(func)
    def wrapped(rank, world_size, port, *args, **kwargs):
        print(f"Rank {rank} starting")
        try:
            setup_ddp(rank, world_size, port)
            func(rank, *args, **kwargs)
        finally:
            print(f"[Rank {rank}] cleaning up")
            cleanup()
            print(f"[Rank {rank}] finished")

    return wrapped


WORLD_SIZE = 2


# @ddp_test(world_size=WORLD_SIZE)
# def ddp_worker(rank):
#     # batches = [BATCH[:2], BATCH[2:4], BATCH[4:]]
#     # metric = TestMetric()
#     if rank == 0:
#         assert rank == 0
#         # metric(batches[0])
#         # metric(batches[2])
#     elif rank == 1:
#         assert rank == 1
#         # metric(batches[1])

#     # if rank == 0:
#     #     assert metric.aggregate() == 0.5


@ddp_test
def ddp_worker(rank):
    ap_metric = AveragePrecisionMetric()
    act = Activations(softmax=True)
    to_onehot = AsDiscrete(to_onehot=2)

    device = rank
    if rank == 0:
        y_pred = [
            torch.tensor([0.1, 0.9], device=device),
            torch.tensor([0.3, 1.4], device=device),
        ]
        y = [torch.tensor([0], device=device), torch.tensor([1], device=device)]

    if rank == 1:
        y_pred = [
            torch.tensor([0.2, 0.1], device=device),
            torch.tensor([0.1, 0.5], device=device),
            torch.tensor([0.3, 0.4], device=device),
        ]
        y = [
            torch.tensor([0], device=device),
            torch.tensor([1], device=device),
            torch.tensor([1], device=device),
        ]

    y_pred = [act(p) for p in y_pred]
    y = [to_onehot(y_) for y_ in y]
    ap_metric(y_pred, y)

    result = ap_metric.aggregate()
    np.testing.assert_allclose(0.7778, result, rtol=1e-4)


@pytest.mark.multi_gpu
def test_metric_dpp():
    port = np.random.randint(10000, 20000)
    world_size = 2
    mp.spawn(ddp_worker, args=(world_size, port), nprocs=world_size, join=True)
