import os
import random
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

Y_1 = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]])
Y_1_PRED = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]])

Y_2 = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]])
Y_2_PRED = torch.tensor([[1, 0], [0, 1], [0, 1], [1, 0]])

Y_3 = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]])
Y_3_PRED = torch.tensor([[0, 1], [0, 1], [0, 1], [1, 0]])


class TestMetric(Metric):
    _optimum = "max"

    def _accumulate(self, batch):
        pred = torch.stack([datapoint["output"] for datapoint in batch])
        gt = torch.stack([datapoint["label"] for datapoint in batch])
        return (pred == gt).all(1)

    def _aggregate(self, data):
        return data.float().mean()


def test_metric():
    metric = TestMetric()

    batches = ((Y_1_PRED, Y_1), (Y_2_PRED, Y_2), (Y_3_PRED, Y_3))
    results = (
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )

    for (preds, ground_truths), result in zip(batches, results):
        batch = [deepcopy(DATAPOINT) for _ in range(len(preds))]
        for datapoint, pred, gt in zip(batch, preds, ground_truths):
            datapoint["label"] = gt
            datapoint["output"] = pred

        assert (metric(batch) == result).all()

    assert metric.aggregate() == 0.25

    assert metric.optimum == "max"


def setup_ddp(rank: int, world_size: int) -> None:
    """
    Expects of course GPUs.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    backend = "nccl"
    assert torch.cuda.device_count() >= world_size
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup() -> None:
    dist.destroy_process_group()


def ddp_test(world_size: int) -> Callable[[Callable], Callable]:
    def ddp_test_builder(func):
        @wraps(func)
        def wrapped(rank, *args, **kwargs):
            setup_ddp(rank, world_size)

            try:
                func(rank, *args, **kwargs)
            finally:
                cleanup()

        return wrapped

    return ddp_test_builder


WORLD_SIZE = 2


# @ddp_test(world_size=WORLD_SIZE)
# def ddp_worker(rank):
#     metric = ConfusionMatrixMetric(metric_name="accuracy")
#     if rank == 0:
#         pred, target = Y_1_PRED.to(rank), Y_1.to(rank)
#         metric(pred, target)
#         pred, target = Y_3_PRED.to(rank), Y_3.to(rank)
#         metric(pred, target)
#     elif rank == 1:
#         pred, target = Y_2_PRED.to(rank), Y_2.to(rank)
#         metric(pred, target)

#     if rank == 0:
#         out = metric.aggregate()
#         assert out[0].item() == 0.25


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


@ddp_test(world_size=WORLD_SIZE)
def ddp_worker(rank):
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    outputs = ddp_model(torch.randn(20, 10))
    assert 1 == 0


@pytest.mark.multi_gpu
def test_metric_dpp():
    mp.spawn(ddp_worker, nprocs=WORLD_SIZE, join=True)
