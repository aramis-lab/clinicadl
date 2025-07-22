import os
import platform
import random
from functools import wraps

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from monai.metrics import ConfusionMatrixMetric


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(random.randint(10000, 20000))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def ddp_test(world_size: int):
    def ddp_test_builder(func):
        @wraps(func)
        def wrapped(rank, *args, **kwargs):
            setup_ddp(rank, world_size)

            try:
                func(rank, *args, **kwargs)
            finally:
                dist.destroy_process_group()

        return wrapped

    return ddp_test_builder


Y_1 = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]])
Y_1_PRED = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]])

Y_2 = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]])
Y_2_PRED = torch.tensor([[1, 0], [0, 1], [0, 1], [1, 0]])

Y_3 = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]])
Y_3_PRED = torch.tensor([[0, 1], [0, 1], [0, 1], [1, 0]])

WORLD_SIZE = 2


def test_metric():
    metric = ConfusionMatrixMetric(metric_name="accuracy")
    metric(Y_1_PRED, Y_1)
    metric(Y_2_PRED, Y_2)
    metric(Y_3_PRED, Y_3)
    assert metric.aggregate()[0].item() == 0.25


@ddp_test(world_size=WORLD_SIZE)
def ddp_worker(rank):
    print(dist.is_available() and dist.is_initialized())
    print(dist.get_rank())
    metric = ConfusionMatrixMetric(metric_name="accuracy")
    if rank == 0:
        pred, target = Y_1_PRED.to(rank), Y_1.to(rank)
        metric(pred, target)
        pred, target = Y_3_PRED.to(rank), Y_3.to(rank)
        metric(pred, target)
    elif rank == 1:
        pred, target = Y_2_PRED.to(rank), Y_2.to(rank)
        metric(pred, target)

    dist.barrier()

    if rank == 0:
        out = metric.aggregate()
        assert out[0].item() == 0.25


@pytest.mark.skipif(platform.system() == "Darwin", reason="Avoid DDP on macOS")
@pytest.mark.timeout(30)
@pytest.mark.ddp
def test_metric_dpp():
    mp.spawn(ddp_worker, nprocs=WORLD_SIZE, join=True)
