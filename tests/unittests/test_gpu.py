import pytest
import torch
import torch.distributed as dist

from tests.utils import ddp_test, ddp_wrapper


@pytest.mark.gpu
def test_gpu():
    assert torch.cuda.is_available()


@pytest.mark.multi_gpu
def test_multi_gpu():
    assert torch.cuda.device_count() > 1


@ddp_wrapper
def sum_parallelism(rank):
    tensor: torch.Tensor = torch.ones(1, device=int(rank)) * (rank + 1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    assert tensor.item() == 3


@pytest.mark.multi_gpu
def test_metric_dpp():
    ddp_test(sum_parallelism, world_size=2)
