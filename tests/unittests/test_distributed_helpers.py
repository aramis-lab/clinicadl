import multiprocessing
import sys
import time
from datetime import timedelta
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from tests import utils


@utils.ddp_wrapper
def sum_on_cpu(rank):
    value = torch.tensor([rank + 1])
    dist.all_reduce(value)
    assert value.item() == 3


@utils.ddp_wrapper
def fail_on_cpu(rank):
    if rank == 0:
        raise RuntimeError("intentional worker failure")
    dist.barrier()


def wait_forever(rank, world_size, port, use_cuda):
    time.sleep(60)


@pytest.mark.parametrize("use_cuda", [False, True])
def test_setup_ddp_selects_backend(monkeypatch, use_cuda):
    initialize = Mock()
    set_device = Mock()
    monkeypatch.setattr(dist, "init_process_group", initialize)
    monkeypatch.setattr(torch.cuda, "device_count", Mock(return_value=2))
    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    utils.setup_ddp(1, 2, 12345, use_cuda=use_cuda)
    kwargs = {} if use_cuda else {"timeout": timedelta(seconds=30)}
    if not use_cuda and sys.platform == "win32":
        kwargs["init_method"] = "env://?use_libuv=0"
    initialize.assert_called_once_with(
        "nccl" if use_cuda else "gloo", rank=1, world_size=2, **kwargs
    )
    if use_cuda:
        set_device.assert_called_once_with(1)
    else:
        set_device.assert_not_called()
        torch.cuda.device_count.assert_not_called()


@pytest.mark.parametrize("initialized", [False, True])
def test_cleanup_only_destroys_initialized_group(monkeypatch, initialized):
    destroy = Mock()
    monkeypatch.setattr(dist, "is_initialized", lambda: initialized)
    monkeypatch.setattr(dist, "destroy_process_group", destroy)
    utils.ddp_cleanup()
    assert destroy.call_count == int(initialized)


def test_wrapper_preserves_setup_error(monkeypatch):
    monkeypatch.setattr(
        utils, "setup_ddp", Mock(side_effect=RuntimeError("setup failed"))
    )
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="setup failed"):
        sum_on_cpu(0, 2, 12345, False)


def test_default_launch_keeps_gpu_arguments(monkeypatch):
    spawn = Mock()
    monkeypatch.setattr(utils.mp, "spawn", spawn)
    monkeypatch.setattr(utils.np.random, "randint", lambda *args: 12345)
    utils.ddp_test(sum_on_cpu, 2)
    spawn.assert_called_once_with(sum_on_cpu, args=(2, 12345), nprocs=2, join=True)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="PyTorch Gloo unavailable")
@pytest.mark.parametrize("worker", [sum_on_cpu, fail_on_cpu])
def test_real_cpu_workers_finish_and_cleanup(worker):
    previous = {p.pid for p in multiprocessing.active_children()}
    if worker is fail_on_cpu:
        with pytest.raises(
            torch.multiprocessing.ProcessRaisedException,
            match="intentional worker failure",
        ):
            utils.ddp_test(worker, 2, use_cuda=False)
    else:
        utils.ddp_test(worker, 2, use_cuda=False)
    assert {p.pid for p in multiprocessing.active_children()} <= previous


def test_cpu_timeout_stops_workers():
    previous = {p.pid for p in multiprocessing.active_children()}
    with pytest.raises(TimeoutError, match="did not finish"):
        utils.ddp_test(wait_forever, 2, use_cuda=False, timeout=1)
    assert {p.pid for p in multiprocessing.active_children()} <= previous
