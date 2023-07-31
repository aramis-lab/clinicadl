import inspect
from dataclasses import dataclass
from logging import Logger
from textwrap import dedent
from types import CodeType, FunctionType, MethodType
from typing import Any, Optional, Set

import torch
import torch.distributed as dist
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel

from . import cluster


@dataclass
class Methods:
    name: str
    method: Any


def methodsWithoutDunders(obj) -> Set[str]:
    namesWithMethods = map(lambda name: Methods(name, getattr(obj, name)), dir(obj))
    withoutDunders = filter(
        lambda method: not (
            method.name.startswith("__") and method.name.endswith("__")
        ),
        namesWithMethods,
    )
    onlyMethods = filter(
        lambda method: isinstance(method.method, (FunctionType, MethodType)),
        withoutDunders,
    )
    onlyNames = map(
        lambda method: method.name,
        onlyMethods,
    )
    return set(onlyNames)


def get_custom_methods(obj) -> Set[str]:
    object_methods = methodsWithoutDunders(obj)
    base_methods = methodsWithoutDunders(Module)
    return object_methods - base_methods


def forward(self, input_dict, criterion=None, use_labels=True):
    if criterion is None:
        return self.predict(input_dict)
    else:
        return self.compute_outputs_and_loss(
            input_dict, criterion, use_labels=use_labels
        )


def monkeypatch(model: Module) -> None:
    method_names = get_custom_methods(model)
    for method_name in method_names:
        method = getattr(model, method_name)
        source_code = inspect.getsource(method)
        if "self.forward" in source_code:
            monkeypatched_code = dedent(
                source_code.replace("self.forward", "self._forward")
            )
            compiled_code = compile(monkeypatched_code, "<string>", "exec")
            for const in compiled_code.co_consts:
                if isinstance(const, CodeType) and const.co_name == method_name:
                    break
            else:
                raise ValueError("Expected to find code object, did not find any.")
            function = FunctionType(
                code=const,
                globals=method.__globals__,
                name=method.__name__,
            )
            method = MethodType(function, model)
            setattr(model, method_name, method)
    model._forward = model.forward
    model.forward = MethodType(forward, model)


class DDP(DistributedDataParallel):
    def __init__(self, model: Module, *args, **kwargs):
        monkeypatch(model)
        super().__init__(model, *args, **kwargs)

    def _forward(self, *args, **kwargs):
        return self.module._forward(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.module.predict(*args, **kwargs)

    def transfer_weights(self, *args, **kwargs):
        return self.module.transfer_weights(*args, **kwargs)

    def state_dict(self):
        return self.module.state_dict()


def get_backend(gpu: bool = False) -> str:
    if gpu and dist.is_nccl_available():
        return "nccl"
    if dist.is_gloo_available():
        return "gloo"
    if dist.is_mpi_available():
        return "mpi"
    raise NotImplementedError("No good backend found")


def init_process_group(gpu: bool = False) -> None:
    dist.init_process_group(
        backend=get_backend(gpu=gpu),
        init_method="env://",
        rank=cluster.rank,
        world_size=cluster.world_size,
    )


def init_ddp(gpu: bool = True, logger: Optional[Logger] = None) -> None:
    gpu = gpu and torch.cuda.is_available()

    if not dist.is_initialized():
        init_process_group(gpu=gpu)
    if gpu:
        torch.cuda.set_device(cluster.local_rank)
    if logger is not None:
        logger.addFilter(cluster.Rank0Filter(rank=cluster.rank))

    assert (
        dist.is_initialized()
    ), "Something went wrong with the distribution initialization!"
