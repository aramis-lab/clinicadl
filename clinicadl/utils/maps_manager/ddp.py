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


def methodsWithoutDunders(obj: Any) -> Set[str]:
    """
    Takes any object, and returns a set of all its methods names, without
    dunders and attributes.

    Args:
        obj (Any): the object whose methods we want.
    Returns:
        (Set[str]): set of methods name.
    """
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


def get_custom_methods(model: Module) -> Set[str]:
    """
    Only get newly-added methods from the object, without any method already
    defined in the Module parent class.

    Args:
        model (torch.nn.Module): the model to analyze.
    Returns:
        (Set[str]): set of methods name without methods from Module parent class.
    """
    model_methods = methodsWithoutDunders(model)
    base_methods = methodsWithoutDunders(Module)
    return model_methods - base_methods


def forward(self, input_dict, criterion=None, use_labels=True):
    if criterion is None:
        return self.predict(input_dict)
    else:
        return self.compute_outputs_and_loss(
            input_dict, criterion, use_labels=use_labels
        )


def monkeypatch(model: Module) -> None:
    """
    Patch a model to replace the forward method and store it in method _forward.
    This is done in order to enable DistributedDataParallelism since Pytorch uses
    "forward" as a keyword for its models.

    Args:
        model (torch.nn.Module): model to be trained and needs to be monkeypatched
        to be distributed
    """
    method_names = get_custom_methods(model)

    if "_forward" in method_names:
        # A _forward function already exists, we abort the monkeypatching
        # procedure in order not to break the object.
        return

    for method_name in method_names:
        method = getattr(model, method_name)

        try:
            source_code = inspect.getsource(method)
        except OSError:
            # The method probably has been created dynamically, this is a
            # failure case of monkeypatching. Therefore we abort the patching
            # of this method and assume it does not need it.
            continue

        if "self.forward" in source_code:
            # The function calls the forward method, so needs to be patched
            monkeypatched_code = dedent(
                source_code.replace("self.forward", "self._forward")
            )
            compiled_code = compile(monkeypatched_code, "<string>", "exec")

            # If the function has default arguments, then the code of the function
            # will not be the first constant in the defined code but will be after
            # in the list so we look for it.
            for const in compiled_code.co_consts:
                if isinstance(const, CodeType) and const.co_name == method_name:
                    break
            else:
                raise ValueError("Expected to find code object, did not find any.")

            # Convert code to a method bound to the given model.
            function = FunctionType(
                code=const,
                globals=method.__globals__,
                name=method.__name__,
            )
            method = MethodType(function, model)
            setattr(model, method_name, method)

    # We introduce our new forward function and store the old one into the
    # "_forward" method.
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
    """
    Initiates the process group if not already done.

    Args:
        gpu (bool): whether or not the training process will be performed on GPU.
        logger (logging.Logger): logger to filter so that only one process prints
            its output.
    """
    gpu = gpu and torch.cuda.is_available()

    if not dist.is_initialized():
        init_process_group(gpu=gpu)
    if gpu:
        torch.cuda.set_device(cluster.local_rank)
    if logger is not None:
        # Make sure the logging is performed only on one process so that our logs
        # is not messed up.
        logger.addFilter(cluster.Rank0Filter(rank=cluster.rank))

    assert (
        dist.is_initialized()
    ), "Something went wrong with the distribution initialization!"
