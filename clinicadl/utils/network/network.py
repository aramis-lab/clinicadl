import abc
from logging import getLogger
from typing import List

import torch.cuda
from torch import nn


class Network(nn.Module):
    """Abstract Template for all networks used in ClinicaDL."""

    def __init__(self, gpu=True):
        super(Network, self).__init__()
        self.device = self._select_device(gpu)

    @staticmethod
    def _select_device(gpu):
        import os

        from numpy import argmax

        logger = getLogger("clinicadl.networks")

        if not gpu:
            return "cpu"
        else:
            # TODO: Add option gpu_device (user chooses the gpu)
            # How to perform multi-GPU ?
            try:
                # In this case, the GPU seen by cuda are restricted and we let cuda choose
                _ = os.environ["CUDA_VISIBLE_DEVICES"]
                return "cuda"
            except KeyError:
                # Else we choose ourselves the GPU with the greatest amount of memory
                from pynvml import (
                    NVMLError,
                    nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetMemoryInfo,
                    nvmlInit,
                )

                try:
                    nvmlInit()
                    memory_list = [
                        nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i)).free
                        for i in range(torch.cuda.device_count())
                    ]
                    free_gpu = argmax(memory_list)
                    return f"cuda:{free_gpu}"
                except NVMLError:
                    logger.warning(
                        "NVML library is not installed. GPU will be chosen arbitrarily"
                    )
                    return "cuda"

    @staticmethod
    @abc.abstractmethod
    def get_input_size() -> str:
        """This static method is used for list_models command.
        Must return the shape of the input size expected (C@HxW or C@HxWxD) for each architecture.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_dimension() -> str:
        """This static method is used for list_models command.
        Return '2D', '3D' or '2D and 3D'
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_task() -> list:
        """This static method is used for list_models command.
        Return the list of tasks for which the model is made.
        """
        pass

    @abc.abstractproperty
    def layers(self):
        pass

    @abc.abstractmethod
    def predict(self, x):
        pass

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=True):
        pass

    def transfer_weights(self, state_dict, transfer_class):
        self.load_state_dict(state_dict)
