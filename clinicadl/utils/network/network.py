import abc
from logging import getLogger

import torch.cuda
from torch import nn


class Network(nn.Module):
    """Abstract Template for all networks used in ClinicaDL"""

    def __init__(self, use_cpu=False):
        super(Network, self).__init__()
        self.device = self._select_device(use_cpu)

    @staticmethod
    def _select_device(use_cpu):
        import os

        from numpy import argmax

        logger = getLogger("clinicadl")

        if use_cpu:
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
                    nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetMemoryInfo,
                    nvmlInit,
                )

                nvmlInit()
                memory_list = [
                    nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i)).free
                    for i in range(torch.cuda.device_count())
                ]
                free_gpu = argmax(memory_list)
                return f"cuda:{free_gpu}"

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
