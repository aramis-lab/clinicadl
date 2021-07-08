import abc

from torch import nn

from clinicadl.utils.descriptors import classproperty


class Network(nn.Module):
    """Abstract Template for all networks used in ClinicaDL"""

    def __init__(self, use_cpu=False):
        super(Network, self).__init__()
        # TODO: check if gpu is available
        self.device = self._select_device(use_cpu)

    @classproperty
    def possible_tasks(cls):
        return cls._possible_tasks

    @staticmethod
    def _select_device(use_cpu):
        import os

        from numpy import argmax

        if use_cpu:
            return "cpu"
        else:
            # TODO: check on cluster (add try except)
            # Add option gpu_device (user chooses the gpu)
            # How to perform multi-GPU ?
            # Use get device properties de pytorch instead of nvidia-smi
            os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
            memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
            free_gpu = argmax(memory_available)
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
