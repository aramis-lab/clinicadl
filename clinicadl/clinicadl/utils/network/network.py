import abc

from torch import nn


class Network(nn.Module):
    """Abstract Template for all networks used in ClinicaDL"""

    def __init__(self, use_cpu=False):
        super(Network, self).__init__()
        # TODO: check if gpu is available
        self.device = self._select_device(use_cpu)

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

    def predict(self, input):
        return self.layers(input)

    @abc.abstractmethod
    def forward(self, input):
        pass

    @abc.abstractmethod
    def compute_outputs_and_loss(self, input_dict, criterion):
        pass

    @abc.abstractproperty
    def layers(self):
        pass


class CNN(Network):
    def __init__(self, use_cpu=False, n_classes=2):
        super().__init__(use_cpu=use_cpu)
        self.n_classes = n_classes
        self.layers = nn.Sequential(self.convolutions, self.classifier)

    @abc.abstractproperty
    def convolutions(self):
        pass

    @abc.abstractproperty
    def classifier(self):
        pass

    def forward(self, input):
        return self.predict(input)

    def compute_outputs_and_loss(self, input_dict, criterion):

        imgs, labels = input_dict["image"].to(self.device), input_dict["label"].to(
            self.device
        )
        train_output = model(imgs)
        return train_output, criterion(train_output, labels)
