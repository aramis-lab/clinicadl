from abc import abstractmethod

import torch.nn as nn
import torch.optim as optim

from clinicadl.losses.config import LossConfig
from clinicadl.losses.factory import get_loss_function
from clinicadl.networks.config import NetworkConfig
from clinicadl.networks.factory import get_network_from_config
from clinicadl.optimization.optimizer.config import OptimizerConfig
from clinicadl.optimization.optimizer.factory import get_optimizer


class ClinicaDLModel:
    @abstractmethod
    def __init__(self, network: nn.Module, loss: nn.Module, optimizer: optim.Optimizer):
        """TO COMPLETE"""
        pass


class ClinicaDLModelClassif(ClinicaDLModel):
    def __init__(self, network: nn.Module, loss: nn.Module, optimizer: optim.Optimizer):
        """TO COMPLETE"""
        pass

    @classmethod
    def from_config(
        cls,
        network_config: NetworkConfig,
        loss_config: LossConfig,
        optimizer_config: OptimizerConfig,
    ):
        loss, _ = get_loss_function(loss_config)
        network, _ = get_network_from_config(network_config)
        optimizer, _ = get_optimizer(network, optimizer_config)
        return ClinicaDLModel(network=network, loss=loss, optimizer=optimizer)
