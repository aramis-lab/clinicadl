from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Type, TypeVar, Union

import torch
import torch.nn as nn

from .base_config import ModelConfig

T = TypeVar("T", bound="ClinicaDLModel")


class ClinicaDLModel(ABC, nn.Module):
    """Abstract template for ClinicaDL Models."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    @abstractmethod
    def from_config(cls: Type[T], config: ModelConfig) -> T:
        """
        Creates a ClinicaDL Model from a config class.

        Parameters
        ----------
        config : ModelConfig
            The config class.

        Returns
        -------
        ClinicaDLModel
            The ClinicaDL Model.
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Pass forward in the network.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, ...]]
            The output. Either a PyTorch tensor (e.g. output of a CNN) or a tuple of tensors
            (e.g. embedding and output of an AutoEncoder).
        """
        pass

    @abstractmethod
    def training_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass forward and loss computation.

        Parameters
        ----------
        x : torch.Tensor
            The batch.

        Returns
        -------
        torch.Tensor
            The loss on which backpropagation will be applied. A 1-item tensor.
        """
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Makes predictions.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The predictions.
        """
        pass

    @abstractmethod
    def save_weights(self, path: Path) -> None:
        """
        Saves network weights.

        Parameters
        ----------
        path : Path
            The file where the weights will be stored.
        """
        pass

    @abstractmethod
    def load_weights(self, path: Path) -> None:
        """
        Loads network weights.

        Parameters
        ----------
        path : Path
            The file where the weights are stored.
        """
        pass
