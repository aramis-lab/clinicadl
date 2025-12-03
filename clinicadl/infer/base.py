from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class Inferer(ABC):
    """
    Base class for inference.

    The only method to overwrite is :py:meth:`__call__`.
    """

    @abstractmethod
    def __call__(
        self, x: torch.Tensor, network: nn.Module, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """
        Defines the inference logic.

        Parameters
        ----------
        x : torch.Tensor
            The input image(s). Can be a single 3D image (CHWD) or a batch (NCHWD).
        network : nn.Module
            The neural network.

        Returns
        -------
        torch.Tensor
            The raw output of the neural network.
        """

    @staticmethod
    def _check_input(x: torch.Tensor) -> None:
        """
        Checks that the input is 4D or 5D tensor.
        """
        if len(x.shape) not in {4, 5}:
            raise ValueError(
                f"Input 'x' must be a single 3D images (4D tensor) or a batch (5D tensor). Got shape: {x.shape}"
            )
