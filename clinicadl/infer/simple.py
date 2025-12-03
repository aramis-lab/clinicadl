from typing import Any

import torch
import torch.nn as nn

from .base import Inferer


class SimpleInferer(Inferer):
    """
    For classical inference, i.e. when the whole images are passed
    in the neural network and the raw outputs are returned.
    """

    def __call__(
        self, x: torch.Tensor, network: nn.Module, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """
        Simple pass forward in the neural network.

        Parameters
        ----------
        x : torch.Tensor
            The input image(s). Can be a single image or a batch.
        network : nn.Module
            The neural network.
        args : Any
            Optional args to be passed to ``network``.
        kwargs : Any
            Optional keyword args to be passed to ``network``.

        Returns
        -------
        torch.Tensor
            The raw output of the neural network.
        """
        return network(x, *args, **kwargs)
