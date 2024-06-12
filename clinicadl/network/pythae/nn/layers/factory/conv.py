from typing import Type

import torch.nn as nn


class ConvLayer:
    """Factory object for creating Convolutional layers."""

    def __new__(cls, dim: int) -> Type[nn.Module]:
        """
        Creates a Convolutional layer.

        Parameters
        ----------
        dim : int
            Dimension of the image.

        Returns
        -------
        Type[nn.Module]
            The Convolutional layer.
        """
        assert dim in {2, 3}, "Input dimension must be 2 or 3."

        layers = [nn.Conv2d, nn.Conv3d]
        return layers[dim - 2]
