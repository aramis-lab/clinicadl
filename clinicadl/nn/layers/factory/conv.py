from typing import Type, Union

import torch.nn as nn


def get_conv_layer(dim: int) -> Union[Type[nn.Conv2d], Type[nn.Conv3d]]:
    """
    A factory function for creating Convolutional layers.

    Parameters
    ----------
    dim : int
        Dimension of the image.

    Returns
    -------
    Type[nn.Module]
        The Convolutional layer.

    Raises
    ------
    AssertionError
        If dim is not 2 or 3.
    """
    assert dim in {2, 3}, "Input dimension must be 2 or 3."

    layers = (nn.Conv2d, nn.Conv3d)
    return layers[dim - 2]
