from typing import Any, Dict, Optional, Tuple, Type, Union

import torch.nn as nn
from monai.networks.layers.factories import LayerFactory, split_args
from monai.utils import has_option

from .enum import UnpoolingLayer

Unpool = LayerFactory(
    name="Unpooling layers", description="Factory for creating unpooling layers."
)


@Unpool.factory_function("upsample")
def upsample_factory(dim: int) -> Type[nn.Upsample]:
    """
    Upsample layer.
    """
    return nn.Upsample


@Unpool.factory_function("convtranspose")
def convtranspose_factory(
    dim: int,
) -> Type[Union[nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]]:
    """
    Transposed convolutional layers in 1,2,3 dimensions.

    Parameters
    ----------
    dim : int
        desired dimension of the transposed convolutional layer.

    Returns
    -------
    type[Union[nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]]
        ConvTranspose[dim]d
    """
    types = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    return types[dim - 1]


def get_unpool_layer(
    name: Union[UnpoolingLayer, Tuple[UnpoolingLayer, Dict[str, Any]]],
    spatial_dims: int,
    in_channels: Optional[int] = None,
    out_channels: Optional[int] = None,
) -> nn.Module:
    """
    Creates an unpooling layer instance.

    Parameters
    ----------
    name : Union[UnpoolingLayer, Tuple[UnpoolingLayer, Dict[str, Any]]]
        the unpooling type, potentially with arguments in a dict.

    Returns
    -------
    nn.Module
        the parametrized unpooling layer.

    Parameters
    ----------
    name : Union[UnpoolingLayer, Tuple[UnpoolingLayer, Dict[str, Any]]]
        the unpooling type, potentially with arguments in a dict.
    spatial_dims : int
        number of spatial dimensions of the input.
    in_channels : Optional[int] (optional, default=None)
        number of input channels if the unpool layer requires this parameter.
    out_channels : Optional[int] (optional, default=None)
        number of output channels if the unpool layer requires this parameter.

    Returns
    -------
    nn.Module
        the parametrized unpooling layer.
    """
    unpool_name, unpool_args = split_args(name)
    unpool_name = UnpoolingLayer(unpool_name)
    unpool_type = Unpool[unpool_name, spatial_dims]
    kw_args = dict(unpool_args)
    if has_option(unpool_type, "in_channels") and "in_channels" not in kw_args:
        kw_args["in_channels"] = in_channels
    if has_option(unpool_type, "out_channels") and "out_channels" not in kw_args:
        kw_args["out_channels"] = out_channels

    return unpool_type(**kw_args)
