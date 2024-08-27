from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, List, Tuple

from torch import nn

from clinicadl.nn.layers import (
    CropMaxUnpool2d,
    CropMaxUnpool3d,
    PadMaxPool2d,
    PadMaxPool3d,
)

if TYPE_CHECKING:
    from clinicadl.nn.networks.cnn import CNN


def autoencoder_from_cnn(model: CNN) -> Tuple[nn.Module, nn.Module]:
    """
    Constructs an autoencoder from a given CNN.

    The encoder part corresponds to the convolutional part of the CNN.
    The decoder part is the symmetrical network of the encoder.

    Parameters
    ----------
    model : CNN
        The input CNN model

    Returns
    -------
    Tuple[nn.Module, nn.Module]
        The encoder and the decoder.
    """

    encoder = deepcopy(model.convolutions)
    decoder = _construct_inv_cnn(encoder)

    for i, layer in enumerate(encoder):
        if isinstance(layer, PadMaxPool3d) or isinstance(layer, PadMaxPool2d):
            encoder[i].set_new_return()
        elif isinstance(layer, nn.MaxPool3d) or isinstance(layer, nn.MaxPool2d):
            encoder[i].return_indices = True

    return encoder, decoder


def _construct_inv_cnn(model: nn.Module) -> nn.Module:
    """
    Implements a decoder from an CNN encoder.

    The decoder part is the symmetrical list of the encoder
    in which some layers are replaced by their transpose counterpart.
    ConvTranspose and ReLU layers are also inverted.

    Parameters
    ----------
    model : nn.Module
        The input CNN encoder.

    Returns
    -------
    nn.Module
        The symmetrical CNN decoder.
    """
    inv_layers = []
    for layer in model:
        if isinstance(layer, nn.Conv3d):
            inv_layers.append(
                nn.ConvTranspose3d(
                    layer.out_channels,
                    layer.in_channels,
                    layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                )
            )
        elif isinstance(layer, nn.Conv2d):
            inv_layers.append(
                nn.ConvTranspose2d(
                    layer.out_channels,
                    layer.in_channels,
                    layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                )
            )
        elif isinstance(layer, PadMaxPool3d):
            inv_layers.append(CropMaxUnpool3d(layer.kernel_size, stride=layer.stride))
        elif isinstance(layer, PadMaxPool2d):
            inv_layers.append(CropMaxUnpool2d(layer.kernel_size, stride=layer.stride))
        elif isinstance(layer, nn.LeakyReLU):
            inv_layers.append(nn.LeakyReLU(negative_slope=1 / layer.negative_slope))
        else:
            inv_layers.append(deepcopy(layer))
    inv_layers = _invert_conv_and_relu(inv_layers)
    inv_layers.reverse()

    return nn.Sequential(*inv_layers)


def _invert_conv_and_relu(inv_layers: List[nn.Module]) -> List[nn.Module]:
    """
    Invert convolutional and ReLU layers (give empirical better results).

    Parameters
    ----------
    inv_layers : List[nn.Module]
        The list of layers.

    Returns
    -------
    List[nn.Module]
        The modified list of layers.
    """
    idx_relu, idx_conv = -1, -1
    for idx, layer in enumerate(inv_layers):
        if isinstance(layer, nn.ConvTranspose3d) or isinstance(
            layer, nn.ConvTranspose2d
        ):
            idx_conv = idx
        elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU):
            idx_relu = idx

        if idx_conv != -1 and idx_relu != -1:
            inv_layers[idx_relu], inv_layers[idx_conv] = (
                inv_layers[idx_conv],
                inv_layers[idx_relu],
            )
            idx_conv, idx_relu = -1, -1

    # Check if number of features of batch normalization layers is still correct
    for idx, layer in enumerate(inv_layers):
        if isinstance(layer, nn.BatchNorm3d):
            conv = inv_layers[idx + 1]
            inv_layers[idx] = nn.BatchNorm3d(conv.out_channels)
        elif isinstance(layer, nn.BatchNorm2d):
            conv = inv_layers[idx + 1]
            inv_layers[idx] = nn.BatchNorm2d(conv.out_channels)

    return inv_layers
