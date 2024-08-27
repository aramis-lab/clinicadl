import numpy as np
from torch import nn

from clinicadl.nn.blocks import Decoder3D, Encoder3D
from clinicadl.nn.layers import (
    CropMaxUnpool2d,
    CropMaxUnpool3d,
    PadMaxPool2d,
    PadMaxPool3d,
    Unflatten3D,
)
from clinicadl.nn.networks.cnn import Conv4_FC3, Conv5_FC3
from clinicadl.nn.networks.factory import autoencoder_from_cnn
from clinicadl.nn.utils import compute_output_size
from clinicadl.utils.enum import BaseEnum


class AE2d(str, BaseEnum):
    """AutoEncoders compatible with 2D inputs."""

    AE_CONV5_FC3 = "AE_Conv5_FC3"
    AE_CONV4_FC3 = "AE_Conv4_FC3"


class AE3d(str, BaseEnum):
    """AutoEncoders compatible with 3D inputs."""

    AE_CONV5_FC3 = "AE_Conv5_FC3"
    AE_CONV4_FC3 = "AE_Conv4_FC3"
    CAE_HALF = "CAE_half"


class ImplementedAE(str, BaseEnum):
    """Implemented AutoEncoders in ClinicaDL."""

    AE_CONV5_FC3 = "AE_Conv5_FC3"
    AE_CONV4_FC3 = "AE_Conv4_FC3"
    CAE_HALF = "CAE_half"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not implemented. Implemented AutoEncoders are: "
            + ", ".join([repr(m.value) for m in cls])
        )


# Networks #
class AE(nn.Module):
    """Base class for AutoEncoders."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        indices_list = []
        pad_list = []
        for layer in self.encoder:
            if (
                (isinstance(layer, PadMaxPool3d) or isinstance(layer, PadMaxPool2d))
                and layer.return_indices
                and layer.return_pad
            ):
                x, indices, pad = layer(x)
                indices_list.append(indices)
                pad_list.append(pad)
            elif (
                isinstance(layer, nn.MaxPool3d) or isinstance(layer, nn.MaxPool2d)
            ) and layer.return_indices:
                x, indices = layer(x)
                indices_list.append(indices)
            else:
                x = layer(x)
        return x, indices_list, pad_list

    def decode(self, x, indices_list=None, pad_list=None):
        for layer in self.decoder:
            if isinstance(layer, CropMaxUnpool3d) or isinstance(layer, CropMaxUnpool2d):
                x = layer(x, indices_list.pop(), pad_list.pop())
            elif isinstance(layer, nn.MaxUnpool3d) or isinstance(layer, nn.MaxUnpool2d):
                x = layer(x, indices_list.pop())
            else:
                x = layer(x)
        return x

    def forward(
        self, x
    ):  # TODO : simplify and remove indices_list and pad_list (it is too complicated, there are lot of cases that can raise an issue)
        encoded, indices_list, pad_list = self.encode(x)
        return self.decode(encoded, indices_list, pad_list)


class AE_Conv5_FC3(AE):
    """
    Autoencoder derived from the convolutional part of CNN Conv5_FC3.
    """

    def __init__(self, input_size, dropout):
        cnn_model = Conv5_FC3(
            input_size=input_size, output_size=1, dropout=dropout
        )  # outputsize is not useful as we only take the convolutional part
        encoder, decoder = autoencoder_from_cnn(cnn_model)
        super().__init__(encoder, decoder)


class AE_Conv4_FC3(AE):
    """
    Autoencoder derived from the convolutional part of CNN Conv4_FC3.
    """

    def __init__(self, input_size, dropout):
        cnn_model = Conv4_FC3(
            input_size=input_size, output_size=1, dropout=dropout
        )  # outputsize is not useful as we only take the convolutional part
        encoder, decoder = autoencoder_from_cnn(cnn_model)
        super().__init__(encoder, decoder)


class CAE_half(AE):
    """
    3D Autoencoder derived from CVAE.
    """

    def __init__(
        self, input_size, latent_space_size
    ):  # TODO: doesn't work for even inputs
        encoder = nn.Sequential(
            Encoder3D(1, 32, kernel_size=3),
            Encoder3D(32, 64, kernel_size=3),
            Encoder3D(64, 128, kernel_size=3),
        )
        conv_output_shape = compute_output_size(input_size, encoder)
        flattened_size = np.prod(conv_output_shape)
        encoder.append(nn.Flatten())
        encoder.append(nn.Linear(flattened_size, latent_space_size))
        decoder = nn.Sequential(
            nn.Linear(latent_space_size, flattened_size * 2),
            Unflatten3D(
                256, conv_output_shape[1], conv_output_shape[2], conv_output_shape[3]
            ),
            Decoder3D(256, 128, kernel_size=3),
            Decoder3D(128, 64, kernel_size=3),
            Decoder3D(64, 1, kernel_size=3),
        )
        super().__init__(encoder, decoder)
