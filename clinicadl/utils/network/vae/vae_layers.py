from typing import List

import torch.nn.functional as F
from torch import nn
import torch

from clinicadl.utils.network.vae.vae_utils import get_norm2d, get_norm3d


class EncoderLayer2D(nn.Module):
    """
    Class defining the encoder's part of the Autoencoder.
    This layer is composed of one 2D convolutional layer,
    a batch normalization layer with a leaky relu
    activation function.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        output_padding=0,
        normalization="batch",
    ):
        super(EncoderLayer2D, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=False,
            ),
            get_norm2d(normalization, output_channels),
        )

    def forward(self, x):
        x = F.leaky_relu(self.layer(x), negative_slope=0.2, inplace=True)
        return x


class DecoderLayer2D(nn.Module):
    """
    Class defining the decoder's part of the Autoencoder.
    This layer is composed of one 2D transposed convolutional layer,
    a batch normalization layer with a relu activation function.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        output_padding=0,
        normalization="batch",
    ):
        super(DecoderLayer2D, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=False,
            ),
            get_norm2d(normalization, output_channels),
        )

    def forward(self, x):
        x = F.relu(self.layer(x), inplace=True)
        return x


class EncoderConv3DLayer(nn.Module):
    """
    Class defining the encoder's part of the Autoencoder.
    This layer is composed of one 3D convolutional layer,
    a batch normalization layer with a leaky relu
    activation function.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        normalization="batch",
    ):
        super(EncoderConv3DLayer, self).__init__()
        self.conv = nn.Conv3d(
            input_channels,
            output_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = get_norm3d(normalization, output_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class DecoderTranspose3DLayer(nn.Module):
    """
    Class defining the decoder's part of the Autoencoder.
    This layer is composed of one 3D transposed convolutional layer,
    a batch normalization layer with a relu activation function.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        input_size=None,
        kernel_size=4,
        stride=1,
        padding=1,
        output_padding=0,
        normalization="batch",
    ):
        super(DecoderTranspose3DLayer, self).__init__()
        self.convtranspose = nn.ConvTranspose3d(
            input_channels,
            output_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=False,
        )
        self.norm = get_norm3d(normalization, output_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.convtranspose(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class DecoderUpsample3DLayer(nn.Module):
    """
    Class defining the decoder's part of the Autoencoder.
    This layer is composed of one 3D transposed convolutional layer,
    a batch normalization layer with a relu activation function.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        input_size,
        kernel_size=3,
        stride=1,
        padding=1,
        output_padding=[0, 0, 0],
        normalization="batch",
    ):
        super(DecoderUpsample3DLayer, self).__init__()
        self.upsample = nn.Upsample(
            size=[
                input_size[0] * 2 + output_padding[0],
                input_size[1] * 2 + output_padding[1],
                input_size[2] * 2 + output_padding[2],
            ],
            mode="nearest",
        )
        self.conv = nn.Conv3d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = get_norm3d(normalization, output_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.input_size = input_size
        self.padding = output_padding
        self.dim = (
            [
                input_size[0] * 2 + output_padding[0],
                input_size[1] * 2 + output_padding[1],
                input_size[2] * 2 + output_padding[2],
            ],
        )

    def forward(self, x):
        out = self.upsample(x)
        out = self.conv(out)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Unflatten2D(nn.Module):
    def __init__(self, channel, height, width):
        super(Unflatten2D, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width

    def forward(self, input):
        return input.view(input.size(0), self.channel, self.height, self.width)


class Unflatten3D(nn.Module):
    def __init__(self, channel, height, width, depth):
        super(Unflatten3D, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width
        self.depth = depth

    def forward(self, input):
        return input.view(
            input.size(0), self.channel, self.height, self.width, self.depth
        )


class EncoderResLayer(nn.Module):
    """
    This layer is composed of a residual block.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        normalization="batch",
    ):
        super(EncoderResLayer, self).__init__()

        if output_channels != input_channels: 
            stride = 2
            kernel_size = 4
        else: 
            stride = 1
            kernel_size = 3

        self.conv1 = nn.Conv3d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.norm1 = get_norm3d(normalization, output_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            output_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = get_norm3d(normalization, output_channels)

        if input_channels == output_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    input_channels,
                    output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm3d(output_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        # print("After (layer x (y) 1st conv)", out.shape)

        out = self.conv2(out)
        out = self.norm2(out)
        # print("After (layer x (y) 2nd conv)", out.shape)

        out += self.shortcut(x)
        out = self.relu(out)
        # print("After (layer x (y) shortcut)", out.shape)

        return out


class DecoderResLayer(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        input_size,
        output_padding,
        normalization="batch",
    ):
        super(DecoderResLayer, self).__init__()

        self.conv2 = nn.Conv3d(
            input_channels,
            input_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.norm2 = get_norm3d(normalization, input_channels)

        if input_channels == output_channels:
            self.conv1 = nn.Conv3d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = nn.Sequential(
                nn.Upsample(
                    size=[
                        input_size[0] * 2 + output_padding[0],
                        input_size[1] * 2 + output_padding[1],
                        input_size[2] * 2 + output_padding[2],
                    ],
                    mode="nearest",
                ),
                nn.Conv3d(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
            )
            self.shortcut = nn.Sequential(
                nn.Upsample(
                    size=[
                        input_size[0] * 2 + output_padding[0],
                        input_size[1] * 2 + output_padding[1],
                        input_size[2] * 2 + output_padding[2],
                    ],
                    mode="nearest",
                ),
                nn.Conv3d(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                get_norm3d(normalization, output_channels),
            )
        
        self.norm1 = get_norm3d(normalization, output_channels)

    def forward(self, x):
        out = self.conv2(x)
        out = self.norm2(out)
        out = F.relu(out)
        # print("After (layer x (y) 1st conv)", out.shape)

        out = self.conv1(out)
        out = self.norm1(out)
        # print("After layer (x (y) 2nd conv)", out.shape)

        out += self.shortcut(x)
        # print("After layer (x (y) shortcut)", out.shape)

        out = F.relu(out)
        return out


class EncoderBlock(nn.Module):
    """Encoder block where we choose the type of layer (conv or res)"""

    def __init__(
        self,
        input_channels,
        output_channels,
        n_layer_per_block_encoder=3,
        block_type="conv",
        normalization="batch",
    ):
        super(EncoderBlock, self).__init__()

        channels = get_channels(
            input_channels, output_channels, n_layer_per_block_encoder
        )

        layers = []

        layers.append(
            get_encoder_layer(
                block_type, channels[0], channels[1], 4, 2, 1, normalization
            )
        )

        for i in range(1, n_layer_per_block_encoder):
            layers.append(
                get_encoder_layer(
                    block_type, channels[i], channels[i + 1], 3, 1, 1, normalization,
                )
            )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # print("Before encoder block (x): ", x.shape)
        out = self.layers(x)
        # print("After encoder block (x): ", out.shape)
        return out


class DecoderBlock(nn.Module):
    """Decoder block where we choose the type of layer (conv or res)"""

    def __init__(
        self,
        input_channels,
        output_channels,
        input_size,
        output_padding,
        n_layer_per_block_decoder,
        block_type="upsample",
        normalization="batch",
    ):
        super(DecoderBlock, self).__init__()

        channels = get_channels(
            output_channels, input_channels, n_layer_per_block_decoder
        )

        layers = []

        for i in range(n_layer_per_block_decoder, 1, -1):
            layers.append(
                get_decoder_layer(
                    block_type,
                    input_channels=channels[i],
                    output_channels=channels[i - 1],
                    input_size=input_size,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    output_padding=[0, 0, 0],
                    normalization=normalization,
                )
            )

        layers.append(
            get_decoder_layer(
                block_type,
                input_channels=channels[1],
                output_channels=channels[0],
                input_size=input_size,
                kernel_size=3,
                stride=1,
                padding=1,
                output_padding=output_padding,
                normalization=normalization,
            )
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # print("Before decoder block (x): ", x.shape)
        out = self.layers(x)
        # print("After decoder block (x): ", out.shape)
        return out


def get_channels(
    input_channels: int, output_channels: int, n_layer_per_block_encoder: int
):
    channels = []
    channels.append(input_channels)
    for _ in range(n_layer_per_block_encoder):
        channels.append(output_channels)
    return channels


def get_encoder_layer(
    block_type: str,
    input_channels: int,
    output_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    normalization: str,
):
    if block_type in ["conv", "transpose", "upsample"]:
        return EncoderConv3DLayer(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding,
            normalization,
        )
    elif block_type == "res":
        return EncoderResLayer(
            input_channels,
            output_channels,
            normalization,
        )
    else:
        raise AttributeError("Bad block type specified. Block type must be conv or res")


def get_decoder_layer(
    block_type: str,
    input_channels: int,
    output_channels: int,
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: List[int],
    normalization: str,
):
    if block_type in ["conv", "upsample"]:
        return DecoderUpsample3DLayer(
            input_channels,
            output_channels,
            input_size,
            kernel_size,
            stride,
            padding,
            output_padding,
            normalization,
        )
    elif block_type == "transpose":
        return DecoderTranspose3DLayer(
            input_channels,
            output_channels,
            input_size,
            kernel_size,
            stride,
            padding,
            output_padding,
            normalization,
        )
    elif block_type == "res":
        return DecoderResLayer(
            input_channels,
            output_channels,
            input_size,
            output_padding,
            normalization,
        )
    else:
        raise AttributeError(
            "Bad block type specified. Block type must be conv, upsample, transpose or res"
        )


class VAE_Encoder(nn.Module):
    """"""

    def __init__(
        self,
        input_shape,
        n_conv=4,
        first_layer_channels=32,
        latent_dim=1,
        feature_size=1024,
    ):
        """
        Feature size is the size of the vector if latent_dim=1
        or is the number of feature maps (number of channels) if latent_dim=2
        """
        super(VAE_Encoder, self).__init__()

        self.input_c = input_shape[0]
        self.input_h = input_shape[1]
        self.input_w = input_shape[2]

        decoder_padding = []
        tensor_h, tensor_w = self.input_h, self.input_w

        self.layers = []

        # Input Layer
        self.layers.append(EncoderLayer2D(self.input_c, first_layer_channels))
        padding_h, padding_w = 0, 0
        if tensor_h % 2 != 0:
            padding_h = 1
        if tensor_w % 2 != 0:
            padding_w = 1
        decoder_padding.append([padding_h, padding_w])
        tensor_h, tensor_w = tensor_h // 2, tensor_w // 2
        # Conv Layers
        for i in range(n_conv - 1):
            self.layers.append(
                EncoderLayer2D(
                    first_layer_channels * 2**i, first_layer_channels * 2 ** (i + 1)
                )
            )
            padding_h, padding_w = 0, 0
            if tensor_h % 2 != 0:
                padding_h = 1
            if tensor_w % 2 != 0:
                padding_w = 1
            decoder_padding.append([padding_h, padding_w])
            tensor_h, tensor_w = tensor_h // 2, tensor_w // 2

        self.decoder_padding = decoder_padding

        # Final Layer
        if latent_dim == 1:
            n_pix = (
                first_layer_channels
                * 2 ** (n_conv - 1)
                * (self.input_h // (2**n_conv))
                * (self.input_w // (2**n_conv))
            )
            self.layers.append(
                nn.Sequential(Flatten(), nn.Linear(n_pix, feature_size), nn.ReLU())
            )
        elif latent_dim == 2:
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        first_layer_channels * 2 ** (n_conv - 1),
                        feature_size,
                        3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.ReLU(),
                )
            )
        else:
            raise AttributeError(
                "Bad latent dimension specified. Latent dimension must be 1 or 2"
            )

        self.sequential = nn.Sequential(*self.layers)

    def forward(self, x):
        z = self.sequential(x)
        return z


class VAE_Decoder(nn.Module):
    """"""

    def __init__(
        self,
        input_shape,
        latent_size,
        n_conv=4,
        last_layer_channels=32,
        latent_dim=1,
        feature_size=1024,
        padding=None,
    ):
        """
        Feature size is the size of the vector if latent_dim=1
        or is the W/H of the output channels if laten_dim=2
        """
        super(VAE_Decoder, self).__init__()

        self.input_c = input_shape[0]
        self.input_h = input_shape[1]
        self.input_w = input_shape[2]

        if not padding:
            output_padding = [[0, 0] for i in range(n_conv - 1)]
        else:
            output_padding = padding

        self.layers = []

        if latent_dim == 1:
            n_pix = (
                last_layer_channels
                * 2 ** (n_conv - 1)
                * (self.input_h // (2**n_conv))
                * (self.input_w // (2**n_conv))
            )
            self.layers.append(
                nn.Sequential(
                    nn.Linear(latent_size, feature_size),
                    nn.ReLU(),
                    nn.Linear(feature_size, n_pix),
                    nn.ReLU(),
                    Unflatten2D(
                        last_layer_channels * 2 ** (n_conv - 1),
                        self.input_h // (2**n_conv),
                        self.input_w // (2**n_conv),
                    ),
                    nn.ReLU(),
                )
            )
        elif latent_dim == 2:
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        latent_size, feature_size, 3, stride=1, padding=1, bias=False
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose2d(
                        feature_size,
                        last_layer_channels * 2 ** (n_conv - 1),
                        3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.ReLU(),
                )
            )
        else:
            raise AttributeError(
                "Bad latent dimension specified. Latent dimension must be 1 or 2"
            )

        for i in range(n_conv - 1, 0, -1):
            self.layers.append(
                DecoderLayer2D(
                    last_layer_channels * 2 ** (i),
                    last_layer_channels * 2 ** (i - 1),
                    output_padding=output_padding[i],
                )
            )

        self.layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    last_layer_channels,
                    self.input_c,
                    4,
                    stride=2,
                    padding=1,
                    output_padding=output_padding[0],
                    bias=False,
                ),
                nn.Sigmoid(),
            )
        )

        self.sequential = nn.Sequential(*self.layers)

    def forward(self, z):
        y = self.sequential(z)
        return y