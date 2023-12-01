from typing import List

from torch import nn

from clinicadl.utils.network.vae.vae_layers import (
    EncoderConv3DLayer,
    DecoderTranspose3DLayer,
    DecoderUpsample3DLayer,
    EncoderResLayer,
    DecoderResLayer,
)


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
                    block_type,
                    channels[i],
                    channels[i + 1],
                    3,
                    1,
                    1,
                    normalization,
                )
            )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


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
        return self.layers(x)


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
