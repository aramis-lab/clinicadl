from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput

from clinicadl.utils.network.vae.vae_layers import (
    DecoderLayer3D,
    EncoderLayer3D,
    Flatten,
    Unflatten3D,
)

import torch
from torch import nn

def build_encoder_decoder(
    input_size = (1, 80, 96, 80),
    latent_space_size=128,
    feature_size=0,
    n_conv=3,
    io_layer_channels=32,
):
    first_layer_channels = io_layer_channels
    last_layer_channels = io_layer_channels
    # automatically compute padding
    decoder_output_padding = []

    input_c = input_size[0]
    input_d = input_size[1]
    input_h = input_size[2]
    input_w = input_size[3]
    d, h, w = input_d, input_h, input_w

    # ENCODER
    encoder_layers = []
    # Input Layer
    encoder_layers.append(EncoderLayer3D(input_c, first_layer_channels))
    decoder_output_padding.append([d % 2, h % 2, w % 2])
    d, h, w = d // 2, h // 2, w // 2
    # Conv Layers
    for i in range(n_conv - 1):
        encoder_layers.append(
            EncoderLayer3D(
                first_layer_channels * 2**i, first_layer_channels * 2 ** (i + 1)
            )
        )
        # Construct output paddings
        decoder_output_padding.append([d % 2, h % 2, w % 2])
        d, h, w = d // 2, h // 2, w // 2
    # Compute size of the feature space
    n_pix = (
        first_layer_channels
        * 2 ** (n_conv - 1)
        * (input_d // (2**n_conv))
        * (input_h // (2**n_conv))
        * (input_w // (2**n_conv))
    )
    # Flatten
    encoder_layers.append(Flatten())
    # Intermediate feature space
    if feature_size == 0:
        feature_space = n_pix
    else:
        feature_space = feature_size
        encoder_layers.append(
            nn.Sequential(nn.Linear(n_pix, feature_space), nn.ReLU())
        )
    encoder = nn.Sequential(*encoder_layers)

    # LATENT SPACE
    mu_layer = nn.Linear(feature_space, latent_space_size)
    var_layer = nn.Linear(feature_space, latent_space_size)

    # DECODER
    decoder_layers = []
    # Intermediate feature space
    if feature_size == 0:
        decoder_layers.append(
            nn.Sequential(
                nn.Linear(latent_space_size, n_pix),
                nn.ReLU(),
            )
        )
    else:
        decoder_layers.append(
            nn.Sequential(
                nn.Linear(latent_space_size, feature_size),
                nn.ReLU(),
                nn.Linear(feature_size, n_pix),
                nn.ReLU(),
            )
        )
    # Unflatten
    decoder_layers.append(
        Unflatten3D(
            last_layer_channels * 2 ** (n_conv - 1),
            input_d // (2**n_conv),
            input_h // (2**n_conv),
            input_w // (2**n_conv),
        )
    )
    # Decoder layers
    for i in range(n_conv - 1, 0, -1):
        decoder_layers.append(
            DecoderLayer3D(
                last_layer_channels * 2 ** (i),
                last_layer_channels * 2 ** (i - 1),
                output_padding=decoder_output_padding[i],
            )
        )
    # Output layer
    decoder_layers.append(
        nn.Sequential(
            nn.ConvTranspose3d(
                last_layer_channels,
                input_c,
                4,
                stride=2,
                padding=1,
                output_padding=decoder_output_padding[0],
                bias=False,
            ),
            nn.Sigmoid(),
        )
    )
    decoder = nn.Sequential(*decoder_layers)
    return encoder, mu_layer, var_layer, decoder


class Encoder(BaseEncoder):
    def __init__(self, encoder_layers, mu_layer, logvar_layer): # Args is a ModelConfig instance
        BaseEncoder.__init__(self)

        self.layers = encoder_layers
        self.mu_layer = mu_layer
        self.logvar_layer = logvar_layer

    def forward(self, x:torch.Tensor) -> ModelOutput:
        h = self.layers(x)
        mu, logVar = self.mu_layer(h), self.logvar_layer(h)
        output = ModelOutput(
            embedding=mu, # Set the output from the encoder in a ModelOutput instance
            log_covariance=logVar
        )
        return output


class Decoder(BaseDecoder):
    def __init__(self, decoder_layers):
        BaseDecoder.__init__(self)

        self.layers = decoder_layers

    def forward(self, x:torch.Tensor) -> ModelOutput:
        out = self.layers(x)
        output = ModelOutput(
            reconstruction=out # Set the output from the decoder in a ModelOutput instance
        )
        return output