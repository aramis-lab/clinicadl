import torch
from torch import nn

from clinicadl.utils.network.vae.base_vae import BaseVAE
from clinicadl.utils.network.vae.vae_utils import (
    DecoderLayer3D,
    EncoderLayer3D,
    VAE_Decoder,
    VAE_Encoder,
)


class VanillaDenseVAE(BaseVAE):
    def __init__(
        self,
        input_size,
        use_cpu=False,
        feature_size=1024,
        latent_size=64,
        n_conv=4,
        io_layer_channel=32,
    ):
        encoder = VAE_Encoder(
            input_shape=input_size,
            feature_size=feature_size,
            latent_dim=1,
            n_conv=n_conv,
            first_layer_channels=io_layer_channel,
        )
        mu_layer = nn.Linear(feature_size, latent_size)
        var_layer = nn.Linear(feature_size, latent_size)
        decoder = VAE_Decoder(
            input_shape=input_size,
            latent_size=latent_size,
            feature_size=feature_size,
            latent_dim=1,
            n_conv=n_conv,
            last_layer_channels=io_layer_channel,
        )

        super(VanillaDenseVAE, self).__init__(
            encoder, decoder, mu_layer, var_layer, use_cpu=use_cpu
        )


class VanillaSpatialVAE(BaseVAE):
    def __init__(
        self,
        input_size,
        use_cpu=False,
        feature_size=1024,
        latent_size=64,
        n_conv=4,
        io_layer_channel=32,
    ):
        encoder = VAE_Encoder(
            input_shape=input_size,
            feature_size=feature_size,
            latent_dim=2,
            n_conv=n_conv,
            first_layer_channels=io_layer_channel,
        )
        mu_layer = nn.Conv2d(
            feature_size, latent_size, 3, stride=1, padding=1, bias=False
        )
        var_layer = nn.Conv2d(
            feature_size, latent_size, 3, stride=1, padding=1, bias=False
        )
        decoder = VAE_Decoder(
            input_shape=input_size,
            latent_size=latent_size,
            feature_size=feature_size,
            latent_dim=2,
            n_conv=n_conv,
            last_layer_channels=io_layer_channel,
        )

        super(VanillaSpatialVAE, self).__init__(
            encoder, decoder, mu_layer, var_layer, use_cpu=use_cpu
        )


class Vanilla3DVAE(BaseVAE):
    def __init__(
        self,
        input_size,
        use_cpu=False,
    ):
        n_conv = 4
        first_layer_channels = 32
        last_layer_channels = 32
        feature_size = 512
        latent_size = 1
        decoder_output_padding = [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ]

        input_c = input_size[0]
        # input_h = input_size[1]
        # input_w = input_size[2]

        ## Encoder
        encoder_layers = []
        # Input Layer
        encoder_layers.append(EncoderLayer3D(input_c, first_layer_channels))
        # Conv Layers
        for i in range(n_conv - 1):
            encoder_layers.append(
                EncoderLayer3D(
                    first_layer_channels * 2 ** i, first_layer_channels * 2 ** (i + 1)
                )
            )
        encoder_layers.append(
            nn.Sequential(
                nn.Conv3d(
                    first_layer_channels * 2 ** (n_conv - 1),
                    feature_size,
                    4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.ReLU(),
            )
        )
        encoder = nn.Sequential(*encoder_layers)

        ## Latent space
        # hidden => mu
        mu_layer = nn.Conv3d(
            feature_size, latent_size, 3, stride=1, padding=1, bias=False
        )
        # hidden => logvar
        var_layer = nn.Conv3d(
            feature_size, latent_size, 3, stride=1, padding=1, bias=False
        )

        ## Decoder
        decoder_layers = []
        decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose3d(
                    latent_size, feature_size, 3, stride=1, padding=1, bias=False
                ),
                nn.ReLU(),
                nn.ConvTranspose3d(
                    feature_size,
                    last_layer_channels * 2 ** (n_conv - 1),
                    4,
                    stride=2,
                    padding=1,
                    output_padding=[0, 1, 1],
                    bias=False,
                ),
                nn.ReLU(),
            )
        )
        for i in range(n_conv - 1, 0, -1):
            decoder_layers.append(
                DecoderLayer3D(
                    last_layer_channels * 2 ** (i),
                    last_layer_channels * 2 ** (i - 1),
                    output_padding=decoder_output_padding[-i],
                )
            )
        decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose3d(
                    last_layer_channels,
                    input_c,
                    4,
                    stride=2,
                    padding=1,
                    output_padding=[1, 0, 1],
                    bias=False,
                ),
                nn.Sigmoid(),
            )
        )
        decoder = nn.Sequential(*decoder_layers)

        super(Vanilla3DVAE, self).__init__(
            encoder, decoder, mu_layer, var_layer, use_cpu=use_cpu
        )
