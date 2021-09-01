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
        feature_size=1024,
        latent_size=64,
        n_conv=4,
        io_layer_channel=32,
        train=False,
    ):
        super(VanillaDenseVAE, self).__init__()

        self.input_size = input_size
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.n_conv = n_conv
        self.io_layer_channel = io_layer_channel

        self.training = train

        self.encoder = VAE_Encoder(
            input_shape=self.input_size,
            feature_size=self.feature_size,
            latent_dim=self.latent_dim,
            n_conv=self.n_conv,
            first_layer_channels=self.io_layer_channel,
        )

        # hidden => mu
        self.mu_layer = nn.Linear(self.feature_size, self.latent_size)
        # hidden => logvar
        self.var_layer = nn.Linear(self.feature_size, self.latent_size)

        self.decoder = VAE_Decoder(
            input_shape=self.input_size,
            latent_size=self.latent_size,
            feature_size=self.feature_size,
            latent_dim=self.latent_dim,
            n_conv=self.n_conv,
            last_layer_channels=self.io_layer_channel,
        )


class VanillaSpatialVAE(BaseVAE):
    def __init__(
        self,
        input_size,
        feature_size=1024,
        latent_size=64,
        n_conv=4,
        io_layer_channel=32,
        train=False,
    ):
        super(VanillaSpatialVAE, self).__init__()

        self.input_size = input_size
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.n_conv = n_conv
        self.io_layer_channel = io_layer_channel

        self.training = train

        self.encoder = VAE_Encoder(
            input_shape=self.input_size,
            feature_size=self.feature_size,
            latent_dim=self.latent_dim,
            n_conv=self.n_conv,
            first_layer_channels=self.io_layer_channel,
        )

        # hidden => mu
        self.mu_layer = nn.Conv2d(
            self.feature_size, self.latent_size, 3, stride=1, padding=1, bias=False
        )
        # hidden => logvar
        self.var_layer = nn.Conv2d(
            self.feature_size, self.latent_size, 3, stride=1, padding=1, bias=False
        )

        self.decoder = VAE_Decoder(
            input_shape=self.input_size,
            latent_size=self.latent_size,
            feature_size=self.feature_size,
            latent_dim=self.latent_dim,
            n_conv=self.n_conv,
            last_layer_channels=self.io_layer_channel,
        )


class Vanilla3DVAE(BaseVAE):
    def __init__(
        self,
        input_size,
        n_conv,
        first_layer_channels,
        last_layer_channels,
        feature_size,
        latent_size,
    ):

        self.input_size = input_size

        self.input_c = self.input_size[0]
        self.input_h = self.input_size[1]
        self.input_w = self.input_size[2]

        ## Encoder
        encoder_layers = []

        # Input Layer
        encoder_layers.append(EncoderLayer3D(self.input_c, first_layer_channels))

        # Conv Layers
        for i in range(n_conv - 1):
            encoder_layers.append(
                EncoderLayer3D(
                    first_layer_channels * 2 ** i, first_layer_channels * 2 ** (i + 1)
                )
            )

        encoder_layers.append(
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

        self.encoder = nn.Sequential(*encoder_layers)

        ## Latent space
        # hidden => mu
        self.mu_layer = nn.Conv3d(
            feature_size, latent_size, 3, stride=1, padding=1, bias=False
        )
        # hidden => logvar
        self.var_layer = nn.Conv3d(
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

        for i in range(n_conv - 1, 0, -1):
            decoder_layers.append(
                DecoderLayer3D(
                    last_layer_channels * 2 ** (i), last_layer_channels * 2 ** (i - 1)
                )
            )

        decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    last_layer_channels,
                    self.input_c,
                    4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.Sigmoid(),
            )
        )

        self.decoder = nn.Sequential(*self.decoder_layers)
