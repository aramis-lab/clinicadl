import torch
from torch import nn

from clinicadl.utils.network.vae.base_vae import BaseVAE
from clinicadl.utils.network.vae.vae_layers import (
    DecoderLayer3D,
    EncoderLayer3D,
    Flatten,
    Unflatten3D,
    VAE_Decoder,
    VAE_Encoder,
)


class VanillaDenseVAE(BaseVAE):
    """
    This network is a 2D convolutional variational autoencoder with a dense latent space.

    reference: Diederik P Kingma et al., Auto-Encoding Variational Bayes.
    https://arxiv.org/abs/1312.6114
    """

    def __init__(
        self,
        input_size,
        latent_space_size,
        feature_size,
        recons_weight,
        kl_weight,
        gpu=True,
    ):
        n_conv = 4
        io_layer_channel = 32

        encoder = VAE_Encoder(
            input_size=input_size,
            feature_size=feature_size,
            latent_dim=1,
            n_conv=n_conv,
            first_layer_channels=io_layer_channel,
        )
        mu_layer = nn.Linear(feature_size, latent_space_size)
        var_layer = nn.Linear(feature_size, latent_space_size)
        decoder = VAE_Decoder(
            input_size=input_size,
            latent_size=latent_space_size,
            feature_size=feature_size,
            latent_dim=1,
            n_conv=n_conv,
            last_layer_channels=io_layer_channel,
            padding=encoder.decoder_padding,
        )

        super(VanillaDenseVAE, self).__init__(
            encoder,
            decoder,
            mu_layer,
            var_layer,
            latent_space_size,
            gpu=gpu,
            recons_weight=recons_weight,
            kl_weight=kl_weight,
            is_3D=False,
        )

    @staticmethod
    def get_input_size():
        return "1@128x128"

    @staticmethod
    def get_dimension():
        return "2D"

    @staticmethod
    def get_task():
        return ["reconstruction"]


class VanillaSpatialVAE(BaseVAE):
    """
    This network is a 3D convolutional variational autoencoder with a spacial latent space.

    reference: Diederik P Kingma et al., Auto-Encoding Variational Bayes.
    https://arxiv.org/abs/1312.6114
    """

    def __init__(
        self,
        input_size,
        latent_space_size,
        feature_size,
        recons_weight,
        kl_weight,
        gpu=True,
    ):
        feature_channels = 64
        latent_channels = 1
        n_conv = 4
        io_layer_channel = 32

        encoder = VAE_Encoder(
            input_size=input_size,
            feature_size=feature_channels,
            latent_dim=2,
            n_conv=n_conv,
            first_layer_channels=io_layer_channel,
        )
        mu_layer = nn.Conv2d(
            feature_channels, latent_channels, 3, stride=1, padding=1, bias=False
        )
        var_layer = nn.Conv2d(
            feature_channels, latent_channels, 3, stride=1, padding=1, bias=False
        )
        decoder = VAE_Decoder(
            input_size=input_size,
            latent_size=latent_channels,
            feature_size=feature_channels,
            latent_dim=2,
            n_conv=n_conv,
            last_layer_channels=io_layer_channel,
            padding=encoder.decoder_padding,
        )

        super(VanillaSpatialVAE, self).__init__(
            encoder,
            decoder,
            mu_layer,
            var_layer,
            latent_space_size,
            gpu=gpu,
            recons_weight=recons_weight,
            kl_weight=kl_weight,
            is_3D=False,
        )

    @staticmethod
    def get_input_size():
        return "1@128x128"

    @staticmethod
    def get_dimension():
        return "2D"

    @staticmethod
    def get_task():
        return ["reconstruction"]


class Vanilla3DspacialVAE(BaseVAE):
    """
    This network is a 3D convolutional variational autoencoder with a spacial latent space.

    reference: Diederik P Kingma et al., Auto-Encoding Variational Bayes.
    https://arxiv.org/abs/1312.6114
    """

    def __init__(
        self,
        input_size,
        latent_space_size,
        feature_size,
        recons_weight,
        kl_weight,
        gpu=True,
    ):
        n_conv = 4
        first_layer_channels = 32
        last_layer_channels = 32
        feature_channels = 512
        latent_channels = 1
        decoder_output_padding = [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ]

        input_c = input_size[0]
        # input_h = input_size[1]
        # input_w = input_size[2]

        # Encoder
        encoder_layers = []
        # Input Layer
        encoder_layers.append(EncoderLayer3D(input_c, first_layer_channels))
        # Conv Layers
        for i in range(n_conv - 1):
            encoder_layers.append(
                EncoderLayer3D(
                    first_layer_channels * 2**i, first_layer_channels * 2 ** (i + 1)
                )
            )
        encoder_layers.append(
            nn.Sequential(
                nn.Conv3d(
                    first_layer_channels * 2 ** (n_conv - 1),
                    feature_channels,
                    4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.ReLU(),
            )
        )
        encoder = nn.Sequential(*encoder_layers)

        # Latent space
        mu_layer = nn.Conv3d(
            feature_channels, latent_channels, 3, stride=1, padding=1, bias=False
        )
        var_layer = nn.Conv3d(
            feature_channels, latent_channels, 3, stride=1, padding=1, bias=False
        )

        # Decoder
        decoder_layers = []
        decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose3d(
                    latent_channels,
                    feature_channels,
                    3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.ReLU(),
                nn.ConvTranspose3d(
                    feature_channels,
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
                    output_padding=decoder_output_padding[i],
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

        super(Vanilla3DspacialVAE, self).__init__(
            encoder,
            decoder,
            mu_layer,
            var_layer,
            latent_space_size,
            gpu=gpu,
            recons_weight=recons_weight,
            kl_weight=kl_weight,
            is_3D=False,
        )

    @staticmethod
    def get_input_size():
        return "1@128x128x128"

    @staticmethod
    def get_dimension():
        return "3D"

    @staticmethod
    def get_task():
        return ["reconstruction"]


class Vanilla3DdenseVAE(BaseVAE):
    """
    This network is a 3D convolutional variational autoencoder with a dense latent space.

    reference: Diederik P Kingma et al., Auto-Encoding Variational Bayes.
    https://arxiv.org/abs/1312.6114
    """

    def __init__(
        self,
        size_reduction_factor,
        latent_space_size=256,
        feature_size=1024,
        n_conv=4,
        io_layer_channels=8,
        recons_weight=1,
        kl_weight=1,
        gpu=True,
    ):
        first_layer_channels = io_layer_channels
        last_layer_channels = io_layer_channels
        # automatically compute padding
        decoder_output_padding = []

        if size_reduction_factor == 2:
            self.input_size = [1, 80, 96, 80]
        elif size_reduction_factor == 3:
            self.input_size = [1, 56, 64, 56]
        elif size_reduction_factor == 4:
            self.input_size = [1, 40, 48, 40]
        elif size_reduction_factor == 5:
            self.input_size = [1, 32, 40, 32]

        input_c = self.input_size[0]
        input_d = self.input_size[1]
        input_h = self.input_size[2]
        input_w = self.input_size[3]
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

        super(Vanilla3DdenseVAE, self).__init__(
            encoder,
            decoder,
            mu_layer,
            var_layer,
            latent_space_size,
            gpu=gpu,
            is_3D=False,
            recons_weight=recons_weight,
            kl_weight=kl_weight,
        )

    @staticmethod
    def get_input_size():
        return "1@dxhxw"

    @staticmethod
    def get_dimension():
        return "3D"

    @staticmethod
    def get_task():
        return ["reconstruction"]
