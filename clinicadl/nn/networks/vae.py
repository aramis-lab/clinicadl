import torch
import torch.nn as nn

from clinicadl.nn.blocks import (
    Decoder3D,
    Encoder3D,
    VAE_Decoder2D,
    VAE_Encoder2D,
)
from clinicadl.nn.layers import Unflatten3D
from clinicadl.nn.utils import multiply_list
from clinicadl.utils.enum import BaseEnum


class VAE2d(str, BaseEnum):
    """VAEs compatible with 2D inputs."""

    VANILLA_DENSE_VAE = "VanillaDenseVAE"
    VANILLA_SPATIAL_VAE = "VanillaSpatialVAE"


class VAE3d(str, BaseEnum):
    """VAEs compatible with 3D inputs."""

    VANILLA_DENSE_VAE3D = "VanillaSpatialVAE3D"
    VANILLA_SPATIAL_VAE3D = "VanillaDenseVAE3D"
    CVAE_3D_FINAL_CONV = "CVAE_3D_final_conv"
    CVAE_3D = "CVAE_3D"
    CVAE_3D_HALF = "CVAE_3D_half"


class ImplementedVAE(str, BaseEnum):
    """Implemented VAEs in ClinicaDL."""

    VANILLA_DENSE_VAE = "VanillaDenseVAE"
    VANILLA_SPATIAL_VAE = "VanillaSpatialVAE"
    VANILLA_DENSE_VAE3D = "VanillaDenseVAE3D"
    VANILLA_SPATIAL_VAE3D = "VanillaSpatialVAE3D"
    CVAE_3D_FINAL_CONV = "CVAE_3D_final_conv"
    CVAE_3D = "CVAE_3D"
    CVAE_3D_HALF = "CVAE_3D_half"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not implemented. Implemented VAEs are: "
            + ", ".join([repr(m.value) for m in cls])
        )


class VAE(nn.Module):
    def __init__(self, encoder, decoder, mu_layers, log_var_layers):
        super().__init__()
        self.encoder = encoder
        self.mu_layers = mu_layers
        self.log_var_layers = log_var_layers
        self.decoder = decoder

    def encode(self, image):
        feature = self.encoder(image)
        mu = self.mu_layers(feature)
        log_var = self.log_var_layers(feature)
        return mu, log_var

    def decode(self, encoded):
        reconstructed = self.decoder(encoded)
        return reconstructed

    @staticmethod
    def _sample(mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image):
        mu, log_var = self.encode(image)
        if self.training:
            encoded = self._sample(mu, log_var)
        else:
            encoded = mu
        reconstructed = self.decode(encoded)
        return mu, log_var, reconstructed


class VanillaDenseVAE(VAE):
    """
    This network is a 2D convolutional variational autoencoder with a dense latent space.

    reference: Diederik P Kingma et al., Auto-Encoding Variational Bayes.
    https://arxiv.org/abs/1312.6114
    """

    def __init__(self, input_size, latent_space_size, feature_size):
        n_conv = 4
        io_layer_channel = 32

        encoder = VAE_Encoder2D(
            input_shape=input_size,
            feature_size=feature_size,
            latent_dim=1,
            n_conv=n_conv,
            first_layer_channels=io_layer_channel,
        )
        mu_layers = nn.Linear(feature_size, latent_space_size)
        log_var_layers = nn.Linear(feature_size, latent_space_size)
        decoder = VAE_Decoder2D(
            input_shape=input_size,
            latent_size=latent_space_size,
            feature_size=feature_size,
            latent_dim=1,
            n_conv=n_conv,
            last_layer_channels=io_layer_channel,
            padding=encoder.decoder_padding,
        )
        super().__init__(encoder, decoder, mu_layers, log_var_layers)


class VanillaDenseVAE3D(VAE):
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
    ):
        first_layer_channels = io_layer_channels
        last_layer_channels = io_layer_channels
        # automatically compute padding
        decoder_output_padding = []

        if (
            size_reduction_factor == 2
        ):  # TODO : specify that it only works with certain images
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
        encoder_layers.append(Encoder3D(input_c, first_layer_channels))
        decoder_output_padding.append([d % 2, h % 2, w % 2])
        d, h, w = d // 2, h // 2, w // 2
        # Conv Layers
        for i in range(n_conv - 1):
            encoder_layers.append(
                Encoder3D(
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
        encoder_layers.append(nn.Flatten())
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
        mu_layers = nn.Linear(feature_space, latent_space_size)
        log_var_layers = nn.Linear(feature_space, latent_space_size)

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
                Decoder3D(
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

        super().__init__(encoder, decoder, mu_layers, log_var_layers)


class VanillaSpatialVAE(VAE):
    """
    This network is a 2D convolutional variational autoencoder with a spatial latent space.

    reference: Diederik P Kingma et al., Auto-Encoding Variational Bayes.
    https://arxiv.org/abs/1312.6114
    """

    def __init__(
        self,
        input_size,
    ):
        feature_channels = 64
        latent_channels = 1
        n_conv = 4
        io_layer_channel = 32

        encoder = VAE_Encoder2D(
            input_shape=input_size,
            feature_size=feature_channels,
            latent_dim=2,
            n_conv=n_conv,
            first_layer_channels=io_layer_channel,
        )
        mu_layers = nn.Conv2d(
            feature_channels, latent_channels, 3, stride=1, padding=1, bias=False
        )
        log_var_layers = nn.Conv2d(
            feature_channels, latent_channels, 3, stride=1, padding=1, bias=False
        )
        decoder = VAE_Decoder2D(
            input_shape=input_size,
            latent_size=latent_channels,
            feature_size=feature_channels,
            latent_dim=2,
            n_conv=n_conv,
            last_layer_channels=io_layer_channel,
            padding=encoder.decoder_padding,
        )
        super().__init__(encoder, decoder, mu_layers, log_var_layers)


class VanillaSpatialVAE3D(VAE):
    """
    This network is a 3D convolutional variational autoencoder with a spatial latent space.

    reference: Diederik P Kingma et al., Auto-Encoding Variational Bayes.
    https://arxiv.org/abs/1312.6114
    """

    def __init__(self, input_size):
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

        encoder_layers = []
        encoder_layers.append(Encoder3D(input_c, first_layer_channels))
        for i in range(n_conv - 1):
            encoder_layers.append(
                Encoder3D(
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
        mu_layers = nn.Conv3d(
            feature_channels, latent_channels, 3, stride=1, padding=1, bias=False
        )
        log_var_layers = nn.Conv3d(
            feature_channels, latent_channels, 3, stride=1, padding=1, bias=False
        )
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
                Decoder3D(
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
        super().__init__(encoder, decoder, mu_layers, log_var_layers)


class CVAE_3D_final_conv(VAE):
    """
    This is the convolutional autoencoder whose main objective is to project the MRI into a smaller space
    with the sole criterion of correctly reconstructing the data. Nothing longitudinal here.
    fc = final layer conv
    """

    def __init__(self, size_reduction_factor, latent_space_size):
        n_conv = 3

        if size_reduction_factor == 2:
            self.input_size = [1, 80, 96, 80]
        elif size_reduction_factor == 3:
            self.input_size = [1, 56, 64, 56]
        elif size_reduction_factor == 4:
            self.input_size = [1, 40, 48, 40]
        elif size_reduction_factor == 5:
            self.input_size = [1, 32, 40, 32]
        feature_size = int(multiply_list(self.input_size[1:], 2**n_conv) * 128)

        encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=2, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Flatten(start_dim=1),
        )
        mu_layers = nn.Sequential(
            nn.Linear(feature_size, latent_space_size),
            nn.Tanh(),
        )
        log_var_layers = nn.Linear(feature_size, latent_space_size)
        decoder = nn.Sequential(
            nn.Linear(latent_space_size, 2 * feature_size),
            nn.LeakyReLU(),
            nn.Unflatten(
                dim=1,
                unflattened_size=(
                    256,
                    self.input_size[1] // 2**n_conv,
                    self.input_size[2] // 2**n_conv,
                    self.input_size[3] // 2**n_conv,
                ),
            ),
            nn.ConvTranspose3d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(64, 1, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm3d(1),
            nn.LeakyReLU(),
            nn.Conv3d(1, 1, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        super().__init__(encoder, decoder, mu_layers, log_var_layers)


class CVAE_3D(VAE):
    """
    This is the convolutional autoencoder whose main objective is to project the MRI into a smaller space
    with the sole criterion of correctly reconstructing the data. Nothing longitudinal here.
    """

    def __init__(self, latent_space_size):  # TODO : only work with 1-channel input
        encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )
        mu_layers = nn.Sequential(
            nn.Linear(1683968, latent_space_size),
            nn.Tanh(),
        )
        log_var_layers = nn.Linear(1683968, latent_space_size)
        decoder = nn.Sequential(
            nn.Linear(latent_space_size, 3367936),
            nn.ReLU(),
            nn.Unflatten(
                dim=1,
                unflattened_size=(
                    256,
                    22,
                    26,
                    23,
                ),
            ),
            nn.ConvTranspose3d(
                256, 128, 3, stride=2, padding=1, output_padding=[0, 1, 0]
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(
                128, 64, 3, stride=2, padding=1, output_padding=[0, 1, 1]
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 1, 3, stride=2, padding=1, output_padding=[0, 1, 0]),
            nn.ReLU(),
        )
        super().__init__(encoder, decoder, mu_layers, log_var_layers)


class CVAE_3D_half(VAE):
    """
    This is the convolutional autoencoder whose main objective is to project the MRI into a smaller space
    with the sole criterion of correctly reconstructing the data. Nothing longitudinal here.
    """

    def __init__(self, size_reduction_factor, latent_space_size):
        n_conv = 3
        if size_reduction_factor == 2:
            self.input_size = [1, 80, 96, 80]
        elif size_reduction_factor == 3:
            self.input_size = [1, 56, 64, 56]
        elif size_reduction_factor == 4:
            self.input_size = [1, 40, 48, 40]
        elif size_reduction_factor == 5:
            self.input_size = [1, 32, 40, 32]
        feature_size = int(multiply_list(self.input_size[1:], 2**n_conv) * 128)

        encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=2, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Flatten(start_dim=1),
        )
        mu_layers = nn.Sequential(
            nn.Linear(feature_size, latent_space_size),
            nn.Tanh(),
        )
        log_var_layers = nn.Linear(feature_size, latent_space_size)
        decoder = nn.Sequential(
            nn.Linear(latent_space_size, 2 * feature_size),
            nn.ReLU(),
            nn.Unflatten(
                dim=1,
                unflattened_size=(
                    256,
                    self.input_size[1] // 2**n_conv,
                    self.input_size[2] // 2**n_conv,
                    self.input_size[3] // 2**n_conv,
                ),
            ),
            nn.ConvTranspose3d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
        super().__init__(encoder, decoder, mu_layers, log_var_layers)
