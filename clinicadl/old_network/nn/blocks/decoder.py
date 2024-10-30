import torch.nn as nn
import torch.nn.functional as F

from clinicadl.nn.layers import Unflatten2D, get_norm_layer

__all__ = [
    "Decoder2D",
    "Decoder3D",
    "VAE_Decoder2D",
]


class Decoder2D(nn.Module):
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
        normalization="BatchNorm",
    ):
        super(Decoder2D, self).__init__()
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
            get_norm_layer(normalization, dim=2)(output_channels),
        )

    def forward(self, x):
        x = F.relu(self.layer(x), inplace=True)
        return x


class Decoder3D(nn.Module):
    """
    Class defining the decoder's part of the Autoencoder.
    This layer is composed of one 3D transposed convolutional layer,
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
        normalization="BatchNorm",
    ):
        super(Decoder3D, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose3d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=False,
            ),
            get_norm_layer(normalization, dim=3)(output_channels),
        )

    def forward(self, x):
        x = F.relu(self.layer(x), inplace=True)
        return x


class VAE_Decoder2D(nn.Module):
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
        super(VAE_Decoder2D, self).__init__()

        self.input_c = input_shape[0]
        self.input_h = input_shape[1]
        self.input_w = input_shape[2]

        if not padding:
            output_padding = [[0, 0] for _ in range(n_conv)]
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
                Decoder2D(
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
