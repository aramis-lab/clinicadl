import torch
import torch.nn.functional as F
from torch import nn


class EncoderLayer2D(nn.Module):
    """
    Class defining the encoder's part of the Autoencoder.
    This layer is composed of one 2D convolutional layer,
    a batch normalization layer with a leaky relu
    activation function.
    """

    def __init__(
        self, input_channels, output_channels, kernel_size=4, stride=2, padding=1
    ):
        super(EncoderLayer2D, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
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
        self, input_channels, output_channels, kernel_size=4, stride=2, padding=1
    ):
        super(DecoderLayer2D, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, x):
        x = F.relu(self.layer(x), inplace=True)
        return x


class EncoderLayer3D(nn.Module):
    """
    Class defining the encoder's part of the Autoencoder.
    This layer is composed of one 3D convolutional layer,
    a batch normalization layer with a leaky relu
    activation function.
    """

    def __init__(
        self, input_channels, output_channels, kernel_size=4, stride=2, padding=1
    ):
        super(EncoderLayer3D, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(output_channels),
        )

    def forward(self, x):
        x = F.leaky_relu(self.layer(x), negative_slope=0.2, inplace=True)
        return x


class DecoderLayer3D(nn.Module):
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
    ):
        super(DecoderLayer3D, self).__init__()
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
            nn.BatchNorm3d(output_channels),
        )

    def forward(self, x):
        x = F.relu(self.layer(x), inplace=True)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, channel, height, width):
        super(Unflatten, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width

    def forward(self, input):
        return input.view(input.size(0), self.channel, self.height, self.width)


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

        self.layers = []

        # Input Layer
        self.layers.append(EncoderLayer2D(self.input_c, first_layer_channels))

        # Conv Layers
        for i in range(n_conv - 1):
            self.layers.append(
                EncoderLayer2D(
                    first_layer_channels * 2 ** i, first_layer_channels * 2 ** (i + 1)
                )
            )

        # Final Layer
        if latent_dim == 1:
            n_pix = (
                first_layer_channels
                * 2 ** (n_conv - 1)
                * (self.input_h // (2 ** n_conv))
                * (self.input_w // (2 ** n_conv))
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
        return self.sequential(x)


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
    ):
        """
        Feature size is the size of the vector if latent_dim=1
        or is the W/H of the output channels if laten_dim=2
        """
        super(VAE_Decoder, self).__init__()

        self.input_c = input_shape[0]
        self.input_h = input_shape[1]
        self.input_w = input_shape[2]

        self.layers = []

        if latent_dim == 1:
            n_pix = (
                last_layer_channels
                * 2 ** (n_conv - 1)
                * (self.input_h // (2 ** n_conv))
                * (self.input_w // (2 ** n_conv))
            )
            self.layers.append(
                nn.Sequential(
                    nn.Linear(latent_size, feature_size),
                    nn.ReLU(),
                    nn.Linear(feature_size, n_pix),
                    nn.ReLU(),
                    Unflatten(
                        last_layer_channels * 2 ** (n_conv - 1),
                        self.input_h // (2 ** n_conv),
                        self.input_w // (2 ** n_conv),
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
                    last_layer_channels * 2 ** (i), last_layer_channels * 2 ** (i - 1)
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
                    bias=False,
                ),
                nn.Sigmoid(),
            )
        )

        self.sequential = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.sequential(x)
