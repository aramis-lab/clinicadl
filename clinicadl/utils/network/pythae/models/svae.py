from clinicadl.utils.network.vae.vae_layers import (
    EncoderLayer3D,
    Flatten,
)

from clinicadl.utils.network.pythae.pythae_utils import BasePythae
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput

import torch
from torch import nn


class pythae_SVAE(BasePythae):
    def __init__(
        self,
        input_size,
        latent_space_size,
        feature_size,
        n_conv,
        io_layer_channels,
        gpu=False,
    ):

        from pythae.models import SVAE, SVAEConfig

        _, decoder = super(pythae_SVAE, self).__init__(
            input_size=input_size,
            latent_space_size=latent_space_size,
            feature_size=feature_size,
            n_conv=n_conv,
            io_layer_channels=io_layer_channels,
            gpu=gpu
        )

        encoder_layers, mu_layer, log_concentration_layer = build_SVAE_encoder(
            input_size=input_size,
            latent_space_size=latent_space_size,
            feature_size=feature_size,
            n_conv=n_conv,
            io_layer_channels=io_layer_channels,
        )

        encoder = Encoder(encoder_layers, mu_layer, log_concentration_layer)

        model_config = SVAEConfig(
            input_dim=self.input_size,
            latent_dim=self.latent_space_size
        )
        self.model = SVAE(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder,
        )

    def get_trainer_config(self, output_dir, num_epochs, learning_rate, batch_size):
        from pythae.trainers import BaseTrainerConfig
        return BaseTrainerConfig(
            output_dir=output_dir,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size
        )


def build_SVAE_encoder(
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
    log_concentration_layer = nn.Linear(feature_space, 1)

    return encoder, mu_layer, log_concentration_layer

class Encoder(BaseEncoder):
    def __init__(self, encoder_layers, mu_layer, logc_layer): # Args is a ModelConfig instance
        BaseEncoder.__init__(self)

        self.layers = encoder_layers
        self.mu_layer = mu_layer
        self.log_concentration = logc_layer

    def forward(self, x:torch.Tensor) -> ModelOutput:
        h = self.layers(x)
        mu, log_concentration = self.mu_layer(h), self.log_concentration(h)
        output = ModelOutput(
            embedding=mu, # Set the output from the encoder in a ModelOutput instance
            log_concentration=log_concentration
        )
        return output
