from clinicadl.utils.network.vae.vae_layers import (
    Flatten,
)
from clinicadl.utils.network.vae.vae_blocks import EncoderBlock

from clinicadl.utils.network.pythae.pythae_utils import BasePythae
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput

import torch
from torch import nn


class pythae_SVAE(BasePythae):
    def __init__(
        self,
        encoder_decoder_config,
        gpu=False,
    ):

        from pythae.models import SVAE, SVAEConfig

        _, decoder = super(pythae_SVAE, self).__init__(
            encoder_decoder_config = encoder_decoder_config,
            gpu=gpu,
        )

        self.svae_encoder = build_SVAE_encoder(
            encoder_decoder_config=encoder_decoder_config
        )

        encoder = Encoder(
            self.svae_encoder.encoder,
            self.svae_encoder.mu_layer,
            self.svae_encoder.log_concentration_layer
        )

        model_config = SVAEConfig(
            input_dim=self.input_size,
            latent_dim=self.latent_space_size
        )
        self.model = SVAE(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder,
        )

    def get_trainer_config(self, output_dir, num_epochs, learning_rate, batch_size, optimizer):
        from pythae.trainers import BaseTrainerConfig
        return BaseTrainerConfig(
            output_dir=output_dir,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            optimizer_cls=optimizer,
        )


class build_SVAE_encoder():

    def __init__(self, encoder_decoder_config):

        self.input_size = encoder_decoder_config.input_size
        self.first_layer_channels = encoder_decoder_config.first_layer_channels
        self.n_block_encoder = encoder_decoder_config.n_block_encoder
        self.feature_size = encoder_decoder_config.feature_size
        self.latent_space_size = encoder_decoder_config.latent_space_size
        self.n_layer_per_block_encoder = encoder_decoder_config.n_layer_per_block_encoder
        self.block_type = encoder_decoder_config.block_type

        self.build_encoder()

    def build_encoder(self):
        input_c = self.input_size[0]
        input_d = self.input_size[1]
        input_h = self.input_size[2]
        input_w = self.input_size[3]

        # ENCODER
        encoder_layers = []
        
        # Input Layer
        encoder_layers.append(
            EncoderBlock(input_c, self.first_layer_channels, self.n_layer_per_block_encoder, self.block_type,)
        )

        # Conv Layers
        for i in range(self.n_block_encoder-1):
            encoder_layers.append(
                EncoderBlock(
                    self.first_layer_channels * 2**i, 
                    self.first_layer_channels * 2**(i+1), 
                    self.n_layer_per_block_encoder, 
                    self.block_type,
                )
            )
        # Construct output paddings

        enc_feature_c = self.first_layer_channels * 2 ** (self.n_block_encoder - 1)
        enc_feature_d = input_d // (2**self.n_block_encoder)
        enc_feature_h = input_h // (2**self.n_block_encoder)
        enc_feature_w = input_w // (2**self.n_block_encoder)

        # Compute size of the feature space
        n_pix_encoder = enc_feature_c * enc_feature_d * enc_feature_h * enc_feature_w

        # Flatten
        encoder_layers.append(Flatten())
        # Intermediate feature space
        if self.feature_size == 0:
            feature_space = n_pix_encoder
        else:
            feature_space = feature_size
            encoder_layers.append(
                nn.Sequential(nn.Linear(n_pix_encoder, feature_space), nn.ReLU())
            )

        self.encoder = nn.Sequential(*encoder_layers)

        # LATENT SPACE
        self.mu_layer = nn.Linear(feature_space, self.latent_space_size)
        self.log_concentration_layer = nn.Linear(feature_space, 1)


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
