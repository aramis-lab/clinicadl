from clinicadl.utils.network.pythae.pythae_utils import BasePythae

from pythae.models.nn import BaseDecoder, BaseDiscriminator
from pythae.models.base.base_utils import ModelOutput

from typing import List

import torch
import torch.nn as nn


class pythae_VAEGAN(BasePythae):
    def __init__(
        self,
        encoder_decoder_config,
        adversarial_loss_scale,
        reconstruction_layer,
        margin,
        equilibrium,
        gpu=False,
    ):

        from pythae.models import VAEGAN, VAEGANConfig

        encoder, decoder = super(pythae_VAEGAN, self).__init__(
            encoder_decoder_config = encoder_decoder_config,
            gpu=gpu,
        )

        discriminator = Discriminator_VAEGAN(input_size=self.input_size)

        model_config = VAEGANConfig(
            input_dim=self.input_size,
            latent_dim=self.latent_space_size,
            adversarial_loss_scale=adversarial_loss_scale,
            reconstruction_layer=reconstruction_layer,
            margin=margin,
            equilibrium=equilibrium,
        )
        self.model = VAEGAN(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder,
            discriminator=discriminator,
        )

    def get_trainer_config(self, output_dir, num_epochs, learning_rate, batch_size, optimizer):
        from pythae.trainers import CoupledOptimizerAdversarialTrainerConfig
        return CoupledOptimizerAdversarialTrainerConfig(
            output_dir=output_dir,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            optimizer_cls=optimizer,
        )


class Discriminator_VAEGAN(BaseDiscriminator):

    def __init__(self, input_size):

        BaseDiscriminator.__init__(self)

        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv3d(self.n_channels, 32, 4, 2, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv3d(32, 64, 4, 2, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv3d(64, 128, 4, 2, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv3d(128, 256, 4, 2, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(),
            )
        )

        n_pix = 256 * (input_size[1] // (2**4)) * (input_size[2] // (2**4)) * (input_size[3] // (2**4))
        layers.append(
            nn.Sequential(
                nn.Linear(n_pix, 512), 
                nn.BatchNorm1d(512), 
                nn.ReLU(),
            )
        )
        layers.append(
            nn.Sequential(
                nn.Linear(512, 1), 
                nn.Sigmoid(),
            )
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x
        print(out.shape)

        for i in range(max_depth):

            if i == 4:
                out = out.reshape(x.shape[0], -1)

            out = self.layers[i](out)
            print(out.shape)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = out

        return output
