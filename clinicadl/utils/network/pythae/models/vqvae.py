from clinicadl.utils.network.vae.vae_layers import (
    EncoderLayer3D,
    Flatten,
)

from clinicadl.utils.network.pythae.pythae_utils import BasePythae
from pythae.models.nn import BaseEncoder
from pythae.models.base.base_utils import ModelOutput

import torch
from torch import nn


class pythae_VQVAE(BasePythae):
    def __init__(
        self,
        input_size,
        latent_space_size,
        feature_size,
        n_conv,
        io_layer_channels,
        commitment_loss_factor,
        quantization_loss_factor,
        num_embeddings,
        use_ema,
        decay,
        gpu=False,
    ):

        from pythae.models import VQVAE, VQVAEConfig

        encoder, decoder = super(pythae_VQVAE, self).__init__(
            input_size=input_size,
            latent_space_size=latent_space_size,
            feature_size=feature_size,
            n_conv=n_conv,
            io_layer_channels=io_layer_channels,
            gpu=gpu,
            is_ae=True
        )

        # encoder_layers, emb_layer = build_VQVAE_encoder(
        #     input_size=input_size,
        #     latent_space_size=latent_space_size,
        #     feature_size=feature_size,
        #     n_conv=n_conv,
        #     io_layer_channels=io_layer_channels,
        # )

        # encoder = Encoder(encoder_layers, emb_layer)

        model_config = VQVAEConfig(
            input_dim=self.input_size,
            latent_dim=self.latent_space_size,
            commitment_loss_factor=commitment_loss_factor,
            quantization_loss_factor=quantization_loss_factor,
            num_embeddings=num_embeddings,
            use_ema=use_ema,
            decay=decay,
        )
        self.model = VQVAE(
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


def build_VQVAE_encoder(
    input_size = (1, 80, 96, 80),
    latent_space_dim=16,
    feature_size=0,
    n_conv=3,
    io_layer_channels=32,
):
    first_layer_channels = io_layer_channels
    last_layer_channels = io_layer_channels

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
    # # Compute size of the feature space
    # n_pix = (
    #     first_layer_channels
    #     * 2 ** (n_conv - 1)
    #     * (input_d // (2**n_conv))
    #     * (input_h // (2**n_conv))
    #     * (input_w // (2**n_conv))
    # )
    # # Flatten
    # encoder_layers.append(Flatten())
    # # Intermediate feature space
    # if feature_size == 0:
    #     feature_space = n_pix
    # else:
    #     feature_space = feature_size
    #     encoder_layers.append(
    #         nn.Sequential(nn.Linear(n_pix, feature_space), nn.ReLU())
    #     )
    # encoder = nn.Sequential(*encoder_layers)

    # LATENT SPACE
    pre_qantized = nn.Conv3d(first_layer_channels * 2 ** (n_conv + 1), latent_space_dim, 1, 1)

    return encoder, pre_qantized


class Encoder(BaseEncoder):
    def __init__(self, encoder_layers, pre_qantized): # Args is a ModelConfig instance
        BaseEncoder.__init__(self)

        self.layers = encoder_layers
        self.pre_qantized = pre_qantized

    def forward(self, x:torch.Tensor) -> ModelOutput:
        out = self.layers(x)
        output = ModelOutput(
            embedding=self.pre_qantized(out)
        )
        return output
