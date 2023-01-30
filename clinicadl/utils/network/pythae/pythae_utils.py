import abc
from collections import OrderedDict

from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput

from clinicadl.utils.network.network import Network
from clinicadl.utils.network.vae.vae_layers import (
    DecoderLayer3D,
    EncoderLayer3D,
    Flatten,
    Unflatten3D,
)

import torch
from torch import nn


class BasePythae(Network):
    def __init__(
        self,
        input_size,
        latent_space_size,
        feature_size,
        n_conv,
        io_layer_channels,
        gpu,
        is_ae=False,
    ):
        super(BasePythae, self).__init__(gpu=gpu)

        #self.model = None

        self.input_size = input_size
        self.latent_space_size = latent_space_size

        encoder_layers, mu_layer, logvar_layer, decoder_layers = build_encoder_decoder(
            input_size=input_size,
            latent_space_size=latent_space_size,
            feature_size=feature_size,
            n_conv=n_conv,
            io_layer_channels=io_layer_channels,
        )

        if is_ae:
            encoder = Encoder_AE(encoder_layers, mu_layer)
        else:
            encoder = Encoder_VAE(encoder_layers, mu_layer, logvar_layer)
        decoder = Decoder(decoder_layers)

        return encoder, decoder

    @abc.abstractmethod
    def get_model(self, encoder, decoder):
        pass

    @property
    def layers(self):
        return torch.nn.Sequential(
            self.model.encoder, self.model.decoder
        )

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=False):
        #x = input_dict["image"].to(self.device)
        model_outputs = self.forward(input_dict)
        loss_dict = {
            "loss": model_outputs.loss, 
        }
        for key in model_outputs.keys():
            if "loss" in key:
                loss_dict[key] = model_outputs[key]
        return model_outputs.recon_x, loss_dict

    # Network specific
    def predict(self, x):
        return self.model.predict(x)

    def forward(self, x):
        return self.model.forward(x)

    def transfer_weights(self, state_dict, transfer_class):
        self.model.load_state_dict(state_dict)

    # VAE specific
    # def encode(self, x):
    #     encoder_output = self.encoder(x)
    #     mu, logVar = encoder_output.embedding, encoder_output.log_covariance
    #     z = self.reparameterize(mu, logVar)
    #     return mu, logVar, z

    # def decode(self, z):
    #     z = self.decoder(z)
    #     return z

    # def reparameterize(self, mu, logVar):
    #     #print(logVar.shape)
    #     std = torch.exp(0.5 * logVar)
    #     eps = torch.randn_like(std)
    #     return eps.mul(std).add_(mu)


def build_encoder_decoder(
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
    return encoder, mu_layer, var_layer, decoder


class Encoder_VAE(BaseEncoder):
    def __init__(self, encoder_layers, mu_layer, logvar_layer): # Args is a ModelConfig instance
        BaseEncoder.__init__(self)

        self.layers = encoder_layers
        self.mu_layer = mu_layer
        self.logvar_layer = logvar_layer

    def forward(self, x:torch.Tensor) -> ModelOutput:
        h = self.layers(x)
        mu, logVar = self.mu_layer(h), self.logvar_layer(h)
        output = ModelOutput(
            embedding=mu, # Set the output from the encoder in a ModelOutput instance
            log_covariance=logVar
        )
        return output


class Encoder_AE(BaseEncoder):
    def __init__(self, encoder_layers, mu_layer): # Args is a ModelConfig instance
        BaseEncoder.__init__(self)

        self.layers = encoder_layers
        self.mu_layer = mu_layer

    def forward(self, x:torch.Tensor) -> ModelOutput:
        h = self.layers(x)
        embedding = self.mu_layer(h)
        output = ModelOutput(
            embedding=embedding, # Set the output from the encoder in a ModelOutput instance
        )
        return output


class Decoder(BaseDecoder):
    def __init__(self, decoder_layers):
        BaseDecoder.__init__(self)

        self.layers = decoder_layers

    def forward(self, x:torch.Tensor) -> ModelOutput:
        out = self.layers(x)
        output = ModelOutput(
            reconstruction=out # Set the output from the decoder in a ModelOutput instance
        )
        return output
