import abc
from collections import OrderedDict

from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput

from clinicadl.utils.network.network import Network
from clinicadl.utils.network.vae.vae_layers import (
    Flatten,
    Unflatten3D,
)
from clinicadl.utils.network.vae.vae_blocks import EncoderBlock, DecoderBlock

import torch
from torch import nn


class BasePythae(Network):
    def __init__(
        self,
        encoder_decoder_config,
        gpu,
        is_ae=False,
    ):
        super(BasePythae, self).__init__(gpu=gpu)

        self.input_size = encoder_decoder_config.input_size
        self.latent_space_size = encoder_decoder_config.latent_space_size

        self.encoder_decoder = Encoder_Decoder(encoder_decoder_config)

        if is_ae:
            encoder = Encoder_AE(self.encoder_decoder.encoder, self.encoder_decoder.mu_layer)
        else:
            encoder = Encoder_VAE(
                self.encoder_decoder.encoder,
                self.encoder_decoder.mu_layer,
                self.encoder_decoder.var_layer
            )
        decoder = Decoder(self.encoder_decoder.decoder)

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
        # x = input_dict["image"].to(self.device)
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
        return self.model.predict(x.data)

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


class Encoder_Decoder():

    def __init__(self, encoder_decoder_config):

        self.input_size = encoder_decoder_config.input_size
        self.first_layer_channels = encoder_decoder_config.first_layer_channels
        self.n_block_encoder = encoder_decoder_config.n_block_encoder
        self.feature_size = encoder_decoder_config.feature_size
        self.latent_space_size = encoder_decoder_config.latent_space_size
        self.n_block_decoder = encoder_decoder_config.n_block_decoder
        self.last_layer_channels = encoder_decoder_config.last_layer_channels
        self.last_layer_conv = encoder_decoder_config.last_layer_conv
        self.n_layer_per_block_encoder = encoder_decoder_config.n_layer_per_block_encoder
        self.n_layer_per_block_decoder = encoder_decoder_config.n_layer_per_block_decoder
        self.block_type = encoder_decoder_config.block_type

        self.build_encoder_decoder()

    def build_encoder_decoder(self):
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
        self.var_layer = nn.Linear(feature_space, self.latent_space_size)

        # DECODER

        # automatically compute output padding and image size
        d, h, w = input_d, input_h, input_w
        decoder_output_padding = []
        decoder_input_size = []
        decoder_output_padding.append([d % 2, h % 2, w % 2])
        d, h, w = d // 2, h // 2, w // 2
        decoder_input_size.append([d, h, w])
        for i in range(self.n_block_decoder - 1):
            decoder_output_padding.append([d % 2, h % 2, w % 2])
            d, h, w = d // 2, h // 2, w // 2
            decoder_input_size.append([d, h, w])

        dec_feature_c = self.last_layer_channels * 2 ** (self.n_block_decoder - 1)
        dec_feature_d = input_d // (2**self.n_block_decoder)
        dec_feature_h = input_h // (2**self.n_block_decoder)
        dec_feature_w = input_w // (2**self.n_block_decoder)

        n_pix_decoder = dec_feature_c * dec_feature_d * dec_feature_h * dec_feature_w

        decoder_layers = []
        # Intermediate feature space
        if self.feature_size == 0:
            decoder_layers.append(
                nn.Sequential(
                    nn.Linear(self.latent_space_size, n_pix_decoder),
                    nn.ReLU(),
                )
            )
        else:
            decoder_layers.append(
                nn.Sequential(
                    nn.Linear(self.latent_space_size, self.feature_size),
                    nn.ReLU(),
                    nn.Linear(self.feature_size, n_pix_decoder),
                    nn.ReLU(),
                )
            )
        # Unflatten
        decoder_layers.append(
            Unflatten3D(dec_feature_c, dec_feature_d, dec_feature_h, dec_feature_w)
        )

        # Decoder layers
        for i in range(self.n_block_decoder-1, 0, -1):
            decoder_layers.append(
                DecoderBlock(
                    self.last_layer_channels * 2 ** (i), 
                    self.last_layer_channels * 2 ** (i-1), 
                    decoder_input_size[i],
                    decoder_output_padding[i],
                    self.n_layer_per_block_decoder,
                    self.block_type,
                ),
            )

        # Output layer
        decoder_layers.append(
            DecoderBlock(
                self.last_layer_channels, 
                input_c, 
                decoder_input_size[0], 
                decoder_output_padding[0], 
                self.n_layer_per_block_decoder,
                self.block_type,
                is_last_block=True, 
                last_layer_conv=self.last_layer_conv,
            )
        )

        self.decoder = nn.Sequential(*decoder_layers)


class Encoder_VAE(BaseEncoder):
    def __init__(self, encoder_layers, mu_layer, logvar_layer): # Args is a ModelConfig instance
        BaseEncoder.__init__(self)

        self.layers = encoder_layers
        self.mu_layer = mu_layer
        self.logvar_layer = logvar_layer

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h = self.layers(x)
        mu, logVar = self.mu_layer(h), self.logvar_layer(h)
        output = ModelOutput(
            embedding=mu,  # Set the output from the encoder in a ModelOutput instance
            log_covariance=logVar,
        )
        return output


class Encoder_AE(BaseEncoder):
    def __init__(self, encoder_layers, mu_layer): # Args is a ModelConfig instance
        BaseEncoder.__init__(self)

        self.layers = encoder_layers
        self.mu_layer = mu_layer

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h = self.layers(x)
        embedding = self.mu_layer(h)
        output = ModelOutput(
            embedding=embedding,  # Set the output from the encoder in a ModelOutput instance
        )
        return output


class Decoder(BaseDecoder):
    def __init__(self, decoder_layers):
        BaseDecoder.__init__(self)

        self.layers = decoder_layers

    def forward(self, x: torch.Tensor) -> ModelOutput:
        out = self.layers(x)
        output = ModelOutput(
            reconstruction=out  # Set the output from the decoder in a ModelOutput instance
        )
        return output