from torch import nn

from clinicadl.utils.network.autoencoder.cnn_transformer import CNN_Transformer
from clinicadl.utils.network.cnn.models import Conv4_FC3, Conv5_FC3, resnet18
from clinicadl.utils.network.sub_network import AutoEncoder
from clinicadl.utils.network.vae.vae_layers import (
    DecoderLayer3D,
    EncoderLayer3D,
    Flatten,
    Unflatten3D,
)


class AE_Conv5_FC3(AutoEncoder):
    """
    Autoencoder derived from the convolutional part of CNN Conv5_FC3.
    """

    def __init__(self, input_size, gpu=True):
        # fmt: off
        cnn_model = Conv5_FC3(input_size=input_size, gpu=gpu)
        autoencoder = CNN_Transformer(cnn_model)
        # fmt: on
        super().__init__(
            encoder=autoencoder.encoder, decoder=autoencoder.decoder, gpu=gpu
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


class AE_Conv4_FC3(AutoEncoder):
    """
    Autoencoder derived from the convolutional part of CNN Conv4_FC3.
    """

    def __init__(self, input_size, gpu=True):
        # fmt: off
        cnn_model = Conv4_FC3(input_size=input_size, gpu=gpu)
        autoencoder = CNN_Transformer(cnn_model)
        # fmt: on
        super().__init__(
            encoder=autoencoder.encoder, decoder=autoencoder.decoder, gpu=gpu
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


class CAE_half(AutoEncoder):
    """
    3D Autoencoder derived from CVAE
    """

    def __init__(self, input_size, latent_space_size, gpu=True):
        # fmt: off
        self.encoder = nn.Sequential(
            EncoderLayer3D(1, 32, kernel_size=3),
            EncoderLayer3D(32, 64, kernel_size=3),
            EncoderLayer3D(64, 128, kernel_size=3),
            Flatten(),
            nn.Linear(153600, latent_space_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_space_size, 307200),
            Unflatten3D(256, 10, 12, 10),
            DecoderLayer3D(256, 128, kernel_size=3),
            DecoderLayer3D(128, 64, kernel_size=3),
            DecoderLayer3D(64, 1, kernel_size=3)
        )
        # fmt: on
        super(CAE_half, self).__init__(
            encoder=self.encoder, decoder=self.decoder, gpu=gpu
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
