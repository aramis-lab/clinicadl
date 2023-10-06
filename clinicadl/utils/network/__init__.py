from .autoencoder.models import AE_Conv4_FC3, AE_Conv5_FC3, CAE_half
from .cnn.models import (
    Conv4_FC3,
    Conv5_FC3,
    Conv5_FC3_SSDA,
    ResNet3D,
    SqueezeExcitationCNN,
    Stride_Conv5_FC3,
    resnet18,
)
from .cnn.random import RandomArchitecture
from .unet.unet import GeneratorUNet
from .vae.advanced_CVAE import CVAE_3D_final_conv
from .vae.convolutional_VAE import CVAE_3D, CVAE_3D_half
from .vae.vanilla_vae import (
    Vanilla3DdenseVAE,
    Vanilla3DspacialVAE,
    VanillaDenseVAE,
    VanillaSpatialVAE,
)
