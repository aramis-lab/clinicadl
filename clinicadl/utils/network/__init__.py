from .autoencoder.models import AE_Conv4_FC3, AE_Conv5_FC3
from .cnn.models import Conv4_FC3, Conv5_FC3, resnet18
from .cnn.random import RandomArchitecture
from .vae.convolutional_VAE import CVAE_3D
from .vae.vanilla_vae import (
    Vanilla3DdenseVAE,
    Vanilla3DVAE,
    VanillaDenseVAE,
    VanillaSpatialVAE,
)
