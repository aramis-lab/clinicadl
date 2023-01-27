from .autoencoder.models import AE_Conv4_FC3, AE_Conv5_FC3, CAE_half
from .cnn.models import Conv4_FC3, Conv5_FC3, Stride_Conv5_FC3, resnet18
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
from .pythae import (
    pythae_VAE,
    pythae_BetaVAE,
    pythae_VAE_LinNF,
    pythae_VAE_IAF,
    pythae_DisentangledBetaVAE,
    pythae_FactorVAE,
    pythae_BetaTCVAE,
    pythae_MSSSIM_VAE,
    pythae_INFOVAE_MMD,
    pythae_SVAE,
    pythae_PoincareVAE,
    pythae_Adversarial_AE,
    pythae_VAEGAN,
    pythae_VQVAE,
    pythae_HVAE,
    pythae_RHVAE,
    pythae_IWAE,
    pythae_CIWAE,
    pythae_PIWAE,
    pythae_MIWAE,
    pythae_VAMP,
    pythae_WAE_MMD,
    pythae_AE,
    pythae_RAE_L2,
    pythae_RAE_GP,
)
