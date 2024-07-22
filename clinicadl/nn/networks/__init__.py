from .ae import AE_Conv4_FC3, AE_Conv5_FC3, CAE_half
from .cnn import (
    Conv4_FC3,
    Conv5_FC3,
    ResNet3D,
    SqueezeExcitationCNN,
    Stride_Conv5_FC3,
    resnet18,
)
from .random import RandomArchitecture
from .ssda import Conv5_FC3_SSDA
from .unet import UNet
from .vae import (
    CVAE_3D,
    CVAE_3D_final_conv,
    CVAE_3D_half,
    VanillaDenseVAE,
    VanillaDenseVAE3D,
    VanillaSpatialVAE,
    VanillaSpatialVAE3D,
)
