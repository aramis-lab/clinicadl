from enum import Enum


class ImplementedNetwork(str, Enum):
    """Implemented neural networks in ClinicaDL."""

    MLP = "MLP"
    CONV_ENCODER = "ConvEncoder"
    CONV_DECODER = "ConvDecoder"
    CNN = "CNN"
    GENERATOR = "Generator"
    AE = "AutoEncoder"
    VAE = "VAE"
    DENSENET = "DenseNet"
    DENSENET_121 = "DenseNet121"
    DENSENET_161 = "DenseNet161"
    DENSENET_169 = "DenseNet169"
    DENSENET_201 = "DenseNet201"
    RESNET = "ResNet"
    RESNET_18 = "ResNet18"
    RESNET_34 = "ResNet34"
    RESNET_50 = "ResNet50"
    RESNET_101 = "ResNet101"
    RESNET_152 = "ResNet152"
    SE_RESNET = "SEResNet"
    SE_RESNET_50 = "SEResNet50"
    SE_RESNET_101 = "SEResNet101"
    SE_RESNET_152 = "SEResNet152"
    UNET = "UNet"
    ATT_UNET = "AttentionUNet"
    VIT = "ViT"
    VIT_B_16 = "ViTB16"
    VIT_B_32 = "ViTB32"
    VIT_L_16 = "ViTL16"
    VIT_L_32 = "ViTL32"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not implemented. Implemented neural networks are: "
            + ", ".join([repr(m.value) for m in cls])
        )
