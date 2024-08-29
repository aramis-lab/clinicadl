from enum import Enum


class ImplementedNetworks(str, Enum):
    """Implemented neural networks in ClinicaDL."""

    REGRESSOR = "Regressor"
    CLASSIFIER = "Classifier"
    DISCRIMINATOR = "Discriminator"
    CRITIC = "Critic"
    AE = "AutoEncoder"
    VAE = "VarAutoEncoder"
    DENSE_NET = "DenseNet"
    FCN = "FullyConnectedNet"
    VAR_FCN = "VarFullyConnectedNet"
    GENERATOR = "Generator"
    RES_NET = "ResNet"
    RES_NET_FEATURES = "ResNetFeatures"
    SEG_RES_NET = "SegResNet"
    UNET = "UNet"
    ATT_UNET = "AttentionUnet"
    VIT = "ViT"
    VIT_AE = "ViTAutoEnc"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not implemented. Implemented neural networks are: "
            + ", ".join([repr(m.value) for m in cls])
        )


class CaseInsensitiveEnum(str, Enum):
    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.lower()
        for member in cls:
            if member.lower() == value:
                return member
        return None


class ImplementedActFunctions(CaseInsensitiveEnum):
    """Supported activation functions in ClinicaDL."""

    ELU = "elu"
    RELU = "relu"
    LEAKY_RELU = "leakyrelu"
    PRELU = "prelu"
    RELU6 = "relu6"
    SELU = "selu"
    CELU = "celu"
    GELU = "gelu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    LOGSOFTMAX = "logsoftmax"
    SWISH = "swish"
    MEMSWISH = "memswish"
    MISH = "mish"
    GEGLU = "geglu"


class ImplementedNormLayers(CaseInsensitiveEnum):
    """Supported normalization layers in ClinicaDL."""

    GROUP = "group"
    LAYER = "layer"
    LOCAL_RESPONSE = "localresponse"
    SYNCBATCH = "syncbatch"
    INSTANCE_NVFUSER = "instance_nvfuser"
    BATCH = "batch"
    INSTANCE = "instance"


class ResNetBlocks(str, Enum):
    """Supported ResNet blocks."""

    BASIC = "basic"
    BOTTLENECK = "bottleneck"


class ShortcutTypes(str, Enum):
    """Supported shortcut types for ResNets."""

    A = "A"
    B = "B"


class ResNets(str, Enum):
    """Supported ResNet networks."""

    RESNET_10 = "resnet10"
    RESNET_18 = "resnet18"
    RESNET_34 = "resnet34"
    RESNET_50 = "resnet50"
    RESNET_101 = "resnet101"
    RESNET_152 = "resnet152"
    RESNET_200 = "resnet200"


class UpsampleModes(str, Enum):
    """Supported upsampling modes for ResNets."""

    DECONV = "deconv"
    NON_TRAINABLE = "nontrainable"
    PIXEL_SHUFFLE = "pixelshuffle"


class PatchEmbeddingTypes(str, Enum):
    """Supported patch embedding types for VITs."""

    CONV = "conv"
    PERCEPTRON = "perceptron"


class PosEmbeddingTypes(str, Enum):
    """Supported positional embedding types for VITs."""

    NONE = "none"
    LEARNABLE = "learnable"
    SINCOS = "sincos"


class ClassificationActivation(str, Enum):
    """Supported activation layer for classification in ViT."""

    TANH = "Tanh"
