from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, PositiveInt, computed_field

from clinicadl.monai_networks.nn.layers.utils import ActivationParameters
from clinicadl.utils.factories import DefaultFromLibrary


class ImplementedNetworks(str, Enum):
    """Implemented neural networks in ClinicaDL."""

    MLP = "MLP"
    CONV_ENCODER = "ConvEncoder"
    CONV_DECODER = "ConvDecoder"
    CNN = "CNN"
    GENERATOR = "Generator"
    AE = "AutoEncoder"
    VAE = "VAE"
    DENSENET = "DenseNet"
    DENSENET_121 = "DenseNet-121"
    DENSENET_161 = "DenseNet-161"
    DENSENET_169 = "DenseNet-169"
    DENSENET_201 = "DenseNet-201"
    RESNET = "VarFullyConnectedNet"
    RESNET_18 = "ResNet-18"
    RESNET_34 = "ResNet-34"
    RESNET_50 = "ResNet-50"
    RESNET_101 = "ResNet-101"
    RESNET_152 = "ResNet-152"
    SE_RESNET = "SEResNet"
    SE_RESNET_50 = "SEResNet-50"
    SE_RESNET_101 = "SEResNet-101"
    SE_RESNET_152 = "SEResNet-152"
    UNET = "UNet"
    ATT_UNET = "AttentionUNet"
    VIT = "ViT"
    VIT_B_16 = "ViT-B/16"
    VIT_B_32 = "ViT-B/32"
    VIT_L_16 = "ViT-L/16"
    VIT_L_32 = "ViT-L/32"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not implemented. Implemented neural networks are: "
            + ", ".join([repr(m.value) for m in cls])
        )


class NetworkType(str, Enum):
    """
    Useful to know where to look for the network.
    See :py:func:`clinicadl.monai_networks.factory.get_network`
    """

    CUSTOM = "custom"  # our own networks
    RESNET = "sota-ReNet"
    DENSENET = "sota-DenseNet"
    SE_RESNET = "sota-SE-ResNet"
    VIT = "sota-ViT"


class NetworkConfig(BaseModel, ABC):
    """Base config class to configure neural networks."""

    # pydantic config
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        validate_default=True,
    )

    @computed_field
    @property
    @abstractmethod
    def name(self) -> ImplementedNetworks:
        """The name of the network."""

    @computed_field
    @property
    def _type(self) -> NetworkType:
        """
        To know where to look for the network.
        Default to 'custom'.
        """
        return NetworkType.CUSTOM


class PreTrainedConfig(NetworkConfig):
    """Base config class for SOTA networks."""

    num_outputs: Optional[PositiveInt]
    output_act: Union[
        Optional[ActivationParameters], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    pretrained: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES
