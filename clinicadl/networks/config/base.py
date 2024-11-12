from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Sequence, Union

from pydantic import (
    PositiveFloat,
    PositiveInt,
    computed_field,
    field_validator,
)

from clinicadl.networks.nn.layers.utils import ActivationParameters
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.factories import DefaultFromLibrary

__all__ = ["ImplementedNetwork", "NetworkConfig"]


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
    DENSENET_121 = "DenseNet-121"
    DENSENET_161 = "DenseNet-161"
    DENSENET_169 = "DenseNet-169"
    DENSENET_201 = "DenseNet-201"
    RESNET = "ResNet"
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
    RESNET = "sota-ResNet"
    DENSENET = "sota-DenseNet"
    SE_RESNET = "sota-SEResNet"
    VIT = "sota-ViT"


class NetworkConfig(ClinicaDLConfig, ABC):
    """Base config class to configure neural networks."""

    @computed_field
    @property
    @abstractmethod
    def name(self) -> ImplementedNetwork:
        """The name of the network."""

    @computed_field
    @property
    def _type(self) -> NetworkType:
        """
        To know where to look for the network.
        Default to 'custom'.
        """
        return NetworkType.CUSTOM


class _FullyConvConfig(ClinicaDLConfig):
    """
    Base config class for fully convolutional networks.
    """

    spatial_dims: PositiveInt
    in_channels: PositiveInt


class _InShapeConfig(ClinicaDLConfig):
    """Base config class for 'in_shape' option."""

    in_shape: Sequence[PositiveInt]


class _OptionalLastLinearLayersConfig(ClinicaDLConfig):
    """Base config class for 'num_outputs' option."""

    num_outputs: Optional[PositiveInt]


class _MandatoryActConfig(ClinicaDLConfig):
    """Base config class for 'output_act' option."""

    act: Union[ActivationParameters, DefaultFromLibrary] = DefaultFromLibrary.YES


class _OutputActConfig(ClinicaDLConfig):
    """Base config class for 'output_act' option."""

    output_act: Union[
        Optional[ActivationParameters], DefaultFromLibrary
    ] = DefaultFromLibrary.YES


class _DropOutConfig(ClinicaDLConfig):
    """Base config class for 'dropout' option."""

    dropout: Union[Optional[PositiveFloat], DefaultFromLibrary] = DefaultFromLibrary.YES

    @field_validator("dropout")
    @classmethod
    def validator_dropout(cls, v):
        """Checks that dropout is between 0 and 1."""
        if isinstance(v, float):
            assert (
                0 <= v <= 1
            ), f"dropout must be between 0 and 1 but it has been set to {v}."
        return v


class _PreTrainedConfig(_OptionalLastLinearLayersConfig, _OutputActConfig):
    """Base config class for SOTA networks."""

    pretrained: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES
