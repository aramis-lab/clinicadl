from enum import Enum
from typing import Optional, Sequence

import torch.nn as nn
from pydantic import (
    PositiveFloat,
    PositiveInt,
    field_validator,
)

import clinicadl.networks.nn as nets
from clinicadl.utils.config import (
    ClinicaDLConfig,
    ObjectConfig,
)

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


class NetworkConfig(ObjectConfig):
    """Base config class to configure neural networks."""

    def get_object(self) -> nn.Module:
        """
        Returns the neural network associated to this configuration,
        parametrized with the parameters passed by the user.

        Returns
        -------
        torch.nn.Module:
            The neural network.
        """
        return super().get_object()

    @classmethod
    def _get_class(cls) -> type[nn.Module]:
        """Returns the network associated to this config class."""
        return getattr(nets, cls._get_name())


class _SpatialDimsConfig(ClinicaDLConfig):
    """
    Config class for 'spatial_dims' option.
    """

    spatial_dims: PositiveInt

    @field_validator("spatial_dims", mode="after")
    @classmethod
    def dimension_validator(cls, v):
        """Checks that the network is 1D, 2D or 3D."""
        if v > 3:
            raise ValueError(f"'spatial_dims' must be between 1 and 3. Got {v}")
        return v


class _InShapeConfig(ClinicaDLConfig):
    """Config class for 'in_shape' option."""

    in_shape: Sequence[PositiveInt]

    @field_validator("in_shape", mode="after")
    @classmethod
    def validator_in_shape(cls, v):
        """Checks that 'in_shape' corresponds to 1D, 2D or 3D images."""
        assert (
            2 <= len(v) <= 4
        ), f"'in_shape' must be of length 2 (1D), 3 (2D image) or 4 (3D image). Don't forget the channel dimension. Got: {v}."
        return v


class _DropoutConfig(ClinicaDLConfig):
    """Config class for 'dropout' option."""

    dropout: Optional[PositiveFloat]

    @field_validator("dropout", mode="after")
    @classmethod
    def validator_dropout(cls, v):
        """Checks that dropout is between 0 and 1."""
        if isinstance(v, float):
            assert (
                0 <= v <= 1
            ), f"'dropout' must be between 0 and 1 but it has been set to {v}."
        return v
