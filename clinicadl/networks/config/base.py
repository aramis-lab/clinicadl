from enum import Enum
from typing import Optional, Sequence, Union

import torch.nn as nn
from pydantic import (
    PositiveFloat,
    PositiveInt,
    field_validator,
)

import clinicadl.networks.nn as nets
from clinicadl.networks.nn.layers.utils import ActivationParameters
from clinicadl.utils.config import (
    ClinicaDLConfig,
    ObjectConfig,
    update_kwargs_with_defaults,
)
from clinicadl.utils.factories import get_defaults_from

NN_MODULE_DEFAULTS = get_defaults_from(nn.Module)

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


class _FullyConvConfig(ClinicaDLConfig):
    """
    Config class for fully convolutional networks.
    """

    spatial_dims: PositiveInt
    in_channels: PositiveInt

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


class _OptionalLastLinearLayersConfig(ClinicaDLConfig):
    """Config class for 'num_outputs' option."""

    num_outputs: Optional[PositiveInt]


class _MandatoryActConfig(ClinicaDLConfig):
    """Config class for 'output_act' option."""

    act: ActivationParameters


class _OutputActConfig(ClinicaDLConfig):
    """Config class for 'output_act' option."""

    output_act: Optional[ActivationParameters]


class _DropOutConfig(ClinicaDLConfig):
    """Base config class for 'dropout' option."""

    dropout: Optional[PositiveFloat]

    @field_validator("dropout")
    @classmethod
    def validator_dropout(cls, v):
        """Checks that dropout is between 0 and 1."""
        if isinstance(v, float):
            assert (
                0 <= v <= 1
            ), f"'dropout' must be between 0 and 1 but it has been set to {v}."
        return v


class _PreTrainedConfig(
    NetworkConfig, _OptionalLastLinearLayersConfig, _OutputActConfig
):
    """Base config class for SOTA networks."""

    num_outputs: Optional[PositiveInt]
    pretrained: bool = False  # default ??
    output_act: Optional[ActivationParameters] = None  # default ???

    # TODO : to remove ??
    # def __init__(
    #     self,
    #     num_outputs: Optional[PositiveInt],
    #     output_act: Optional[ActivationParameters] = None,
    #     pretrained: bool = False,
    # ):
    #     kwargs = {
    #         "num_outputs": num_outputs,
    #         "output_act": output_act,
    #         "pretrained": pretrained,
    #     }
    #     associated_getter = (
    #         self._get_class()
    #     )  # special cas here: _get_class does not return a class
    #     kwargs = update_kwargs_with_defaults(kwargs, function=associated_getter)
    #     super().__init__(**kwargs)

    def get_object(self) -> nn.Module:
        """
        Returns the neural network associated to this configuration,
        parametrized with the parameters passed by the user.

        Returns
        -------
        torch.nn.Module:
            The neural network.
        """
        associated_getter = self._get_class()
        return associated_getter(
            name=self._get_name(), **self.model_dump(exclude="name")
        )
