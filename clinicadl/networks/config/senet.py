from typing import Any, Callable, Optional, Sequence, Union

import torch.nn as nn
from pydantic import PositiveInt, field_validator, model_validator

import clinicadl.networks.nn as nets
from clinicadl.networks.nn.layers.utils import ActivationParameters
from clinicadl.networks.nn.resnet import ResNetBlockType
from clinicadl.networks.nn.senet import check_se_channels
from clinicadl.utils.factories import get_defaults_from

from .base import ImplementedNetwork, _PreTrainedConfig
from .resnet import ResNetConfig

SERESNET_DEFAULTS = get_defaults_from(nets.SEResNet)
print(SERESNET_DEFAULTS)

__all__ = [
    "SEResNetConfig",
    "SEResNet50Config",
    "SEResNet101Config",
    "SEResNet152Config",
]


class SEResNetConfig(ResNetConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.SEResNet`.
    """

    se_reduction: PositiveInt
    spatial_dims: PositiveInt
    in_channels: PositiveInt
    num_outputs: Optional[PositiveInt]
    se_reduction: PositiveInt = SERESNET_DEFAULTS["se_reduction"]

    @model_validator(mode="after")
    def check_se_channels(self):
        check_se_channels(self.n_features, self.se_reduction)

        return self


class _PreTrainedSEResNetConfig(_PreTrainedConfig):
    """Base config class for SOTA SE-ResNets."""

    pretrained: bool = False

    @field_validator("pretrained")
    @classmethod
    def check_not_pretrained(cls, v):
        assert not v, "Pretrained networks are not yet available for SE-ResNets. Please leave 'pretrained' to False."

        return v

    @classmethod
    def _get_class(cls) -> Callable[[Any], nn.Module]:
        """Returns the network associated to this config class."""
        return nets.get_seresnet


class SEResNet50Config(_PreTrainedSEResNetConfig):
    """
    Config class for :py:func:`SE-ResNet-50 <clinicadl.networks.nn.get_seresnet>`.
    """

    @classmethod
    def _get_name(cls) -> str:
        """Returns the name of the class associated to this config class."""
        return ImplementedNetwork.SE_RESNET_50.value


class SEResNet101Config(_PreTrainedSEResNetConfig):
    """
    Config class for :py:func:`SE-ResNet-101 <clinicadl.networks.nn.get_seresnet>`.
    """

    @classmethod
    def _get_name(cls) -> str:
        """Returns the name of the class associated to this config class."""
        return ImplementedNetwork.SE_RESNET_101.value


class SEResNet152Config(_PreTrainedSEResNetConfig):
    """
    Config class for :py:func:`SE-ResNet-152 <clinicadl.networks.nn.get_seresnet>`.
    """

    @classmethod
    def _get_name(cls) -> str:
        """Returns the name of the class associated to this config class."""
        return ImplementedNetwork.SE_RESNET_152.value
