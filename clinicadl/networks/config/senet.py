from typing import Optional, Sequence, Union

from pydantic import PositiveInt, model_validator

import clinicadl.networks.nn as nets
from clinicadl.networks.nn.layers.utils import ActivationParameters
from clinicadl.networks.nn.resnet import ResNetBlockType
from clinicadl.networks.nn.senet import check_se_channels
from clinicadl.utils.factories import get_defaults_from

from .base import NetworkConfig
from .resnet import ResNetConfig

SE_RES_NET_DEFAULTS = get_defaults_from(nets.SEResNet)
SE_RES_NET_50_DEFAULTS = get_defaults_from(nets.SEResNet50)
SE_RES_NET_101_DEFAULTS = get_defaults_from(nets.SEResNet101)
SE_RES_NET_152_DEFAULTS = get_defaults_from(nets.SEResNet152)

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

    se_reduction: PositiveInt = SE_RES_NET_DEFAULTS["se_reduction"]

    @model_validator(mode="after")
    def check_se_channels(self):
        check_se_channels(self.n_features, self.se_reduction)

        return self


class SEResNet50Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.SEResNet50`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = SE_RES_NET_50_DEFAULTS["output_act"]


class SEResNet101Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.SEResNet101`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = SE_RES_NET_101_DEFAULTS["output_act"]


class SEResNet152Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.SEResNet152`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = SE_RES_NET_152_DEFAULTS["output_act"]
