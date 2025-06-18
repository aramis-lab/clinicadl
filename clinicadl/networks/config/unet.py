from typing import Optional, Sequence, Union

from pydantic import PositiveFloat, PositiveInt, field_validator

import clinicadl.networks.nn as nets
from clinicadl.networks.nn.layers.utils import ActivationParameters
from clinicadl.utils.config import update_kwargs_with_defaults
from clinicadl.utils.factories import get_defaults_from

from ..nn.att_unet import AttentionUNet
from ..nn.unet import UNet
from .base import (
    NetworkConfig,
    _DropOutConfig,
    _FullyConvConfig,
    _MandatoryActConfig,
    _OutputActConfig,
)

BASE_UNET_DEFAULTS = get_defaults_from(nets.unet.BaseUNet)
ATT_UNET_DEFAULTS = get_defaults_from(nets.AttentionUNet)
ATT_UNET_DEFAULTS.update(BASE_UNET_DEFAULTS)

__all__ = [
    "UNetConfig",
    "AttentionUNetConfig",
]


class UNetConfig(
    NetworkConfig,
    _FullyConvConfig,
    _MandatoryActConfig,
    _OutputActConfig,
    _DropOutConfig,
):
    """
    Config class for :py:class:`clinicadl.networks.nn.UNet`.
    """

    spatial_dims: PositiveInt
    in_channels: PositiveInt
    out_channels: PositiveInt
    channels: Sequence[PositiveInt] = BASE_UNET_DEFAULTS["channels"]
    act: ActivationParameters = BASE_UNET_DEFAULTS["act"]
    output_act: Optional[ActivationParameters] = BASE_UNET_DEFAULTS["output_act"]
    dropout: Optional[PositiveFloat] = BASE_UNET_DEFAULTS["dropout"]

    @field_validator("channels")
    @classmethod
    def channels_validator(cls, v):
        if isinstance(v, Sequence) and len(v) < 2:
            raise ValueError(f"length of channels must be no less than 2. Got {v}")
        return v


class AttentionUNetConfig(UNetConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.AttentionUNet`.
    """

    spatial_dims: PositiveInt
    in_channels: PositiveInt
    out_channels: PositiveInt
    channels: Sequence[PositiveInt] = ATT_UNET_DEFAULTS["channels"]
    act: ActivationParameters = ATT_UNET_DEFAULTS["act"]
    output_act: Optional[ActivationParameters] = ATT_UNET_DEFAULTS["output_act"]
    dropout: Optional[PositiveFloat] = ATT_UNET_DEFAULTS["dropout"]
