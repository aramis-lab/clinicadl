from typing import Optional, Sequence

from pydantic import PositiveFloat, PositiveInt, field_validator

import clinicadl.networks.nn as nets
from clinicadl.networks.nn.layers.utils import ActivationParameters
from clinicadl.utils.factories import get_defaults_from

from .base import (
    NetworkConfig,
    _DropoutConfig,
    _SpatialDimsConfig,
)

__all__ = [
    "UNetConfig",
    "AttentionUNetConfig",
]

UNET_DEFAULTS = get_defaults_from(nets.UNet)
ATTENTION_UNET_DEFAULTS = get_defaults_from(nets.AttentionUNet)


class UNetConfig(
    NetworkConfig,
    _SpatialDimsConfig,
    _DropoutConfig,
):
    """
    Config class for :py:class:`clinicadl.networks.nn.UNet`.
    """

    spatial_dims: PositiveInt
    in_channels: PositiveInt
    out_channels: PositiveInt
    channels: Sequence[PositiveInt] = UNET_DEFAULTS["channels"]
    act: ActivationParameters = UNET_DEFAULTS["act"]
    output_act: Optional[ActivationParameters] = UNET_DEFAULTS["output_act"]
    dropout: Optional[PositiveFloat] = UNET_DEFAULTS["dropout"]

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
