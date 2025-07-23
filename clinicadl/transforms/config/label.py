from typing import Optional, Union

import torchio as tio
from pydantic import field_validator

from clinicadl.utils.factories import get_defaults_from

from .base import Bounds, MaskingMethodConfig, TorchioTransformConfig
from .enum import AnatomicalLabel

__all__ = ["RemapLabelsConfig", "OneHotConfig"]

REMAP_LABELS_TORCHIO_DEFAULTS = get_defaults_from(tio.transforms.RemapLabels)
ONE_HOT_TORCHIO_DEFAULTS = get_defaults_from(tio.transforms.OneHot)


class RemapLabelsConfig(TorchioTransformConfig, MaskingMethodConfig):
    """
    Config class for :py:class:`torchio.transforms.RemapLabels`.
    """

    remapping: dict[int, int]
    masking_method: Optional[
        Union[str, AnatomicalLabel, Bounds]
    ] = REMAP_LABELS_TORCHIO_DEFAULTS["masking_method"]


class OneHotConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.OneHot`.
    """

    num_classes: int = ONE_HOT_TORCHIO_DEFAULTS["num_classes"]

    @field_validator("num_classes", mode="after")
    @classmethod
    def validator_num_classes(cls, v):
        """Checks that 'num_classes' is a positive integer (or -1)."""
        if isinstance(v, int) and (v <= 0) and (v != -1):
            raise ValueError(f"'num_classes' must be a positive integer or -1. Got {v}")
        return v
