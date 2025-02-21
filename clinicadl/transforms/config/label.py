from typing import Optional, Union

import torchio as tio
from pydantic import computed_field, field_validator

from clinicadl.utils.config import DefaultFromLibrary

from .base import Bounds, ImplementedTransform, MaskingMethodConfig, TransformConfig
from .enum import AnatomicalLabel

__all__ = ["RemapLabelsConfig", "OneHotConfig"]


class RemapLabelsConfig(TransformConfig, MaskingMethodConfig):
    """
    Config class for :py:class:`torchio.transforms.RemapLabels`.
    """

    remapping: dict[int, int]

    def __init__(
        self,
        remapping: dict[int, int],
        masking_method: Optional[
            Union[str, AnatomicalLabel, Bounds, DefaultFromLibrary]
        ] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            remapping=remapping,
            masking_method=masking_method,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.REMAP_LABELS.value

    def _get_class(self) -> type[tio.Transform]:
        """Returns the transform associated to this config class."""
        return tio.RemapLabels


class OneHotConfig(TransformConfig):
    """
    Config class for :py:class:`torchio.transforms.OneHot`.
    """

    num_classes: int

    def __init__(
        self, num_classes: Union[int, DefaultFromLibrary] = DefaultFromLibrary.YES
    ):
        super().__init__(
            num_classes=num_classes,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.ONE_HOT.value

    def _get_class(self) -> type[tio.Transform]:
        """Returns the transform associated to this config class."""
        return tio.OneHot

    @field_validator("num_classes", mode="after")
    @classmethod
    def validator_num_classes(cls, v):
        """Checks that 'num_classes' is a positive integer (or -1)."""
        if isinstance(v, int) and (v <= 0) and (v != -1):
            raise ValueError(f"'num_classes' must be a positive integer or -1. Got {v}")
        return v
