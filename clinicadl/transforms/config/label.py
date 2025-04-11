from typing import Optional, Union

from pydantic import field_validator

from clinicadl.utils.config import DefaultFromLibrary

from .base import Bounds, MaskingMethodConfig, TransformConfig
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

    @field_validator("num_classes", mode="after")
    @classmethod
    def validator_num_classes(cls, v):
        """Checks that 'num_classes' is a positive integer (or -1)."""
        if isinstance(v, int) and (v <= 0) and (v != -1):
            raise ValueError(f"'num_classes' must be a positive integer or -1. Got {v}")
        return v
