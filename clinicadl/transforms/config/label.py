from typing import Union

from pydantic import computed_field, field_validator

from clinicadl.utils.factories import DefaultFromLibrary

from .base import ImplementedTransform, TransformConfig, _MaskingMethodConfig

__all__ = ["RemapLabelsConfig", "OneHotConfig"]


class RemapLabelsConfig(TransformConfig, _MaskingMethodConfig):
    """Config class for RemapLabels transform."""

    remapping: dict[int, int]

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.REMAP_LABELS


class OneHotConfig(TransformConfig):
    """Config class for OneHot transform."""

    num_classes: Union[int, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.ONE_HOT

    @field_validator("num_classes", mode="after")
    @classmethod
    def validator_num_classes(cls, v):
        """Checks that 'num_classes' is a positive integer (or -1)."""
        if isinstance(v, int) and (v <= 0) and (v != -1):
            raise ValueError(f"'num_classes' must be a positive integer or -1. Got {v}")
        return v
