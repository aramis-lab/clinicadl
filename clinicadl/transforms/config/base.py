from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

from pydantic import computed_field, field_validator

from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.factories import DefaultFromLibrary

from .enum import AnatomicalLabel, ImplementedTransform
from .utils import Bounds


class TransformConfig(ClinicaDLConfig, ABC):
    """Base config class for the transforms."""

    @computed_field
    @property
    @abstractmethod
    def name(self) -> ImplementedTransform:
        """The name of the transform."""


class _MaskingMethodConfig(ClinicaDLConfig):
    """Base config class for normalization transforms."""

    masking_method: Optional[
        Union[str, AnatomicalLabel, Bounds, DefaultFromLibrary]
    ] = DefaultFromLibrary.YES

    @field_validator("masking_method", mode="before")
    @classmethod
    def validator_masking_method(cls, v):
        """To handle 'masking_method' different types."""
        if isinstance(v, Callable):
            raise ValueError("'masking_method' passed as a callable is not supported.")
        elif isinstance(v, str):
            try:
                v = AnatomicalLabel(v)
            except ValueError:
                pass
        return v
