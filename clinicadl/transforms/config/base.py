from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple, Union

from pydantic import NonNegativeInt, computed_field, field_validator

from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.factories import DefaultFromLibrary

from .enum import AnatomicalLabel, ImplementedTransform, NumericalAxis, TransformType


class TransformConfig(ClinicaDLConfig, ABC):
    """Base config class for the transforms."""

    @computed_field
    @property
    @abstractmethod
    def name(self) -> ImplementedTransform:
        """The name of the transform."""

    @computed_field
    @property
    def _type(self) -> TransformType:
        """The source where the transform can be found."""
        return TransformType.TORCHIO

    @staticmethod
    def _is_couple_sorted(tup: Tuple[Any, Any], field_name: str) -> None:
        """Checks that a couple is sorted. Useful for many fields."""
        if sorted(list(tup)) != list(tup):
            raise ValueError(
                f"If {field_name} is a couple, the first element must be smaller "
                f"than the second. Got {tup}"
            )

    @classmethod
    def _is_six_tuple_sorted(
        cls, tup: Tuple[Any, Any, Any, Any, Any, Any], field_name: str
    ) -> None:
        """
        Checks that a tuple of size 6, with 2 values for each dimension, is sorted for
        each dimension. Useful for many fields.
        """
        cls._is_couple_sorted(tup[:2], field_name)
        cls._is_couple_sorted(tup[2:4], field_name)
        cls._is_couple_sorted(tup[4:], field_name)

    @classmethod
    def _check_spatial_tuple(
        cls,
        tup: Union[Tuple[Any, Any], Tuple[Any, Any, Any, Any, Any, Any]],
        field_name: str,
    ) -> None:
        """
        Global checks for spatial parameters that are passed as a tuple (either a common tuple
        or a tuple for each dimension).
        """
        if len(tuple) == 2:
            cls._is_couple_sorted(tup, field_name)
        elif len(tuple) == 6:
            cls._is_six_tuple_sorted(tup, field_name)


Bounds = Union[
    NonNegativeInt,
    Tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt],
    Tuple[
        NonNegativeInt,
        NonNegativeInt,
        NonNegativeInt,
        NonNegativeInt,
        NonNegativeInt,
        NonNegativeInt,
    ],
]


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


class _NumericalAxesConfig(ClinicaDLConfig):
    """Config class for 'axes' option when it supports only numerical values."""

    axes: Union[
        NumericalAxis, Tuple[NumericalAxis, ...], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
