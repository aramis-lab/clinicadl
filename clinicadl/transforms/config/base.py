from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union

from pydantic import (
    NonNegativeFloat,
    NonNegativeInt,
    computed_field,
    field_validator,
    model_validator,
)

from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.factories import DefaultFromLibrary

from .enum import (
    AnatomicalAxis,
    AnatomicalLabel,
    ImplementedTransform,
    NumericalAxis,
    TransformType,
)


class TransformConfig(ClinicaDLConfig, ABC):
    """Base config class for the transforms."""

    @computed_field
    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the transform."""

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
        if len(tup) == 2:
            cls._is_couple_sorted(tup, field_name)
        elif len(tup) == 6:
            cls._is_six_tuple_sorted(tup, field_name)


class OneOfConfig(TransformConfig):
    """Config class for OneOf augmentation."""

    transforms: List[TransformConfig]
    probabilities: Optional[List[NonNegativeFloat]] = None

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.ONE_OF.value

    @model_validator(mode="after")
    def check_probabilities(self):
        """Checks that 'probabilities' is the same length as 'transforms'."""
        if self.probabilities is None:
            self.probabilities = [(1 / len(self.transforms)) for _ in self.transforms]
        else:
            if len(self.transforms) != len(self.probabilities):
                raise ValueError(
                    "If 'probabilities' is passed, it must be the same length as 'transforms'."
                )
        return self


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


class _AnatomicalAxesConfig(ClinicaDLConfig):
    """Config class for 'axes' option when it supports anatomical values."""

    axes: Union[
        NumericalAxis,
        Tuple[NumericalAxis, ...],
        AnatomicalAxis,
        Tuple[AnatomicalAxis, ...],
        DefaultFromLibrary,
    ] = DefaultFromLibrary.YES
