from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Tuple, Union

from pydantic import (
    NonNegativeFloat,
    NonNegativeInt,
    field_validator,
    model_validator,
)
from torchio import Compose, transforms
from torchio import Transform as TorchioTransform

from clinicadl.utils.config import ClinicaDLConfig, ObjectConfig

from .enum import AnatomicalLabel

if TYPE_CHECKING:
    from ..types import Transform

__all__ = [
    "TransformConfig",
    "OneOfConfig",
]


class TransformConfig(ObjectConfig):
    """Base config class for the transforms."""

    include: Optional[Sequence[str]] = None
    exclude: Optional[Sequence[str]] = None

    @model_validator(mode="after")
    def check_include_exclude(self):
        """Checks that 'include' and 'exclude' are not both specified."""
        if self.include and self.exclude:
            raise ValueError("'include' and 'exclude' cannot be both specified.")
        return self


class TorchioTransformConfig(TransformConfig):
    """Base config class for the transforms from TorchIO."""

    def get_object(self) -> Transform:
        """
        Returns the transform associated to this configuration,
        parametrized with the parameters passed by the user.

        Returns
        -------
        Transform:
            The associated transform.
        """
        return super().get_object()

    @classmethod
    def _get_class(cls) -> type[TorchioTransform]:
        """Returns the transform associated to this config class."""
        return getattr(transforms, cls._get_name())

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


class OneOfConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.OneOf`.
    TODO: Explain why 2 lists are used for transforms and probabilities instead of a dictionary
    """

    transforms: List[Union[TransformConfig, List[TransformConfig]]]
    probabilities: Optional[List[NonNegativeFloat]] = None

    def get_object(self) -> Transform:
        """
        Returns the transform associated to this configuration,
        parametrized with the parameters passed by the user.

        Returns
        -------
        Transform:
            The associated transform.
        """
        config_dict = {}
        for transform, proba in zip(self.transforms, self.probabilities):
            if isinstance(transform, TransformConfig):
                config_dict[transform.get_object()] = proba
            else:
                transform: List[TransformConfig]
                config_dict[Compose([t.get_object() for t in transform])] = proba

        one_of = self._get_class()(transforms=config_dict)
        return one_of

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


class MaskingMethodConfig(ClinicaDLConfig):
    """Base config class 'masking_method' argument."""

    masking_method: Optional[Union[str, AnatomicalLabel, Bounds]]

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
