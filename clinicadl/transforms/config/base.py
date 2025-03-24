from typing import Any, Callable, List, Optional, Tuple, Union

import torchio as tio
from pydantic import (
    NonNegativeFloat,
    NonNegativeInt,
    computed_field,
    field_validator,
    model_validator,
)

from clinicadl.utils.config import NewClinicaDLConfig

from .enum import (
    AnatomicalLabel,
    ImplementedTransform,
)

__all__ = [
    "TransformConfig",
    "OneOfConfig",
]


class TransformConfig(NewClinicaDLConfig):
    """Base config class for the transforms."""

    def get_object(self) -> tio.Transform:
        """
        Returns the transform associated to this configuration,
        parametrized with the parameters passed by the user.

        Returns
        -------
        tio.Transform:
            The TorchIO transform.
        """
        return super().get_object()

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
    """
    Config class for :py:class:`torchio.transforms.OneOf`.
    """

    transforms: List[TransformConfig]
    probabilities: Optional[List[NonNegativeFloat]] = None

    def __init__(
        self,
        transforms: List[TransformConfig],
        probabilities: Optional[List[NonNegativeFloat]] = None,
    ):
        super().__init__(
            transforms=transforms,
            probabilities=probabilities,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.ONE_OF.value

    def get_object(self) -> tio.Transform:
        """
        Returns the transform associated to this configuration,
        parametrized with the parameters passed by the user.

        Returns
        -------
        tio.Transform:
            The TorchIO transform.
        """
        config_dict = {
            transform.get_object(): proba
            for transform, proba in zip(self.transforms, self.probabilities)
        }
        one_of = self._get_class()(transforms=config_dict)
        return one_of

    def _get_class(self) -> type[tio.Transform]:
        """Returns the transform associated to this config class."""
        return tio.OneOf

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


class MaskingMethodConfig(NewClinicaDLConfig):
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
