from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    computed_field,
    field_validator,
    model_validator,
)

from clinicadl.utils.factories import DefaultFromLibrary

from .utils.enum import (
    ImplementedActFunctions,
    ImplementedNetworks,
    ImplementedNormLayers,
)


class NetworkConfig(BaseModel, ABC):
    """Base config class to configure neural networks."""

    kernel_size: Union[
        PositiveInt, Tuple[PositiveInt, ...], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    up_kernel_size: Union[
        PositiveInt, Tuple[PositiveInt, ...], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    num_res_units: Union[NonNegativeInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    act: Union[
        ImplementedActFunctions,
        Tuple[ImplementedActFunctions, Dict[str, Any]],
        DefaultFromLibrary,
    ] = DefaultFromLibrary.YES
    norm: Union[
        ImplementedNormLayers,
        Tuple[ImplementedNormLayers, Dict[str, Any]],
        DefaultFromLibrary,
    ] = DefaultFromLibrary.YES
    bias: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES
    adn_ordering: Union[Optional[str], DefaultFromLibrary] = DefaultFromLibrary.YES
    # pydantic config
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        validate_default=True,
        protected_namespaces=(),
    )

    @computed_field
    @property
    @abstractmethod
    def network(self) -> ImplementedNetworks:
        """The name of the network."""

    @computed_field
    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of the images."""

    @classmethod
    def base_validator_dropout(cls, v):
        """Checks that dropout is between 0 and 1."""
        if isinstance(v, float):
            assert (
                0 <= v <= 1
            ), f"dropout must be between 0 and 1 but it has been set to {v}."
        return v

    @field_validator("kernel_size", "up_kernel_size")
    @classmethod
    def base_is_odd(cls, value, field):
        """Checks if a field is odd."""
        if value != DefaultFromLibrary.YES:
            if isinstance(value, int):
                value_ = (value,)
            else:
                value_ = value
            for v in value_:
                assert v % 2 == 1, f"{field.field_name} must be odd."
        return value

    @field_validator("adn_ordering", mode="after")
    @classmethod
    def base_adn_validator(cls, v):
        """Checks ADN sequence."""
        if v != DefaultFromLibrary.YES:
            for letter in v:
                assert (
                    letter in {"A", "D", "N"}
                ), f"adn_ordering must be composed by 'A', 'D' or/and 'N'. You passed {letter}."
            assert len(v) == len(
                set(v)
            ), "adn_ordering cannot contain duplicated letter."

        return v

    @classmethod
    def base_at_least_2d(cls, v, ctx):
        """Checks that a tuple has at least a length of two."""
        if isinstance(v, tuple):
            assert (
                len(v) >= 2
            ), f"{ctx.field_name} should have at least two dimensions (with the first one for the channel)."
        return v

    @model_validator(mode="after")
    def base_model_validator(self):
        """Checks coherence between parameters."""
        if self.kernel_size != DefaultFromLibrary.YES:
            assert self._check_dimensions(
                self.kernel_size
            ), f"You must passed an int or a sequence of {self.dim} ints (the dimensionality of your images) for kernel_size. You passed {self.kernel_size}."
        if self.up_kernel_size != DefaultFromLibrary.YES:
            assert self._check_dimensions(
                self.up_kernel_size
            ), f"You must passed an int or a sequence of {self.dim} ints (the dimensionality of your images) for up_kernel_size. You passed {self.up_kernel_size}."
        return self

    def _check_dimensions(
        self,
        value: Union[float, Tuple[float, ...]],
    ) -> bool:
        """Checks if a tuple has the right dimension."""
        if isinstance(value, tuple):
            return len(value) == self.dim
        return True


class VaryingDepthNetworkConfig(NetworkConfig, ABC):
    """
    Base config class to configure neural networks.
    More precisely, we refer to MONAI's networks with 'channels' and 'strides' parameters.
    """

    channels: Tuple[PositiveInt, ...]
    strides: Tuple[Union[PositiveInt, Tuple[PositiveInt, ...]], ...]
    dropout: Union[
        Optional[NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @field_validator("dropout")
    @classmethod
    def validator_dropout(cls, v):
        """Checks that dropout is between 0 and 1."""
        return cls.base_validator_dropout(v)

    @model_validator(mode="after")
    def channels_strides_validator(self):
        """Checks coherence between parameters."""
        n_layers = len(self.channels)
        assert (
            len(self.strides) == n_layers
        ), f"There are {n_layers} layers but you passed {len(self.strides)} strides."
        for s in self.strides:
            assert self._check_dimensions(
                s
            ), f"You must passed an int or a sequence of {self.dim} ints (the dimensionality of your images) for strides. You passed {s}."

        return self
