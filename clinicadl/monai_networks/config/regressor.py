from __future__ import annotations

from typing import Tuple

from pydantic import PositiveInt, computed_field, field_validator

from .base import VaryingDepthNetworkConfig
from .utils.enum import ImplementedNetworks

__all__ = ["RegressorConfig"]


class RegressorConfig(VaryingDepthNetworkConfig):
    """Config class for regressors."""

    in_shape: Tuple[PositiveInt, ...]
    out_shape: Tuple[PositiveInt, ...]

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.REGRESSOR

    @computed_field
    @property
    def dim(self) -> int:
        """Dimension of the images."""
        return len(self.in_shape[1:])

    @field_validator("in_shape")
    def at_least_2d(cls, v, ctx):
        """Checks that a tuple has at least a length of two."""
        return cls.base_at_least_2d(v, ctx)
