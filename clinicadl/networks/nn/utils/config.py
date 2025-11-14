from typing import Optional, Sequence

import torch.nn as nn
from pydantic import (
    NonNegativeFloat,
    PositiveInt,
    field_validator,
)

from clinicadl.utils.config import (
    ClinicaDLConfig,
    ObjectConfig,
)


class NetworkConfig(ObjectConfig[nn.Module]):
    """Base config class to configure neural networks."""


class _SpatialDimsConfig(ClinicaDLConfig):
    """
    Config class for 'spatial_dims' option.
    """

    spatial_dims: PositiveInt

    @field_validator("spatial_dims", mode="after")
    @classmethod
    def dimension_validator(cls, v):
        """Checks that the network is 1D, 2D or 3D."""
        if v > 3:
            raise ValueError(f"'spatial_dims' must be between 1 and 3. Got {v}")
        return v


class _InShapeConfig(ClinicaDLConfig):
    """Config class for 'in_shape' option."""

    in_shape: Sequence[PositiveInt]

    @field_validator("in_shape", mode="after")
    @classmethod
    def validator_in_shape(cls, v):
        """Checks that 'in_shape' corresponds to 1D, 2D or 3D images."""
        assert (
            2 <= len(v) <= 4
        ), f"'in_shape' must be of length 2 (1D), 3 (2D image) or 4 (3D image). Don't forget the channel dimension. Got: {v}."
        return v


class _DropoutConfig(ClinicaDLConfig):
    """Config class for 'dropout' option."""

    dropout: Optional[NonNegativeFloat]

    @field_validator("dropout", mode="after")
    @classmethod
    def validator_dropout(cls, v):
        """Checks that dropout is between 0 and 1."""
        if isinstance(v, float):
            assert (
                0 <= v <= 1
            ), f"'dropout' must be between 0 and 1 but it has been set to {v}."
        return v
