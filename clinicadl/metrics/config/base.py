from abc import ABC, abstractmethod
from typing import Union

from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    field_validator,
)

from clinicadl.utils.factories import DefaultFromLibrary

from .enum import Reduction


class MetricConfig(BaseModel, ABC):
    """Base config class to configure metrics."""

    reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES
    get_not_nans: bool = False
    include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES
    # pydantic config
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        validate_default=True,
    )

    @computed_field
    @property
    @abstractmethod
    def metric(self) -> str:
        """The name of the metric."""

    @field_validator("get_not_nans", mode="after")
    @classmethod
    def validator_get_not_nans(cls, v):
        assert not v, "get_not_nans not supported in ClinicaDL. Please set to False."

        return v
