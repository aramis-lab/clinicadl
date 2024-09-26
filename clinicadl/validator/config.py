from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    field_validator,
)

from clinicadl.utils.factories import DefaultFromLibrary


class ValidatorConfig(BaseModel):
    """Base config class to configure the validator."""

    maps_path: Path
    mode: str
    network_task: str
    split_name: Optional[str] = None
    num_networks: Optional[int] = None
    fsdp: Optional[bool] = None
    amp: Optional[bool] = None
    metrics_module: Optional = None
    n_classes: Optional[int] = None
    nb_unfrozen_layers: Optional[int] = None
    std_amp: Optional[bool] = None

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
