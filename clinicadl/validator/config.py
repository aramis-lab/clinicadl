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

    # maps_path: Path
    mode: str

    metrics_module: Optional = None
    report_ci: bool = False
    selection_metrics: list

    n_classes: int = 1
    network_task: str
    num_networks: int = 1
    use_labels: bool = True

    gpu: Optional[bool] = None
    amp: bool = False
    fsdp: bool = False

    split_name: Optional[str] = None
    nb_unfrozen_layers: Optional[int] = None
    std_amp: Optional[bool] = None

    # pydantic config
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    @computed_field
    @property
    @abstractmethod
    def metric(self) -> str:
        """The name of the metric."""

    # @field_validator("get_not_nans", mode="after")
    # @classmethod
    # def validator_get_not_nans(cls, v):
    #     assert not v, "get_not_nans not supported in ClinicaDL. Please set to False."

    #     return v
