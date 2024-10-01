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
    n_classes: int = 1
    network_task: str
    # model: Network
    # dataloader: DataLoader
    # criterion: _Loss
    use_labels: bool = True
    amp: bool = False
    fsdp: bool = False
    report_ci = False
    gpu: Optional[bool] = None
    selection_metrics: list

    split_name: Optional[str] = None
    num_networks: Optional[int] = None
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
