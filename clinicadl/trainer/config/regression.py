from logging import getLogger
from typing import Tuple, Union

from pydantic import computed_field, field_validator

from clinicadl.caps_dataset.data_config import DataConfig as BaseDataConfig
from clinicadl.network.config import NetworkConfig as BaseNetworkConfig
from clinicadl.trainer.config.train import TrainConfig
from clinicadl.utils.enum import RegressionLoss, RegressionMetric, Task
from clinicadl.validator.validation import ValidationConfig as BaseValidationConfig

logger = getLogger("clinicadl.reconstruction_config")
logger = getLogger("clinicadl.regression_config")


class DataConfig(BaseDataConfig):  # TODO : put in data module
    """Config class to specify the data in regression mode."""

    label: str = "age"

    @field_validator("label")
    def validator_label(cls, v):
        return v  # TODO : check if label in columns


class NetworkConfig(BaseNetworkConfig):  # TODO : put in model module
    """Config class for regression models."""

    architecture: str = "Conv5_FC3"
    loss: RegressionLoss = RegressionLoss.MSELoss

    @field_validator("architecture")
    def validator_architecture(cls, v):
        return v  # TODO : connect to network module to have list of available architectures


class ValidationConfig(BaseValidationConfig):
    """Config class for the validation procedure in regression mode."""

    selection_metrics: Tuple[RegressionMetric, ...] = (RegressionMetric.LOSS,)

    @field_validator("selection_metrics", mode="before")
    def list_to_tuples(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v


class RegressionConfig(TrainConfig):
    """
    Config class for the training of a regression model.

    The user must specified at least the following arguments:
    - caps_directory
    - preprocessing_json
    - tsv_path
    - output_maps_directory
    """

    data: DataConfig
    model: NetworkConfig
    validation: ValidationConfig

    @computed_field
    @property
    def network_task(self) -> Task:
        return Task.REGRESSION
