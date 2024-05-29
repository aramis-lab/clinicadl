from enum import Enum
from logging import getLogger
from typing import Tuple

from pydantic import PositiveFloat, PositiveInt, computed_field, field_validator

from clinicadl.config.config import DataConfig as BaseDataConfig
from clinicadl.config.config import ModelConfig as BaseModelConfig
from clinicadl.config.config import ValidationConfig as BaseValidationConfig
from clinicadl.train.trainer.training_config import TrainingConfig
from clinicadl.utils.enum import RegressionLoss, RegressionMetric, Task

logger = getLogger("clinicadl.reconstruction_config")
logger = getLogger("clinicadl.regression_config")


class DataConfig(BaseDataConfig):  # TODO : put in data module
    """Config class to specify the data in regression mode."""

    label: str = "age"

    @field_validator("label")
    def validator_label(cls, v):
        return v  # TODO : check if label in columns


class ModelConfig(BaseModelConfig):  # TODO : put in model module
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


class RegressionConfig(TrainingConfig):
    """
    Config class for the training of a regression model.

    The user must specified at least the following arguments:
    - caps_directory
    - preprocessing_json
    - tsv_directory
    - output_maps_directory
    """

    data: DataConfig
    model: ModelConfig
    validation: ValidationConfig

    @computed_field
    @property
    def network_task(self) -> Task:
        return Task.REGRESSION
