from enum import Enum
from logging import getLogger
from typing import Tuple

from pydantic import computed_field, field_validator

from clinicadl.train.trainer import DataConfig as BaseDataConfig
from clinicadl.train.trainer import ModelConfig as BaseModelConfig
from clinicadl.train.trainer import Task, TrainingConfig
from clinicadl.train.trainer import ValidationConfig as BaseValidationConfig

logger = getLogger("clinicadl.regression_config")


class RegressionLoss(str, Enum):  # TODO : put in loss module
    """Available regression losses in ClinicaDL."""

    L1Loss = "L1Loss"
    MSELoss = "MSELoss"
    KLDivLoss = "KLDivLoss"
    BCEWithLogitsLoss = "BCEWithLogitsLoss"
    HuberLoss = "HuberLoss"
    SmoothL1Loss = "SmoothL1Loss"


class RegressionMetric(str, Enum):  # TODO : put in metric module
    """Available regression metrics in ClinicaDL."""

    R2_score = "R2_score"
    MAE = "MAE"
    RMSE = "RMSE"
    LOSS = "loss"


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
