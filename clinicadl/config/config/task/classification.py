from enum import Enum
from logging import getLogger
from typing import Tuple

from pydantic import computed_field, field_validator

from clinicadl.config.config import DataConfig as BaseDataConfig
from clinicadl.config.config import ModelConfig as BaseModelConfig
from clinicadl.config.config import ValidationConfig as BaseValidationConfig
from clinicadl.train.trainer import TrainingConfig
from clinicadl.utils.enum import ClassificationLoss, ClassificationMetric, Task

logger = getLogger("clinicadl.classification_config")


class DataConfig(BaseDataConfig):  # TODO : put in data module
    """Config class to specify the data in classification mode."""

    label: str = "diagnosis"

    @field_validator("label")
    def validator_label(cls, v):
        return v  # TODO : check if label in columns

    @field_validator("label_code")
    def validator_label_code(cls, v):
        return v  # TODO : check label_code


class ModelConfig(BaseModelConfig):  # TODO : put in model module
    """Config class for classification models."""

    architecture: str = "Conv5_FC3"
    loss: ClassificationLoss = ClassificationLoss.CrossEntropyLoss
    selection_threshold: float = 0.0

    @field_validator("architecture")
    def validator_architecture(cls, v):
        return v  # TODO : connect to network module to have list of available architectures

    @field_validator("selection_threshold")
    def validator_threshold(cls, v):
        assert (
            0 <= v <= 1
        ), f"selection_threshold must be between 0 and 1 but it has been set to {v}."
        return v


class ValidationConfig(BaseValidationConfig):
    """Config class for the validation procedure in classification mode."""

    selection_metrics: Tuple[ClassificationMetric, ...] = (ClassificationMetric.LOSS,)

    @field_validator("selection_metrics", mode="before")
    def list_to_tuples(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v


class ClassificationConfig(TrainingConfig):
    """
    Config class for the training of a classification model.

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
        return Task.CLASSIFICATION
