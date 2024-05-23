from enum import Enum
from logging import getLogger
from typing import Tuple

from pydantic import computed_field, field_validator

from clinicadl.train.trainer import DataConfig as BaseDataConfig
from clinicadl.train.trainer import ModelConfig as BaseModelConfig
from clinicadl.train.trainer import Task, TrainingConfig
from clinicadl.train.trainer import ValidationConfig as BaseValidationConfig

logger = getLogger("clinicadl.classification_config")


class ClassificationLoss(str, Enum):  # TODO : put in loss module
    """Available classification losses in ClinicaDL."""

    CrossEntropyLoss = "CrossEntropyLoss"
    MultiMarginLoss = "MultiMarginLoss"


class ClassificationMetric(str, Enum):  # TODO : put in metric module
    """Available classification metrics in ClinicaDL."""

    BA = "BA"
    ACCURACY = "accuracy"
    F1_SCORE = "F1_score"
    SENSITIVITY = "sensitivity"
    SPECIFICITY = "specificity"
    PPV = "PPV"
    NPV = "NPV"
    MCC = "MCC"
    MK = "MK"
    LR_PLUS = "LR_plus"
    LR_MINUS = "LR_minus"
    LOSS = "loss"


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
