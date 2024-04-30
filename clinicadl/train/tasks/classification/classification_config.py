from enum import Enum
from logging import getLogger
from typing import Dict, Tuple

from pydantic import PrivateAttr, field_validator

from clinicadl.train.tasks import BaseTaskConfig, Task

logger = getLogger("clinicadl.classification_config")


class ClassificationLoss(str, Enum):
    """Available classification losses in ClinicaDL."""

    CrossEntropyLoss = "CrossEntropyLoss"
    MultiMarginLoss = "MultiMarginLoss"


class ClassificationMetric(str, Enum):
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


class ClassificationConfig(BaseTaskConfig):
    """Config class to handle parameters of the classification task."""

    architecture: str = "Conv5_FC3"
    loss: ClassificationLoss = ClassificationLoss.CrossEntropyLoss
    label: str = "diagnosis"
    label_code: Dict[str, int] = {}
    selection_threshold: float = 0.0
    selection_metrics: Tuple[ClassificationMetric, ...] = (ClassificationMetric.LOSS,)
    # private
    _network_task: Task = PrivateAttr(default=Task.CLASSIFICATION)

    @field_validator("selection_metrics", mode="before")
    def list_to_tuples(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("selection_threshold")
    def validator_threshold(cls, v):
        assert (
            0 <= v <= 1
        ), f"selection_threshold must be between 0 and 1 but it has been set to {v}."
        return v

    @field_validator("architecture")
    def validator_architecture(cls, v):
        return v  # TODO : connect to network module to have list of available architectures

    @field_validator("label")
    def validator_label(cls, v):
        return v  # TODO : check if label in columns

    @field_validator("label_code")
    def validator_label_code(cls, v):
        return v  # TODO : check label_code
