from enum import Enum
from logging import getLogger
from typing import Dict, Tuple

from pydantic import PrivateAttr, field_validator

from clinicadl.train.tasks import BaseTaskConfig

logger = getLogger("clinicadl.classification_config")


class ClassificationLoss(str, Enum):
    """Available classification losses in ClinicaDL."""

    CrossEntropyLoss = "CrossEntropyLoss"
    MultiMarginLoss = "MultiMarginLoss"


class ClassificationConfig(BaseTaskConfig):
    """Config class to handle parameters of the classification task."""

    architecture: str = "Conv5_FC3"
    loss: ClassificationLoss = ClassificationLoss.CrossEntropyLoss
    label: str = "diagnosis"
    label_code: Dict[str, int] = {}
    selection_threshold: float = 0.0
    selection_metrics: Tuple[str, ...] = (
        "loss",
    )  # TODO : enum class for this parameter
    # private
    _network_task: str = PrivateAttr(default="classification")

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
