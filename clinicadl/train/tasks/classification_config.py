from logging import getLogger
from typing import Dict, List, Tuple

from pydantic import PrivateAttr, field_validator

from .base_training_config import BaseTaskConfig

logger = getLogger("clinicadl.classification_config")


class ClassificationConfig(BaseTaskConfig):
    """Config class to handle parameters of the classification task."""

    architecture: str = "Conv5_FC3"
    loss: str = "CrossEntropyLoss"
    label: str = "diagnosis"
    label_code: Dict[str, int] = {}
    selection_threshold: float = 0.0
    selection_metrics: Tuple[str, ...] = ("loss",)
    # private
    _network_task: str = PrivateAttr(default="classification")

    @field_validator("selection_metrics", mode="before")
    def list_to_tuples(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v

    @classmethod
    def get_compatible_losses(cls) -> List[str]:
        """To get the list of losses implemented and compatible with this task."""
        compatible_losses = [  # TODO : connect to the Loss module
            "CrossEntropyLoss",
            "MultiMarginLoss",
        ]
        return compatible_losses

    @field_validator("loss")
    def validator_loss(cls, v):
        compatible_losses = cls.get_compatible_losses()
        assert (
            v in compatible_losses
        ), f"Loss '{v}' can't be used for this task. Please choose among: {compatible_losses}"
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
