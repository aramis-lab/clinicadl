from logging import getLogger
from typing import List, Tuple

from pydantic import computed_field, field_validator

from .base_training_config import BaseTaskConfig

logger = getLogger("clinicadl.reconstruction_config")


class ReconstructionConfig(BaseTaskConfig):
    """Config class to handle parameters of the reconstruction task."""

    architecture: str = "AE_Conv5_FC3"
    loss: str = "CrossEntropyLoss"
    selection_metrics: Tuple[str, ...] = ("loss",)

    @computed_field
    def _network_task(self) -> str:
        """To have a task field that is immutable."""
        return "reconstruction"

    @field_validator("selection_metrics", mode="before")
    def list_to_tuples(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v

    @classmethod
    def get_compatible_losses(cls) -> List[str]:
        """To get the list of losses implemented and compatible with this task."""
        compatible_losses = [  # TODO : connect to the Loss module
            "L1Loss",
            "MSELoss",
            "KLDivLoss",
            "BCEWithLogitsLoss",
            "HuberLoss",
            "SmoothL1Loss",
            "VAEGaussianLoss",
            "VAEBernoulliLoss",
            "VAEContinuousBernoulliLoss",
        ]
        return compatible_losses

    @field_validator("loss")
    def validator_loss(cls, v):
        compatible_losses = cls.get_compatible_losses()
        assert (
            v in compatible_losses
        ), f"Loss '{v}' can't be used for this task. Please choose among: {compatible_losses}"
        return v

    @field_validator("architecture")
    def validator_architecture(cls, v):
        return v  # TODO : connect to network module to have list of available architectures
