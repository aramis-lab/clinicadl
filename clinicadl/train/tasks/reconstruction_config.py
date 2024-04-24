from enum import Enum
from logging import getLogger
from typing import List, Tuple

from pydantic import PrivateAttr, field_validator

from .base_training_config import BaseTaskConfig

logger = getLogger("clinicadl.reconstruction_config")


class Normalization(str, Enum):
    """Available normalization layers in ClinicaDL."""

    BATCH = "batch"
    GROUP = "group"
    INSTANCE = "instance"


class ReconstructionConfig(BaseTaskConfig):
    """Config class to handle parameters of the reconstruction task."""

    loss: str = "MSELoss"
    selection_metrics: Tuple[str, ...] = ("loss",)
    # model
    architecture: str = "AE_Conv5_FC3"
    latent_space_size: int = 128
    feature_size: int = 1024
    n_conv: int = 4
    io_layer_channels: int = 8
    recons_weight: int = 1
    kl_weight: int = 1
    normalization: Normalization = Normalization.BATCH
    # private
    _network_task: str = PrivateAttr(default="reconstruction")

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
