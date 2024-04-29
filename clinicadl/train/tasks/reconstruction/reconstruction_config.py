from enum import Enum
from logging import getLogger
from typing import Tuple

from pydantic import PrivateAttr, field_validator

from clinicadl.train.tasks import BaseTaskConfig

logger = getLogger("clinicadl.reconstruction_config")


class ReconstructionLoss(str, Enum):
    """Available reconstruction losses in ClinicaDL."""

    L1Loss = "L1Loss"
    MSELoss = "MSELoss"
    KLDivLoss = "KLDivLoss"
    BCEWithLogitsLoss = "BCEWithLogitsLoss"
    HuberLoss = "HuberLoss"
    SmoothL1Loss = "SmoothL1Loss"
    VAEGaussianLoss = "VAEGaussianLoss"
    VAEBernoulliLoss = "VAEBernoulliLoss"
    VAEContinuousBernoulliLoss = "VAEContinuousBernoulliLoss"


class Normalization(str, Enum):
    """Available normalization layers in ClinicaDL."""

    BATCH = "batch"
    GROUP = "group"
    INSTANCE = "instance"


class ReconstructionConfig(BaseTaskConfig):
    """Config class to handle parameters of the reconstruction task."""

    loss: ReconstructionLoss = ReconstructionLoss.MSELoss
    selection_metrics: Tuple[str, ...] = (
        "loss",
    )  # TODO : enum class for this parameter
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

    @field_validator("architecture")
    def validator_architecture(cls, v):
        return v  # TODO : connect to network module to have list of available architectures
