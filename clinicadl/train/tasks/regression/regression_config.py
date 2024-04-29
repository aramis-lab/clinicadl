from enum import Enum
from logging import getLogger
from typing import Tuple

from pydantic import PrivateAttr, field_validator

from clinicadl.train.tasks import BaseTaskConfig

logger = getLogger("clinicadl.regression_config")


class RegressionLoss(str, Enum):
    """Available regression losses in ClinicaDL."""

    L1Loss = "L1Loss"
    MSELoss = "MSELoss"
    KLDivLoss = "KLDivLoss"
    BCEWithLogitsLoss = "BCEWithLogitsLoss"
    HuberLoss = "HuberLoss"
    SmoothL1Loss = "SmoothL1Loss"


class RegressionConfig(BaseTaskConfig):
    """Config class to handle parameters of the regression task."""

    architecture: str = "Conv5_FC3"
    loss: RegressionLoss = RegressionLoss.MSELoss
    label: str = "age"
    selection_metrics: Tuple[str, ...] = (
        "loss",
    )  # TODO : enum class for this parameter
    # private
    _network_task: str = PrivateAttr(default="regression")

    @field_validator("selection_metrics", mode="before")
    def list_to_tuples(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("architecture")
    def validator_architecture(cls, v):
        return v  # TODO : connect to network module to have list of available architectures
