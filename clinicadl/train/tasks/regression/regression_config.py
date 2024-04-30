from enum import Enum
from logging import getLogger
from typing import Tuple

from pydantic import PrivateAttr, field_validator

from clinicadl.train.tasks import BaseTaskConfig, Task

logger = getLogger("clinicadl.regression_config")


class RegressionLoss(str, Enum):
    """Available regression losses in ClinicaDL."""

    L1Loss = "L1Loss"
    MSELoss = "MSELoss"
    KLDivLoss = "KLDivLoss"
    BCEWithLogitsLoss = "BCEWithLogitsLoss"
    HuberLoss = "HuberLoss"
    SmoothL1Loss = "SmoothL1Loss"


class RegressionMetric(str, Enum):
    """Available regression metrics in ClinicaDL."""

    R2_score = "R2_score"
    MAE = "MAE"
    RMSE = "RMSE"
    LOSS = "loss"


class RegressionConfig(BaseTaskConfig):
    """Config class to handle parameters of the regression task."""

    architecture: str = "Conv5_FC3"
    loss: RegressionLoss = RegressionLoss.MSELoss
    label: str = "age"
    selection_metrics: Tuple[RegressionMetric, ...] = (RegressionMetric.LOSS,)
    # private
    _network_task: Task = PrivateAttr(default=Task.REGRESSION)

    @field_validator("selection_metrics", mode="before")
    def list_to_tuples(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("architecture")
    def validator_architecture(cls, v):
        return v  # TODO : connect to network module to have list of available architectures

    @field_validator("label")
    def validator_label(cls, v):
        return v  # TODO : check if column is in labels
