from logging import getLogger

from pydantic import BaseModel, ConfigDict
from pydantic.types import NonNegativeFloat, PositiveFloat

from clinicadl.utils.enum import Optimizer

logger = getLogger("clinicadl.optimizer_config")


class OptimizerConfig(BaseModel):
    """Config class to configure the optimizer."""

    learning_rate: PositiveFloat = 1e-4
    optimizer: Optimizer = Optimizer.ADAM
    weight_decay: NonNegativeFloat = 1e-4
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)
