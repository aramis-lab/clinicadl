from logging import getLogger

from pydantic import BaseModel, ConfigDict
from pydantic.types import NonNegativeFloat, NonNegativeInt

logger = getLogger("clinicadl.early_stopping_config")


class EarlyStoppingConfig(BaseModel):
    """Config class to perform Early Stopping."""

    patience: NonNegativeInt = 0
    tolerance: NonNegativeFloat = 0.0
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)
