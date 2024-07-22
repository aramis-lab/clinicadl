from logging import getLogger
from typing import Optional

from pydantic import BaseModel, ConfigDict

from clinicadl.utils.enum import ExperimentTracking

logger = getLogger("clinicadl.callbacks_config")


class CallbacksConfig(BaseModel):
    """Config class to add callbacks to the training."""

    emissions_calculator: bool = False
    track_exp: Optional[ExperimentTracking] = None
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)
