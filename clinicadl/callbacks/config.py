from logging import getLogger
from typing import Optional

from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.enum import ExperimentTracking

logger = getLogger("clinicadl.callbacks_config")


class CallbacksConfig(ClinicaDLConfig):
    """Config class to add callbacks to the training."""

    emissions_calculator: bool = False
    track_exp: Optional[ExperimentTracking] = None
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)
