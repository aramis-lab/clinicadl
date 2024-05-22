from logging import getLogger

from pydantic import BaseModel, ConfigDict

logger = getLogger("clinicadl.lr_config")


class LRschedulerConfig(BaseModel):
    """Config class to instantiate an LR Scheduler."""

    adaptive_learning_rate: bool = False
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)
