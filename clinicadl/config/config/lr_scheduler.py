from logging import getLogger

from clinicadl.utils.config import ClinicaDLConfig

logger = getLogger("clinicadl.lr_config")


class LRschedulerConfig(BaseModel):
    """Config class to instantiate an LR Scheduler."""

    adaptive_learning_rate: bool = False
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)
