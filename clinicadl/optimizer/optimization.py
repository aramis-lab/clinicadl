from logging import getLogger

from pydantic import BaseModel, ConfigDict
from pydantic.types import PositiveInt

logger = getLogger("clinicadl.optimization_config")


class OptimizationConfig(BaseModel):
    """Config class to configure the optimization process."""

    accumulation_steps: PositiveInt = 1
    epochs: PositiveInt = 20
    profiler: bool = False
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)
