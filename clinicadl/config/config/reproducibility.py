from logging import getLogger
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from clinicadl.utils.enum import Compensation

logger = getLogger("clinicadl.reproducibility_config")


class ReproducibilityConfig(BaseModel):
    """Config class to handle reproducibility parameters."""

    compensation: Compensation = Compensation.MEMORY
    deterministic: bool = False
    save_all_models: bool = False
    seed: int = 0
    config_file: Optional[Path] = None
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)
