from logging import getLogger
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic.types import NonNegativeInt

from clinicadl.config.config import (
    ComputationalConfig,
    CrossValidationConfig,
    DataConfig,
    DataLoaderConfig,
    MapsManagerConfig,
    PredictConfig,
    ValidationConfig,
)

logger = getLogger("clinicadl.training_config")


class PredictPipelineConfig(
    MapsManagerConfig,
    PredictConfig,
    DataConfig,
    ValidationConfig,
    CrossValidationConfig,
    ComputationalConfig,
    DataLoaderConfig,
):
    """Config class to perform Transfer Learning."""
