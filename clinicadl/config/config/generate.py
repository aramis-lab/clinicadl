from enum import Enum
from logging import getLogger
from pathlib import Path
from time import time
from typing import Annotated, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, field_validator

from clinicadl.utils.enum import (
    Pathology,
    Preprocessing,
    SUVRReferenceRegions,
    Tracer,
)
from clinicadl.utils.exceptions import ClinicaDLTSVError

logger = getLogger("clinicadl.predict_config")


class GenerateConfig(BaseModel):
    generated_caps_directory: Path
    n_subjects: int = 300
    n_proc: int = 1

    # pydantic config
    model_config = ConfigDict(validate_assignment=True)
