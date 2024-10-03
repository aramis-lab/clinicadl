from logging import getLogger
from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic.types import NonNegativeInt

# from clinicadl.maps_manager.maps_manager import MapsManager
from clinicadl.splitter.split_utils import find_splits

logger = getLogger("clinicadl.cross_validation_config")
