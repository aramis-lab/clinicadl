from logging import getLogger
from typing import Tuple

from pydantic import BaseModel, ConfigDict
from pydantic.types import NonNegativeInt

logger = getLogger("clinicadl.caps_dataset")


class CapsDatasetConfig(BaseModel):
    """
    Abstract config class for the validation procedure.


    """

    # TODO add option

    # pydantic config
    model_config = ConfigDict(validate_assignment=True)
