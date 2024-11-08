# coding: utf8
# TODO: create a folder for generate/ prepare_data/ data to deal with capsDataset objects ?
from logging import getLogger
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from pydantic import BaseModel, ConfigDict

from clinicadl.dataset.config import extraction as extraction
from clinicadl.utils.enum import ExtractionMethod

logger = getLogger("clinicadl")


class CapsDatasetOutput(BaseModel):
    image: torch.Tensor
    participant_id: Union[int, str]
    session_id: Union[int, str]
    label: Optional[Union[float, int]] = None
    image_id: Optional[Union[float, str]] = None
    image_path: Optional[Path] = None
    # domain: Optional[int]=None
    mode: ExtractionMethod

    # pydantic config
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)
