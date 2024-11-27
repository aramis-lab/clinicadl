# coding: utf8
# TODO: create a folder for generate/ prepare_data/ data to deal with capsDataset objects ?
from logging import getLogger
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict

from clinicadl.dataset.config import preprocessing
from clinicadl.dataset.transforms import extraction
from clinicadl.dataset.transforms.transforms import Transforms
from clinicadl.utils.enum import ExtractionMethod, Preprocessing
from clinicadl.utils.iotools.utils import read_preprocessing

logger = getLogger("clinicadl")
PARTICIPANT_ID = "participant_id"
SESSION_ID = "session_id"


def get_extraction(
    extract_method: Union[str, ExtractionMethod],
) -> type[extraction.BaseExtraction]:
    extract_method = ExtractionMethod(extract_method)
    if extract_method == ExtractionMethod.ROI:
        return extraction.ROI
    elif extract_method == ExtractionMethod.SLICE:
        return extraction.Slice
    elif extract_method == ExtractionMethod.IMAGE:
        return extraction.Image
    elif extract_method == ExtractionMethod.PATCH:
        return extraction.Patch
    else:
        raise ValueError(f"Preprocessing {extract_method.value} is not implemented.")


def get_preprocessing(
    preprocessing_type: Union[str, Preprocessing],
) -> type[preprocessing.PreprocessingConfig]:
    preprocessing_type = Preprocessing(preprocessing_type)
    if preprocessing_type == Preprocessing.T1_LINEAR:
        return preprocessing.PreprocessingT1
    elif preprocessing_type == Preprocessing.PET_LINEAR:
        return preprocessing.PreprocessingPET
    elif preprocessing_type == Preprocessing.FLAIR_LINEAR:
        return preprocessing.PreprocessingFlair
    elif preprocessing_type == Preprocessing.CUSTOM:
        return preprocessing.PreprocessingCustom
    elif preprocessing_type == Preprocessing.DWI_DTI:
        return preprocessing.PreprocessingDTI
    else:
        raise ValueError(
            f"Preprocessing {preprocessing_type.value} is not implemented."
        )


def get_infos_from_json(
    json_path: Path,
) -> Tuple[preprocessing.PreprocessingConfig, extraction.BaseExtraction, Transforms]:
    """
    Extracts the preprocessing and mode from a json file.

    Parameters
    ----------
    json_path : Path
        Path to the json file containing the preprocessing and mode.

    Returns
    -------
    Tuple[Preprocessing, SliceMode]
        The preprocessing and mode extracted from the json file.
    """

    dict_ = read_preprocessing(json_path)
    return get_infos_from_parameters(**dict_)


def get_infos_from_parameters(
    **kwargs,
) -> Tuple[preprocessing.PreprocessingConfig, extraction.BaseExtraction, Transforms]:
    """
    Extracts the preprocessing and mode from a json file.

    Returns
    -------
    Tuple[Preprocessing, SliceMode]
        The preprocessing and mode extracted from the json file.
    """

    if "preprocessing_dict" in kwargs:
        kwargs = kwargs["preprocessing_dict"]

    preprocessing = Preprocessing(kwargs["preprocessing"])
    mode = ExtractionMethod(kwargs["extract_method"])
    extraction = get_extraction(mode)(**kwargs)
    transforms = Transforms(extraction=extraction, **kwargs)
    return get_preprocessing(preprocessing)(**kwargs), extraction, transforms
