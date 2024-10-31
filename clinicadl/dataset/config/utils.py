# coding: utf8
# TODO: create a folder for generate/ prepare_data/ data to deal with capsDataset objects ?
from logging import getLogger
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from pydantic import BaseModel, ConfigDict

from clinicadl.dataset.config import extraction, preprocessing
from clinicadl.utils.enum import ExtractionMethod, Preprocessing
from clinicadl.utils.iotools.utils import read_json

logger = getLogger("clinicadl")


def get_extraction(
    extract_method: Union[str, ExtractionMethod],
) -> type[extraction.ALL_EXTRACTION_TYPES]:
    extract_method = ExtractionMethod(extract_method)
    if extract_method == ExtractionMethod.ROI:
        return extraction.ExtractionROIConfig
    elif extract_method == ExtractionMethod.SLICE:
        return extraction.ExtractionSliceConfig
    elif extract_method == ExtractionMethod.IMAGE:
        return extraction.ExtractionImageConfig
    elif extract_method == ExtractionMethod.PATCH:
        return extraction.ExtractionPatchConfig
    else:
        raise ValueError(f"Preprocessing {extract_method.value} is not implemented.")


def get_preprocessing(
    preprocessing_type: Union[str, Preprocessing],
) -> type[preprocessing.ALL_PREPROCESSING_TYPES]:
    preprocessing_type = Preprocessing(preprocessing_type)
    if preprocessing_type == Preprocessing.T1_LINEAR:
        return preprocessing.T1PreprocessingConfig
    elif preprocessing_type == Preprocessing.PET_LINEAR:
        return preprocessing.PETPreprocessingConfig
    elif preprocessing_type == Preprocessing.FLAIR_LINEAR:
        return preprocessing.FlairPreprocessingConfig
    elif preprocessing_type == Preprocessing.CUSTOM:
        return preprocessing.CustomPreprocessingConfig
    elif preprocessing_type == Preprocessing.DWI_DTI:
        return preprocessing.DTIPreprocessingConfig
    else:
        raise ValueError(
            f"Preprocessing {preprocessing_type.value} is not implemented."
        )


def get_preprocessing_and_mode_from_json(json_path: Path):
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

    dict_ = read_json(json_path)
    return get_preprocessing_and_mode_from_parameters(**dict_)


def get_preprocessing_and_mode_from_parameters(**kwargs):
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
    return get_preprocessing(preprocessing)(**kwargs), get_extraction(mode)(**kwargs)
